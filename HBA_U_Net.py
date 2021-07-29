from Import import *
from Utility import *

class HBA(keras.layers.MultiHeadAttention):
    def __init__(self, num_heads=4, bottleneck_dimension=512, relative=True, **kwargs):
        self.key_dim = bottleneck_dimension // num_heads
        super(HBA, self).__init__(num_heads=num_heads, key_dim=self.key_dim, **kwargs)
        self.num_heads, self.bottleneck_dimension, self.relative = num_heads, bottleneck_dimension, relative

    def _build_from_signature(self, featuremap):
        super(HBA, self)._build_from_signature(query=featuremap, value=featuremap)
        _, hh, ww, _ = featuremap.shape
        stddev = self.key_dim ** -0.5
        self.rel_emb_w = self.add_weight(
            name="r_width",
            shape=(self.key_dim, 2 * ww - 1),
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
            dtype=featuremap.dtype,
        )
        self.rel_emb_h = self.add_weight(
            name="r_height",
            shape=(self.key_dim, 2 * hh - 1),
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
            dtype=featuremap.dtype,
        )

    def get_config(self):
        base_config = super(HBA, self).get_config()
        base_config.pop("key_dim", None)
        base_config.update(
            {"num_heads": self.num_heads, "bottleneck_dimension": self.bottleneck_dimension, "relative": self.relative}
        )
        return base_config

    def rel_to_abs(self, rel_pos):
        _, heads, hh, ww, dim = rel_pos.shape
        col_pad = tf.zeros_like(rel_pos[:, :, :, :, :1], dtype=rel_pos.dtype)
        rel_pos = tf.concat([rel_pos, col_pad], axis=-1)
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * 2 * ww])
        flat_pad = tf.zeros_like(flat_x[:, :, :, : ww - 1], dtype=rel_pos.dtype)
        flat_x_padded = tf.concat([flat_x, flat_pad], axis=-1)
        final_x = tf.reshape(flat_x_padded, [-1, heads, hh, ww + 1, 2 * ww - 1])
        final_x = final_x[:, :, :, :ww, ww - 1 :]
        return final_x

    def relative_logits_1d(self, query, rel_k, transpose_mask):
        _, _, hh, _, _ = query.shape
        rel_logits = tf.matmul(query, rel_k)
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = tf.expand_dims(rel_logits, axis=3)
        rel_logits = tf.tile(rel_logits, [1, 1, 1, hh, 1, 1])
        rel_logits = tf.transpose(rel_logits, transpose_mask)
        return rel_logits

    def relative_logits(self, query):
        query = tf.transpose(query, [0, 3, 1, 2, 4])
        rel_logits_w = self.relative_logits_1d(query=query, rel_k=self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        query = tf.transpose(query, [0, 1, 3, 2, 4])
        rel_logits_h = self.relative_logits_1d(query=query, rel_k=self.rel_emb_h, transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def channel_attn(self, value, output, ratio=8):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = value.shape[channel_axis]
        shared_layer_one = Dense(channel//ratio, activation='relu',kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
        shared_layer_two = Dense(channel,kernel_initializer='he_normal',use_bias=True,bias_initializer='zeros')
        avg_pool = GlobalAveragePooling2D()(value)
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        max_pool = GlobalMaxPooling2D()(value)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        channel_feature = Add()([avg_pool,max_pool])
        channel_feature = Activation('sigmoid')(channel_feature)

        if K.image_data_format() == "channels_first":
            channel_feature = Permute((3, 1, 2))(channel_feature)
        return multiply([output, channel_feature])
    
    def call(self, inputs, attention_mask=None, return_attention_scores=False, training=None):
        if not self._built_from_signature:
            self._build_from_signature(featuremap=inputs)

        query = self._query_dense(inputs)
        key = self._key_dense(inputs)
        value = self._value_dense(inputs)

        query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))
        attention_scores = special_math_ops.einsum(self._dot_product_equation, key, query)
        if self.relative:
            attention_scores += self.relative_logits(query)
        
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = special_math_ops.einsum(self._combine_equation, attention_scores_dropout, value)

        hh, ww = inputs.shape[1], inputs.shape[2]
        attention_output = tf.reshape(attention_output, [-1, hh, ww, self.num_heads * self.key_dim])
        attention_output = self.channel_attn(inputs,attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

def HBA_block(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    pos_enc_type="relative",
    strides=1,
    target_dimension=2048,
    name="HBA",
):
    if strides != 1 or featuremap.shape[-1] != target_dimension:
        shortcut = conv2d_no_bias(featuremap, target_dimension, 1, strides=strides, name=name + "_0_")
        shortcut = batchnorm_with_activation(shortcut, activation=activation, zero_gamma=False, name=name + "_0_")
    else:
        shortcut = featuremap

    bottleneck_dimension = target_dimension // proj_factor
    nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=1, name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")

    nn = HBA(num_heads=heads, bottleneck_dimension=bottleneck_dimension, name=name + "_2_mhsa")(nn)
    if strides != 1:
        nn = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(nn)
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_2_")

    nn = conv2d_no_bias(nn, target_dimension, 1, strides=1, name=name + "_3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "_3_")

    nn = layers.Add(name=name + "_add")([shortcut, nn])
    return layers.Activation(activation, name=name + "_out")(nn)


def HBA_stack(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    pos_enc_type="relative",
    name="HBA_stack",
    strides=2,
    num_layers=3,
    target_dimension=1024, #2048
):
    """Use `activation=swish` for `silu` """
    for i in range(num_layers):
        featuremap = HBA_block(
            featuremap,
            heads=heads,
            proj_factor=proj_factor,
            activation=activation,
            pos_enc_type=pos_enc_type,
            strides=strides if i == 0 else 1,
            target_dimension=target_dimension,
            name=name + "_{}".format(i),
        )
    return featuremap

def convert(model, include_top=True, classes=1000, num_layers=3, strides=2, activation="relu", **kwargs):
    add_layer_count = 0
    for idx, layer in enumerate(model.layers[::-1]):
        if isinstance(layer, keras.layers.Add):
            add_layer_count += 1
        if add_layer_count == num_layers + 1:
            break

    inputs = model.inputs[0]
    nn = model.layers[-idx - 1 + 1].output  
    nn = HBA_stack(nn, strides=strides, activation=activation, **kwargs)

    if include_top:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(classes, activation="softmax", name="predictions")(nn)
    return keras.models.Model(inputs, nn)


def Encoder(include_top=True, input_tensor=None, input_shape=None, classes=1000, num_layers=3, strides=2, activation="relu", **kwargs):
    mm = keras.applications.ResNet50(include_top=False, weights = None, input_tensor=input_tensor, input_shape=input_shape, **kwargs)
    return convert(mm, include_top=include_top, classes=classes, strides=strides, activation=activation, num_layers=num_layers)

def HBA_skip(
    featuremap,
    heads=4,
    activation="relu",
    pos_enc_type="relative",
    strides=2,
    target_dimension=512,
    name="HBA_skip"
):
    nn = layers.AveragePooling2D(pool_size=strides, strides=strides, padding="same")(featuremap)
    nn = HBA(num_heads=heads, bottleneck_dimension=target_dimension, name=name + "_mhsa")(nn)
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "_3_")
    nn = UpSampling2D(size=strides)(nn)

    return nn

def HBA_U_Net(input_shape, dropout_rate = 0.4, use_attnDecoder = False, skip = False, num_layers = 3):
    resnet_base = Encoder(
      include_top=True,
      classes = 2,
      input_shape = (512,512,3),
      num_layers = num_layers
    )

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("conv1_relu").output 
    conv2 = resnet_base.get_layer("conv2_block3_out").output 
    conv3 = resnet_base.get_layer("conv3_block4_out").output 
    conv4 = resnet_base.get_layer("conv4_block6_out").output 
    conv5 = resnet_base.get_layer("HBA_stack_"+str(num_layers-1)+"_out").output 
    

    if skip:
        attn_3 = HBA_skip(
          conv3,
          heads=4, 
          activation="relu",
          pos_enc_type="relative",
          strides=2,
          target_dimension=512,
          name="HBA_enco3",
        )
        attn_2 = HBA_skip(
          conv2,
          heads=4,
          activation="relu",
          pos_enc_type="relative",
          strides=2,
          target_dimension=256,
          name="HBA_enco2",
        )
        attn_1 = HBA_skip(
          conv1,
          heads=4,
          activation="relu",
          pos_enc_type="relative",
          strides=4,
          target_dimension=64,
          name="HBA_enco1",
        )

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    up6 = Dropout(dropout_rate)(up6)
    conv6 = conv_block_simple(up6, 256, "conv6_1") 
    conv6 = Dropout(dropout_rate)(conv6)
    conv6 = conv_block_simple(conv6, 256, "conv6_2") 
    conv6 = Dropout(dropout_rate)(conv6)

    if skip:
        up7 = concatenate([UpSampling2D()(conv6), conv3, attn_3], axis=-1)
    else:
        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    up7 = Dropout(dropout_rate)(up7)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = Dropout(dropout_rate)(conv7)
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = Dropout(dropout_rate)(conv7)

    if skip:
        up8 = concatenate([UpSampling2D()(conv7), conv2, attn_2], axis=-1)
    else:
        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    up8 = Dropout(dropout_rate)(up8)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = Dropout(dropout_rate)(conv8)
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = Dropout(dropout_rate)(conv8)

    if skip:
        up9 = concatenate([UpSampling2D()(conv8), conv1, attn_1], axis=-1)
    else:
        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    up9 = Dropout(dropout_rate)(up9)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = Dropout(dropout_rate)(conv9)
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = Dropout(dropout_rate)(conv9)

    up10 = UpSampling2D()(conv9)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)

    model = Model(inputs=resnet_base.input, outputs=x)
    
    return model
