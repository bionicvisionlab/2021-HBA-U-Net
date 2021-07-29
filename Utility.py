from Import import *

class_names = ['AMD', 'NON-AMD']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)

IMAGE_SIZE = (512, 512)
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, name=""):
    padding = "SAME" if strides == 1 else "VALID"
    return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name + "conv")(inputs)

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv

def create_dir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if os.path.isdir(mypath):
            pass
        else:
            raise


def plot_loss(loss, label, filename, log_dir, acc=None, title='', ylim=None):
    """Plot a loss function and save it in a file."""
    loss = np.array(loss)
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        if acc is None:
            plt.ylim((0, 0.5))
        else:
            plt.ylim((0,1.))

    plt.title(title)
    plt.savefig(os.path.join(log_dir, filename))
    plt.clf()
    plt.close('all')

def resizefile(pathname,dirname,savepathname):
    
    for item in sorted(dirname):
        if item == '.DS_Store':
             continue
        if os.path.isfile(pathname+item):
        
            img = Image.open(pathname+item)
            x,y = img.size
            print(x)
            Ratio.append([x,y])
            f, e = os.path.splitext(item)
            imResize = img.resize((512,512),Image.ANTIALIAS)
            
            imResize.save(savepathname + f + ' resized.jpg', 'JPEG')

def get_dist_maps(coords, shp=(512,512)):
    fx, fy = coords[0]
    

    distance = np.ones(shp)
    distance[fy, fx] = 0
    
    distance = distance_transform_edt(distance)
    distance = distance[:,:,np.newaxis]
    if shp != (512,512):
        distance=resize(1 - distance / np.max(distance), (512,512,3)) ** 7
    else:
        distance = (1 - distance / np.max(distance)) ** 7
    return distance

"""
Iterator to load images from the datasets, and related functions.
"""
def normalize_for_tanh(batch):
    """Make input image values lie between -1 and 1."""
    tanh_batch = batch - np.max(batch)/2.
    tanh_batch /= np.max(batch)/2.
    return tanh_batch

class TwoImageIterator(Iterator):
    """Class to iterate A and B images at the same time, while applying desired
    transformations online."""

    def __init__(self, directory, a_dir_name='A', b_dir_name=None, N=-1,
                 batch_size=32, shuffle=True, seed=None, target_size=(512,512),
                 cspace='rgb', nch_gdt=1,
                 zscore=True, normalize_tanh=False,
                 return_mode='normal', decay=5, dataset='idrid',
                 rotation_range=0., height_shift_range=0., shear_range=0.,
                 width_shift_range=0., zoom_range=0., fill_mode='constant',
                 cval=0., horizontal_flip=False, vertical_flip=False, transform=None):

        """
        Iterate through the image directoriy, apply transformations and return
        distance map calculated on the fly. If b_dir_name is not None, it will
        retrieve the ground truth from the directory.

        Files under the directory A and B will be returned at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator;
        - return_mode: 'normal', 'fnames'. Default: 'normal'
            - 'normal' returns: [batch_a, batch_b]
            - 'fnames' returns: [batch_a, batch_b, files]
        - decay: decay at which to compute de distance map. Default: 5
        - dataset: dataset to load. Can handle Messidor and Idrid. Default: Idrid

        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.a_fnames = sorted(os.listdir(self.a_dir))

        self.b_dir_name = b_dir_name
        if b_dir_name is not None:
            self.b_dir = os.path.join(directory, b_dir_name)
            self.b_fnames = sorted(os.listdir(self.b_dir))

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            self.filenames = self.a_fnames[:N]
        self.N = len(self.a_fnames)

        self.ch_order = 'tf'

        # Preprocess images
        self.cspace = cspace #colorspace

        # Image shape
        self.target_size = target_size
        self.nch_gdt = nch_gdt

        self.nch = len(self.cspace) # for example if grayscale

        #self.select_vessels = select_vessels

        self.img_shape_a = self._get_img_shape(self.target_size, ch=self.nch)
        self.img_shape_b = self._get_img_shape(self.target_size, ch=self.nch_gdt)
       

        if self.ch_order == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2
        else:
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3

        #Normalizations
        self.normalize_tanh = normalize_tanh
        self.zscore = zscore

        # Transformations
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        self.transform = transform

        self.return_mode = return_mode
        self.decay=decay
        self.dataset = dataset

        super(TwoImageIterator, self).__init__(len(self.a_fnames), batch_size,
                                               shuffle, seed)

    def _get_img_shape(self, size, ch=3):

        if self.ch_order == 'tf':
            img_shape = size + (ch,)
        else:
            img_shape = (ch,) + size

        return img_shape

    def _load_img_pair(self, idx):
        """
        Load images and apply pre-processing
        :param idx: index of file to load in the list of names
        :return: aa: image
                 bb: ground truth
        """
        #print(self.a_fnames[276])
        #print(len(self.a_fnames),len(self.b_fnames))
        
        
        aa = cv2.imread(os.path.join(self.a_dir, self.a_fnames[idx]))
        aa = cv2.cvtColor(aa, cv2.COLOR_BGR2RGB)
        bb = cv2.imread(os.path.join(self.b_dir, self.b_fnames[idx]))
        bb = cv2.cvtColor(bb, cv2.COLOR_BGR2RGB)
        #print(self.a_fnames[idx]+" "+self.b_fnames[idx])
        
        if self.nch_gdt == 3:
            # fix for the case when the .png has an alpha channel
            if bb.shape[-1] == 4:
                bb = bb[:,:,:3]
        elif self.nch_gdt == 1:
            # fix for the case when the .png has an alpha channel
            
            if len(bb.shape) == 3:
                bb = rgb2gray(bb)
                bb = bb.reshape((512,512,1))
                
        #if self.select_vessels is True:
            #bb = self.select_vessel_width(bb)
        
        return aa, bb

    def _random_transform(self, a, b, is_batch=False):
        if is_batch is False:
        # a and b are single images, so they don't have image number at index 0
            img_row_index = self.row_index - 1
            img_col_index = self.col_index - 1
            img_channel_index = self.channel_index - 1
        else:
            img_row_index = self.row_index
            img_col_index = self.col_index
            img_channel_index = self.channel_index
        """
        New augumentation implement

        """
        if self.transform is not None:
          a = self.transform(image=a)['image']
          # b = self.transform(image=b)['image'] #no need to transform ground truth img

        return a, b

    def get_dist_maps(self, coords, shp=(512,512)):
        fx, fy = coords[0]
    

        distance = np.ones(shp)
        distance[fy, fx] = 0
    
        distance = distance_transform_edt(distance)
        distance = distance[:,:,np.newaxis]
        if shp != (512,512):
            distance=resize(1 - distance / np.max(distance), (512,512,3)) ** 7
        else:
            distance = (1 - distance / np.max(distance)) ** 7
        return distance

    def next(self):
        """Get the next pair of the sequence."""

        # Lock the iterator when the index is changed.
        with self.lock:
            index_array = next(self.index_generator)
        current_batch_size = len(index_array)
            
        # Initialize the arrays according to the size of the output images
       
        batch_a = np.zeros((current_batch_size,) + self.img_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.img_shape_b[:-1]
                           + (self.nch_gdt,))

        files = []
        ind = []


        # Load images and apply transformations
        for i, j in enumerate(index_array):
            
            if self.a_fnames[j] == ".DS_Store":
                continue
            
            im_id = self.a_fnames[j][:-4]
            a_img, b_img = self._load_img_pair(j)

            #Transform
            a_img, b_img = self._random_transform(a_img, b_img)

            # #NORMALIZE
            if self.zscore is True:
                a_img = (a_img - a_img.mean()) / (a_img.std())        

            batch_a[i] = a_img
            batch_b[i] = b_img

            files.append(self.a_fnames[j])

        # when using tanh activation the inputs must be between [-1 1]
        if self.normalize_tanh is True and self.zscore is False:
            batch_a = normalize_for_tanh(batch_a)
            batch_b = normalize_for_tanh(batch_b)

        if self.return_mode == 'normal':
            return [batch_a, batch_b]

        elif self.return_mode == 'fnames':
            return [batch_a, batch_b, files]

def check_EarlyStop(vloss, tr_loss, patience=5):

    Eopt = np.min(vloss[:-1])
    GL = ((vloss[-1] / Eopt) - 1)
    Pk = (np.sum(tr_loss[-patience:]) / (
            patience * (np.min(tr_loss[-patience:]))))
    PQ = GL / Pk

    # return [PQ, GL, Pk]
    if (GL > 0.):
        if PQ > 0.5:
            return 'early_stop'
        elif Pk < 1.1:
            return 'early_stop'
        else:
            return 'pass'
    else:
        return 'pass'