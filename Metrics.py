from Import import *
from Utility import *

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    print((2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    # ipdb.set_trace()
    # return 1-dice_coef(y_true, y_pred)
    # return tf.reduce_sum(2*(1-dice_coef(y_true[:,:,175:,:], y_pred[:,:,175:,:])))\
    #  + tf.reduce_sum(1-dice_coef(y_true[:,:,:175,:], y_pred[:,:,:175,:])) 
     #fovea weights twice as the OD
    
    Fovea_loss = 0.7*(1-dice_coef(y_true[:,:,175:,:], y_pred[:,:,175:,:]))
    print(Fovea_loss)
    OD_loss = 0.3*(1-dice_coef(y_true[:,:,:175,:], y_pred[:,:,:175,:]))
    return Fovea_loss

def total_loss(y_true, y_pred):
  mse = tf.keras.losses.MeanSquaredError()
  mse_loss = mse(y_true, y_pred)
  dice_loss =  dice_coef_loss(y_true, y_pred)
  return 0.7*mse_loss+0.3*dice_loss 