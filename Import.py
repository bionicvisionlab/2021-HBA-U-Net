import numpy as np
import random
import pandas as pd
from skimage.color import rgb2gray
from skimage import io
from skimage import img_as_float
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt
from skimage.color import rgb2gray
from skimage.feature import blob_log
from skimage.feature import peak_local_max
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_fill_holes

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
import os
import pickle
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from PIL import Image
# import ipdb
from tqdm import tqdm
import csv

import math
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
# tf.config.run_functions_eagerly
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D, AveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
from keras import layers
from keras.layers import Input, Concatenate, Dense, Multiply, Lambda, Add
from keras.layers.core import Activation, Reshape, Dropout, SpatialDropout2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate,Concatenate,UpSampling2D
from keras.models import Model
from keras.preprocessing.image import Iterator
from keras.optimizers import Adam


