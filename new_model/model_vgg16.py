from keras.applications import VGG16, ResNet50
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.regularizers import l2
import keras.backend as K

def vgg_512(image_size,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0]):
    
    conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(512, 512, 3))
    
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    
    def identity_layer(tensor):
        return tensor
    
    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)
        
    vgg_output = conv_base(x1)
    
    model = Model(inputs=x, outputs=vgg_output)
    return model

def resnet_512(image_size,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0]):
    
    conv_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(512, 512, 3))
    
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    
    def identity_layer(tensor):
        return tensor
    
    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)
        
    resnet_output = conv_base(x1)
    
    model = Model(inputs=x, outputs=resnet_output)
    return model
#%%
img_height = 512 # Height of the model input images
img_width = 512 # Width of the model input images
img_channels = 3
#%%
model = vgg_512(image_size=(img_height, img_width, img_channels))
#%%
#model.summary()
model.save_weights('D:/Data/Python/deeplearning/SSD/ssd_keras-master/path/to/vggbase.h5')
#%%
model = resnet_512(image_size=(img_height, img_width, img_channels))
#model.save_weights('D:/Data/Python/deeplearning/SSD/ssd_keras-master/path/to/resnetbase.h5')
#%%
model.summary()
#%%
from keras import Input, layers
from keras.layers import Conv2D, ZeroPadding2D
from keras.regularizers import l2
pool5 = Input(shape=(32,32,512))
#fc6 = Conv2D(1024, (3, 3), dilation_rate=(7, 7), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01), name='fc6')(pool5)
fc6 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv10_padding')(pool5)



