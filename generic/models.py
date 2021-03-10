# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
import os
import numpy as np
import tensorflow as tf


############################################################################
#                          Functional API                                  #
############################################################################

def miniGoogleNet(width, height, depth, classes):
	def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
		# define a CONV => BN => RELU pattern
		x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation("relu")(x)
		return x
		
	def inception_module(x, numK1x1, numK3x3, chanDim):
		# define two CONV modules, then concatenate across the channel dimension
		conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
		conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
		return x
		
	def downsample_module(x, K, chanDim):
		# define the CONV module and POOL, then concatenate across the channel dimensions
		conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
		pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		x = concatenate([conv_3x3, pool], axis=chanDim)
		return x

    # initialize the input shape to be "channels last" and the channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input and first CONV module
	inputs = Input(shape=inputShape)
	x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
	
    # two Inception modules followed by a downsample module
	x = inception_module(x, 32, 32, chanDim)
	x = inception_module(x, 32, 48, chanDim)
	x = downsample_module(x, 80, chanDim)
	
    # four Inception modules followed by a downsample module
	x = inception_module(x, 112, 48, chanDim)
	x = inception_module(x, 96, 64, chanDim)
	x = inception_module(x, 80, 80, chanDim)
	x = inception_module(x, 48, 96, chanDim)
	x = downsample_module(x, 96, chanDim)
	
    # two Inception modules followed by global POOL and dropout
	x = inception_module(x, 176, 160, chanDim)
	x = inception_module(x, 176, 160, chanDim)
	x = AveragePooling2D((7, 7))(x)
	x = Dropout(0.5)(x)
	
    # softmax classifier
	x = Flatten()(x)
	x = Dense(classes)(x)
	x = Activation("softmax")(x)
	
    # create the model
	model = Model(inputs, x, name="minigooglenet")
	return model


############################################################################
#                          Model Subclassing                               #
############################################################################


# use Layer class to define inner computation blocks:
# 1. Convolutional module 
# 2. Inception module
# 3. Downsample module
class ConvModule(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides, padding='same'):
        super(ConvModule, self).__init__()
        # conv layer
        self.conv = tf.keras.layers.Conv2D(kernel_num,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        # batch norm layer
        self.bn   = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

        
class InceptionModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size1x1, kernel_size3x3):
        super(InceptionModule, self).__init__()
        # two conv modules: they will take same input tensor 
        self.conv1 = ConvModule(kernel_size1x1, 
                                kernel_size=(1,1), 
                                strides=(1,1))
        self.conv2 = ConvModule(kernel_size3x3, 
                                kernel_size=(3,3), 
                                strides=(1,1))
        self.cat   = tf.keras.layers.Concatenate()

    def call(self, input_tensor, training=False):
        # forward pass & merge
        x_1x1 = self.conv1(input_tensor)
        x_3x3 = self.conv2(input_tensor)
        x = self.cat([x_1x1, x_3x3])
        return x 


class DownsampleModule(tf.keras.layers.Layer):
    def __init__(self, kernel_size):
        super(DownsampleModule, self).__init__()
        # conv layer
        self.conv3 = ConvModule(kernel_size,
                                kernel_size=(3,3),
                                strides=(2,2),
                                padding="valid") 
        # pooling layer
        self.pool  = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))
        self.cat   = tf.keras.layers.Concatenate()
        
    def call(self, input_tensor, training=False):
        # forward pass & merge
        conv_x = self.conv3(input_tensor, training=training)
        pool_x = self.pool(input_tensor)
        x = self.cat([conv_x, pool_x])
        return x

        
class CustomConvNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(CustomConvNet, self).__init__()

        # the first conv module
        self.conv_block = ConvModule(96, (3,3), (1,1))

        # 2 inception module & 1 downsample module
        self.inception_block1  = InceptionModule(32, 32)
        self.inception_block2  = InceptionModule(32, 48)
        self.downsample_block1 = DownsampleModule(80)
        
        # 4 inception module & 1 downsample module
        self.inception_block3  = InceptionModule(112, 48)
        self.inception_block4  = InceptionModule(96, 64)
        self.inception_block5  = InceptionModule(80, 80)
        self.inception_block6  = InceptionModule(48, 96)
        self.downsample_block2 = DownsampleModule(96)

        # 2 inception module 
        self.inception_block7 = InceptionModule(176, 160)
        self.inception_block8 = InceptionModule(176, 160)

        # average pooling
        self.avg_pool = tf.keras.layers.AveragePooling2D((7,7))
        
        # dropout
        self.dropout = tf.keras.layers.Dropout(0.5)

        # softmax classifier
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes)
        self.softmax = tf.keras.layers.Activation("softmax")
    
    def call(self, input_tensor, training=False, **kwargs):
        
        x = self.conv_block(input_tensor)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.downsample_block1(x)

        x = self.inception_block3(x)
        x = self.inception_block4(x)
        x = self.inception_block5(x)
        x = self.inception_block6(x)
        x = self.downsample_block2(x)

        x = self.inception_block7(x)
        x = self.inception_block8(x)
        x = self.avg_pool(x)
        
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)

        return x

    def build_graph(self, raw_shape):
        # helper function to plot model summary information
        x = tf.keras.layers.Input(shape=raw_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

