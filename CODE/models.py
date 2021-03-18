import os
import PIL
import cv2
import PIL.Image
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, LSTM, Bidirectional, Input, GlobalAveragePooling2D, Activation, TimeDistributed
from attention import Attention
from keras.utils.vis_utils import plot_model


class FPMLayer(tf.keras.Model):
    def __init__(self):
        super(FPMLayer, self).__init__(name="fpm")
        self.conv1 = tf.keras.layers.Conv2D(kernel_size=(1,1),filters=128,activation='relu',padding="same")
        self.conv2 = tf.keras.layers.Conv2D(kernel_size=(3,3),filters=128, activation='relu', padding="same")
        self.conv3 = tf.keras.layers.Conv2D(kernel_size=(3,3),filters=128, dilation_rate=2, activation='relu', padding="same")
        self.conv4 = tf.keras.layers.Conv2D(kernel_size=(3,3),filters=128, dilation_rate=4, activation='relu', padding="same")

    def call(self, inputs):
        print("called")
        a = self.conv1(inputs)
        b = self.conv2(inputs)
        c = self.conv3(inputs)
        d = self.conv4(inputs)
        return tf.keras.layers.concatenate([a, b, c, d], axis=-1)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        return tf.TensorShape([1, 16,16, 512])

    def Vis(self):
        plot_model(self, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# without last max pooling layer
def VGG(data):
    input_shape = data.shape[1:]
    seq_len = data.shape[0]

    base_model = tf.keras.applications.VGG16(
        include_top=False, # dont need to have FC layer
        weights='imagenet')
    
    model = Sequential()
    for layer in base_model.layers[:-1]: # just exclude last layer from copying
        model.add(layer)

    # print("No. of layers in the base model: ", len(model.layers))
    # model.summary()

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during fine-tuning
    model.trainable = False

    # Fine-tune on block5_conv2 , block5_conv3
    # need to train everything
    train_from_layer = 15 # index 0 to 16
    # # Freeze all the layers before 'train_from_layer'
    
    # print(len(model.layers[train_from_layer:]))
    for layer in model.layers[train_from_layer:]:
        layer.trainable = True # trains from block5_conv2 , block5_conv3 onwards
   
    finalModel = Sequential()
    #finalModel.add(Input(shape=(data.shape)))
    print(data.shape)
    finalModel.add(TimeDistributed(model, input_shape=(None, 256,256, 3)))
    
    finalModel.add(TimeDistributed(FPMLayer()))
    print("Entire cnn done")
    print(finalModel.summary())    

    finalModel.add(
        TimeDistributed(
            GlobalAveragePooling2D() # Or Flatten()
        )) 
    
    finalModel.add(LSTM(512, activation='relu', return_sequences=True))
    # finalModel.add(Bidirectional(LSTM(512, activation='relu', return_sequences=True), input_shape=(1, None, 16, 16, 512)))

    # finalModel.add(Attention(512))
    
    finalModel.add(Dense(226, activation="softmax"))

    print(finalModel.summary())
    
    # timeDistributed_layer = TimeDistributed(model)(input_tensor)
    # my_time_model = Model( inputs = input_tensor, outputs = timeDistributed_layer )
    # my_time_model.summary()
    
    return finalModel


###################################################################
#                       MODEL SUBCLASSING                         #    
###################################################################
# # use Layer class to define inner computation blocks:
# # 1. Convolutional module 
# # 2. Inception module
# # 3. Downsample module
# class ConvModule(tf.keras.layers.Layer):
#     def __init__(self, kernel_num, kernel_size, strides, padding='same'):
#         super(ConvModule, self).__init__()
#         # conv layer
#         self.conv = tf.keras.layers.Conv2D(kernel_num,
#                                            kernel_size=kernel_size,
#                                            strides=strides,
#                                            padding=padding)
#         # batch norm layer
#         self.bn   = tf.keras.layers.BatchNormalization()

#     def call(self, input_tensor, training=False):
#         x = self.conv(input_tensor)
#         x = self.bn(x, training=training)
#         x = tf.nn.relu(x)
#         return x

        
# class InceptionModule(tf.keras.layers.Layer):
#     def __init__(self, kernel_size1x1, kernel_size3x3):
#         super(InceptionModule, self).__init__()
#         # two conv modules: they will take same input tensor 
#         self.conv1 = ConvModule(kernel_size1x1, 
#                                 kernel_size=(1,1), 
#                                 strides=(1,1))
#         self.conv2 = ConvModule(kernel_size3x3, 
#                                 kernel_size=(3,3), 
#                                 strides=(1,1))
#         self.cat   = tf.keras.layers.Concatenate()

#     def call(self, input_tensor, training=False):
#         # forward pass & merge
#         x_1x1 = self.conv1(input_tensor)
#         x_3x3 = self.conv2(input_tensor)
#         x = self.cat([x_1x1, x_3x3])
#         return x 


# class DownsampleModule(tf.keras.layers.Layer):
#     def __init__(self, kernel_size):
#         super(DownsampleModule, self).__init__()
#         # conv layer
#         self.conv3 = ConvModule(kernel_size,
#                                 kernel_size=(3,3),
#                                 strides=(2,2),
#                                 padding="valid") 
#         # pooling layer
#         self.pool  = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2,2))
#         self.cat   = tf.keras.layers.Concatenate()
        
#     def call(self, input_tensor, training=False):
#         # forward pass & merge
#         conv_x = self.conv3(input_tensor, training=training)
#         pool_x = self.pool(input_tensor)
#         x = self.cat([conv_x, pool_x])
#         return x


# class VGGModule(tf.keras.layers.Layer):
#     def __init__(self, input_shape):
#         super(VGGModule, self).__init__()

#         # construct base model
#         self.base_model = tf.keras.applications.VGG16(
#             input_tensor=tf.keras.layers.Input(shape=(input_shape)),
#             include_top=False,
#             weights='imagenet')

#         # loop over all layers in the base model and freeze them so they will
#         # *not* be updated during fine-tuning
#         self.base_model.trainable = False

#         #print("No. of layers in the base model: ", len(self.base_model.layers))
#         # Fine-tune on block5_conv2 , block5_conv3
#         train_from_layer = 16
#         # Freeze all the layers before 'train_from_layer'
#         for layer in self.base_model.layers[train_from_layer:]:
#             layer.trainable = True
#         #for layer in self.base_model.layers:
#         #    print("{}: {}".format(layer, layer.trainable))
        
#         self.vgg_module = tf.keras.Model(inputs=self.base_model.input,
#                                          outputs=self.base_model.get_layer('block5_conv3').output)
    
#     def call(self, input_tensor, training=False):
#         return self.vgg_module(input_tensor, training=training) 
    
#     def build_graph(self, raw_shape):
#         # helper function to plot model summary information
#         x = tf.keras.layers.Input(shape=raw_shape)
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    
# class CustomCnnModule(tf.keras.Model):
#     def __init__(self, num_classes):
#         super(CustomCnnModule, self).__init__()

#         # the first conv module
#         self.conv_block = ConvModule(96, (3,3), (1,1))

#         # 2 inception module & 1 downsample module
#         self.inception_block1  = InceptionModule(32, 32)
#         self.inception_block2  = InceptionModule(32, 48)
#         self.downsample_block1 = DownsampleModule(80)
        
#         # 4 inception module & 1 downsample module
#         self.inception_block3  = InceptionModule(112, 48)
#         self.inception_block4  = InceptionModule(96, 64)
#         self.inception_block5  = InceptionModule(80, 80)
#         self.inception_block6  = InceptionModule(48, 96)
#         self.downsample_block2 = DownsampleModule(96)

#         # 2 inception module 
#         self.inception_block7 = InceptionModule(176, 160)
#         self.inception_block8 = InceptionModule(176, 160)

#         # average pooling
#         self.avg_pool = tf.keras.layers.AveragePooling2D((7,7))
        
#         # dropout
#         self.dropout = tf.keras.layers.Dropout(0.5)

#         # softmax classifier
#         self.flatten = tf.keras.layers.Flatten()
#         self.dense = tf.keras.layers.Dense(num_classes)
#         self.softmax = tf.keras.layers.Activation("softmax")
    
#     def call(self, input_tensor, training=False):
        
#         x = self.conv_block(input_tensor)
#         x = self.inception_block1(x)
#         x = self.inception_block2(x)
#         x = self.downsample_block1(x)

#         x = self.inception_block3(x)
#         x = self.inception_block4(x)
#         x = self.inception_block5(x)
#         x = self.inception_block6(x)
#         x = self.downsample_block2(x)

#         x = self.inception_block7(x)
#         x = self.inception_block8(x)
#         x = self.avg_pool(x)
        
#         x = self.dropout(x)

#         x = self.flatten(x)
#         x = self.dense(x)
#         x = self.softmax(x)
#         return x

#     def build_graph(self, raw_shape):
#         # helper function to plot model summary information
#         x = tf.keras.layers.Input(shape=raw_shape)
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))
