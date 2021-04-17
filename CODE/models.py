import os
import PIL
import cv2
import PIL.Image
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, LSTM, Bidirectional, Input, GlobalAveragePooling2D, Activation, TimeDistributed, Activation, Dropout
from attention import Attention
# import utility files
import config


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, activation='relu', padding='same'):
        super(ConvLayer, self).__init__()
        # conv layer
        self.conv = tf.keras.layers.Conv2D(filters,
                                           kernel_size=kernel_size,
                                           dilation_rate=dilation_rate,
                                           activation=activation,
                                           padding=padding)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        return x


class FPModule(tf.keras.Model):
    def __init__(self):
        super(FPModule, self).__init__(name="feature_pooling_module")
        self.layer1 = ConvLayer(filters=128, kernel_size=(1,1), dilation_rate=(1,1))
        self.layer2 = ConvLayer(filters=128, kernel_size=(3,3), dilation_rate=(1,1))
        self.layer3 = ConvLayer(filters=128, kernel_size=(3,3), dilation_rate=(2,2))
        self.layer4 = ConvLayer(filters=128, kernel_size=(3,3), dilation_rate=(4,4))
        self.cat    = tf.keras.layers.Concatenate()

    def call(self, input_tensor):
        x_1 = self.layer1(input_tensor)
        x_2 = self.layer2(input_tensor)
        x_3 = self.layer3(input_tensor)
        x_4 = self.layer4(input_tensor)
        x   = self.cat([x_1, x_2, x_3, x_4])
        return x 

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        return tf.TensorShape([1,16,16,512])

    def build_graph(self):
        # helper function to plot model summary information
        x = tf.keras.layers.Input(shape=(16,16,512))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


def final_model(MODEL_ARCH):

    if MODEL_ARCH.split('_')[0] == 'vgg':
        model = tf.keras.applications.VGG16(
            include_top=False, # dont need to have FC layer,
            weights='imagenet')

        base_model = tf.keras.Model(inputs=model.input,
                                    outputs=model.layers[-2].output
                                    )

        base_model.trainable = False
        train_from_layer = 15
        for layer in base_model.layers[train_from_layer:]:
            layer.trainable = True

    elif MODEL_ARCH.split('_')[0] == 'resnet':
        model = tf.keras.applications.ResNet50(
            include_top=False, # dont need to have FC layer
            weights='imagenet')

        base_model = tf.keras.Model(inputs=model.input,
                                    outputs=model.layers[-2].output
                                    )

        base_model.trainable = False
        train_from_layer = 143
        for layer in base_model.layers[train_from_layer:]:
            layer.trainable = True

    # print(base_model.summary())
    # for i,layer in enumerate(base_model.layers):
    #     print(i, layer.input_shape, layer.output_shape, layer.trainable)

    finalModel = Sequential()
    finalModel.add(TimeDistributed(base_model, input_shape=(None,256,256,3), name=MODEL_ARCH.split('_')[0]+"_backbone"))
    
    # add feature pooling module
    if MODEL_ARCH.split('_')[1] == 'fpm':
        finalModel.add(TimeDistributed(FPModule())) 

    # need to have only 1 dimension per cnn/fpm output to feed into LSTM layer - Flatten or use Pooling
    finalModel.add(TimeDistributed(GlobalAveragePooling2D()))

    # add dropout of 0.25
    finalModel.add(Dropout(0.25))
    
    # add LSTM layer with 512 hidden units
    if 'lstm' in MODEL_ARCH.split('_'):
        finalModel.add(LSTM(512, activation='relu', return_sequences=True if MODEL_ARCH.split('_')[-1] == 'attention' else False))
    # add Bidirectional extensions
    if 'blstm' in MODEL_ARCH.split('_'):
        finalModel.add(Bidirectional(LSTM(512, activation='relu', return_sequences=True if MODEL_ARCH.split('_')[-1] == 'attention' else False), input_shape=(1,None,16,16,512)))
    
    # add attention mechanism
    if MODEL_ARCH.split('_')[-1] == 'attention':
        finalModel.add(Attention(512))
    
    # add dropout of 0.25
    finalModel.add(Dropout(0.25))

    # add FC layer of num_classes=226 with softmax classifier head
    finalModel.add(Dense(config.NUM_CLASSES, activation="softmax"))

    # output the final model summary
    print(finalModel.summary())
    
    return finalModel
