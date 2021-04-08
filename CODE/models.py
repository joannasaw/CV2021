import os
import PIL
import cv2
import PIL.Image
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dropout, LSTM, Flatten, Dense, LSTM, Bidirectional, Input, GlobalAveragePooling2D, Activation, TimeDistributed
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
    #print(data.shape)
    finalModel.add(TimeDistributed(model, input_shape=(None,256,256,3)))
    
   # finalModel.add(TimeDistributed(FPMLayer()))
    #print("Entire cnn done")
    print(finalModel.summary())    

    finalModel.add(
        TimeDistributed(
            GlobalAveragePooling2D() # Or Flatten()
        )) 
    
    finalModel.add(LSTM(512, activation='relu', return_sequences=False))
    # finalModel.add(Bidirectional(LSTM(512, activation='relu', return_sequences=True), input_shape=(1, None, 16, 16, 512)))

    # finalModel.add(Attention(512))
    finalModel.add(Dropout(0.25))
    finalModel.add(Dense(226, activation="softmax"))

    print(finalModel.summary())
    
    # timeDistributed_layer = TimeDistributed(model)(input_tensor)
    # my_time_model = Model( inputs = input_tensor, outputs = timeDistributed_layer )
    # my_time_model.summary()
    
    return finalModel
