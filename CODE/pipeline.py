import os
import cv2
import glob
import time
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

# import utility files
from models import *
import config

# ############################################################################
# #                          Data Preparation                                #
# ############################################################################

def parse_image(filepath):
    # retrieve label and image data
    label = int(filepath.split(os.path.sep)[-1].split('_')[-2])
    image = cv2.imread(filepath)
    return image, label


def preprocess(img, height, width, seed=(1,2), augment=False):
    # apply resizing & rescaling
    resize_and_rescale = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(height, width),
      tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    img = resize_and_rescale(img)
    # apply data augmentation only to training set
    if augment:
        img = tf.image.adjust_brightness(img, 0.4)
        img = tf.image.adjust_contrast(img, 0.2)
        img = tf.image.adjust_hue(img, 0.2)
        img = tf.image.adjust_saturation(img, 2)
    return img


def image_gen(subdir_ls, batch_size=1, augment=False):
    while True:

        # select a no. of subfolders (no. of video samples) for the batch
        batch_subdir_ls = np.random.choice(a=subdir_ls, size=batch_size)

        batch_x, batch_y = [], []
        # loop thru each subfolder (video sample)
        for _, subdir in enumerate(batch_subdir_ls):
            # retrieve list of image filepaths for a video sample 
            imgPaths = [f for f in glob.glob(subdir + "**/*")]

            x_vid, y_vid = [], []
            # loop thru the filepaths to obtain the img data and label
            for path in imgPaths:
                img, label = parse_image(path)
                x_vid.append(img)
                y_vid.append(label)    

            ###################################
            #     Data Augmentation START     #
            ###################################
            x_vid_arr = np.array(x_vid, dtype='float32')
            x_vid_arr = preprocess(x_vid_arr,  
                                height=config.HEIGHT, 
                                width=config.WIDTH,
                                seed=(1,2),
                                augment=augment)
            #################################
            #     Data Augmentation END     #
            #################################


        #################################
        #          WITH Padding         #
        #################################
            # pad the array to follow a predefined no. of frames per video
            # image tensors of shape (1, 30, 512, 512, 3) means
            # batch size/no. of videos: 1
            # no. of frames per video AFTER padding: 30 
            # frame height: 512
            # frame width: 512
            # no. of channels: 3

            # res = np.zeros((config.FRAMES_PADDED, 
            #                 x_vid_arr.shape[1], 
            #                 x_vid_arr.shape[2], 
            #                 x_vid_arr.shape[3]))

            # res[:x_vid_arr.shape[0], 
            #     :x_vid_arr.shape[1], 
            #     :x_vid_arr.shape[2], 
            #     :x_vid_arr.shape[3]] = x_vid_arr

            # batch_x.append(res) 
            # batch_y.append([y_vid[0]])      #for shape: (1, 226)
            # OR
            # batch_y.append(y_vid)           #for shape: (N, 226) where N is no. of frames

        # # create the batch img array and batch labels array 
        # batch_x = np.array(batch_x, dtype='float32')
        # batch_y = np.array(batch_y)
        #################################
        #           WITH Padding        #
        #################################


        #################################
        #            NO Padding         #
        #################################
        batch_x = x_vid_arr
        batch_y = np.array(y_vid)
        # OR
        #batch_y = np.array([y_vid[0]])
        #################################
        #            NO Padding         #
        #################################

        # convert class vectors (integers from 0 to num_classes) into one-hot encoded class matrix 
        batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=config.NUM_CLASSES)
            
        #return (batch_x, batch_y)
        yield(batch_x, batch_y)


# get list of subdirectories for training and validation datasets ie. [signer1_sample1, signer1_sample2, ...]
print("[INFO] Retrieving lists of subdirectory names for training & validation...")
val_subdir_ls = [x[0] for x in os.walk(config.VAL_IMGS_PATH) if x[0] != config.VAL_IMGS_PATH]
train_subdir_ls = [x[0] for x in os.walk(config.TRAIN_IMGS_PATH) if x[0] != config.TRAIN_IMGS_PATH]
print("[INFO] No. of train samples: ", len(train_subdir_ls), 
      "No. of val samples: ", len(val_subdir_ls))


print("[INFO] Preparing training & validation generators...")
val_dataset = image_gen(val_subdir_ls, config.BS)
train_dataset = image_gen(train_subdir_ls, config.BS, augment=True)
         

# check shapes
for batch in train_dataset:
    print("x_batch_train shape: ", batch[0].shape, "y_batch_train shape: ", batch[1].shape)
    break
for batch in val_dataset:
    print("x_batch_val shape: ", batch[0].shape, "y_batch_val shape: ", batch[1].shape)
    break


############################################################################
#                          Training Loop                                   #
############################################################################

# init model object
model = CustomCnnModule(num_classes=config.NUM_CLASSES)
model.build_graph(raw_shape=(config.HEIGHT,config.WIDTH,config.DEPTH)).summary()

# vgg_module = VGGModule(input_shape=(config.HEIGHT,config.WIDTH,config.DEPTH))
# vgg_module.build_graph(raw_shape=(config.HEIGHT,config.WIDTH,config.DEPTH)).summary()

# init optimizer
optimizer = tf.keras.optimizers.Adam()

# init loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# prepare metrics
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

# init tensorboard writers
writer_ls = [config.TENSORBOARD_TRAIN_WRITER, config.TENSORBOARD_VAL_WRITER]
for writer in writer_ls:
    if not os.path.exists(writer):
        os.makedirs(writer)
train_writer = tf.summary.create_file_writer(config.TENSORBOARD_TRAIN_WRITER)
test_writer  = tf.summary.create_file_writer(config.TENSORBOARD_VAL_WRITER)


# # '@tf.function' decorator compiles a function into a callable tensorflow graph
# @tf.function
def train_step(step, x, y):
    '''
    input: x, y <- batches
    input: step <- batch step 
    return: loss value
    '''
    # start the scope of gradient 
    with tf.GradientTape() as tape:
       logits = model(x, training=True) # forward pass
       train_loss_value = loss_fn(y, logits) # compute loss 
       
    # compute gradient 
    grads = tape.gradient(train_loss_value, model.trainable_weights)

    # update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # update metrics
    train_acc_metric.update_state(y, logits)

    # write training loss and accuracy to tensorboard
    with train_writer.as_default():
        tf.summary.scalar('train loss', train_loss_value, step=step)
        tf.summary.scalar('train accuracy', train_acc_metric.result(), step=step)
    
    return train_loss_value


# @tf.function
def test_step(step, x, y):
    '''
    input: x, y <- batches 
    input: step <- batch step
    return: loss value
    '''
    # forward pass, no backprop, inference mode 
    val_logits = model(x, training=False) 

    # compute the loss value 
    val_loss_value = loss_fn(y, val_logits)

    # update val metric
    val_acc_metric.update_state(y, val_logits)

    # write test loss and accuracy to tensorboard
    with test_writer.as_default():
        tf.summary.scalar('val loss', val_loss_value, step=step)
        tf.summary.scalar('val accuracy', val_acc_metric.result(), step=step) 

    return val_loss_value


# 1. Iterate over the number of epochs
# 2. For each epoch, iterate over the datasets, in batches (x, y)
# 3. For each batch, open GradientTape() scope
# 4. Inside this scope, call the model, the forward pass, compute the loss
# 5. Outside this scope, retrieve the gradients of the weights w.r.t loss
# 6. Next, use the optimizer to update the weights based on the gradients

print("[INFO] Training model...")
for epoch in range(config.EPOCHS):
    
    t = time.time()

    # iterate over the batches of the train dataset
    for train_batch_step, (batch_x, batch_y) in enumerate(train_dataset):

        train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
        train_loss_value = train_step(train_batch_step, batch_x, batch_y)

    # evaluation on validation set 
    # run a validation loop at the end of each epoch
    for test_batch_step, (batch_x, batch_y) in enumerate(val_dataset):

        test_batch_step = tf.convert_to_tensor(test_batch_step, dtype=tf.int64)
        val_loss_value = test_step(test_batch_step, batch_x, batch_y)

    template = 'ETA: {} - epoch: {} loss: {}  acc: {} val loss: {} val acc: {}\n'

    print(template.format(
        round((time.time() - t)/60, 2), epoch + 1,
        train_loss_value, float(train_acc_metric.result()),
        val_loss_value, float(val_acc_metric.result())
    ))
        
    # reset metrics at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()


# ############################################################################
# #                          Model Evaluation                                #
# ############################################################################

# # Multiclass ROC AUC score #
# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#     lb = sklearn.preprocessing.LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)
#     return roc_auc_score(y_test, y_pred, average=average)

# # ROC AUC score #
# print("ROC AUC score: ", multiclass_roc_auc_score(y_test, preds.argmax(axis=1)))

# # Classification report #
# print("evaluating network...")
# preds = model.predict(x_test, verbose=1)
# print(sklearn.metrics.classification_report(y_test.argmax(axis=1), 
#                                             preds.argmax(axis=1), 
#                                             target_names=config.CLASSES))

# # Confusion matrix #
# preds = model.predict(x_test, verbose=2)
# preds = np.argmax(preds, axis=1)
# cm = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1), preds)
# cm = pd.DataFrame(cm, range(10),range(10))
# plt.figure(figsize = (10,10))
# sns.heatmap(cm, annot=True, annot_kws={"size": 12})
# plt.savefig(config.CONFUSION_MATRIX)


# ############################################################################
# #                          Saving                                          #
# ############################################################################

# # The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture, 
# # while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and 
# # custom layers without requiring the orginal code
# model.save('net', save_format='tf')
# # A new folder 'net' will be created in the working directory: contains 'assets', 'saved_model.pb', 'variables'
# # The model architecture and training configuration, including the optimizer, losses, and metrics are stored in saved_model.pb
# # The weights are saved in the variables directory

# # OR

# # save only the trained weights
# model.save_weights('net.h5')


# ############################################################################
# #                          Loading                                         #
# ############################################################################

# # When saving the model and its layers, the SavedModel format stores the class name, call function, losses, 
# # and weights (and the config, if implemented). The call function defines the computation graph of the model/layer. 
# # In the absence of the model/layer config, the call function is used to create a model that exists like the original model 
# # which can be trained, evaluated, and used for inference.
# new_model = tf.keras.models.load_model("net", compile=False)

# # OR

# # call the build method
# new_model = CustomConvNet() 
# new_model.build((x_train.shape))
# # reload the weights 
# new_model.load_weights('net.h5')


# ############################################################################
# #                          Fine-tuning                                     #
# ############################################################################

# # init base_model
# base_model = VGGCnnModule(input_shape=(256,256,3))

# # compile for the changes to the model to take affect
# print("compiling model...")
# loss = tf.keras.losses.CategoricalCrossentropy()

# opt = tf.keras.optimizers.Adam(lr=config.INIT_LR, 
# 		                       decay=config.INIT_LR / config.FINETUNE_EPOCHS)

# metric = tf.keras.metrics.CategoricalAccuracy()

# base_model.compile(loss=loss, optimizer=opt, metrics=metric)

# # # train the model again, fine-tuning the final CONV layers
# # H = base_model.fit(,
# # 	steps_per_epoch=,
# # 	validation_data=,
# # 	validation_steps=,
# # 	epochs=)

# # # serialize the model to disk using hdf5
# # print("serializing network...")
# # model.save(config.MODEL_PATH, save_format="h5")
