import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import features
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import os
import pathlib

# import utility files
from models import *
import config

############################################################################
#                          Data Loading                                    #
############################################################################

# def parse_image(filename):
#     image = tf.io.read_file(filename)
#     image = tf.io.decode_image(image)
#     image = tf.image.convert_image_dtype(image, tf.float32)

#     return image


# raw_ds = tf.data.Dataset.list_files(str(config.TRAIN_PATH+'/*/*'))
# train_ds = raw_ds.map(parse_image)
# for image in train_ds.take(2):
#     print(image.shape)

# raw_ds = tf.data.Dataset.list_files(str(config.VAL_PATH+'/*/*'))
# val_ds = raw_ds.map(parse_image)
# for image in val_ds.take(2):
#     print(image.shape)

# raw_ds = tf.data.Dataset.list_files(str(config.TEST_PATH+'/*/*'))
# test_ds = raw_ds.map(parse_image)
# for image in test_ds.take(2):
#     print(image.shape)


# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# # print(x_train.shape, y_train.shape)
# # print(x_test.shape, y_test.shape)

# i = int(config.VAL_SPLIT * x_train.shape[0])
# x_train = x_train[i:,:,:,:]
# y_train = y_train[i:,:]
# x_val = x_train[:i,:,:,:]
# y_val = y_train[:i,:]
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape) 
# print(x_test.shape, y_test.shape)  

# convert the class vectors (integers from 0 to num_classes)- both train & val - to a binary class matrix 
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Prepare the training dataset (separate elements of the input tensor for efficient input pipelines)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(config.BS)

# Prepare the validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(config.BS)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(config.BS)

for batch in train_dataset.take(1):
    print("x_batch_train shape: ", batch[0].shape, "y_batch_train shape: ", batch[1].shape)


############################################################################
#                          Data Augmentation                               #
############################################################################

IMG_SIZE = config.WIDTH
# construct resizing, rescaling layers
resize_and_rescale = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  # rescale tensors to [0,1]
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

# construct data augmentation layers
aug = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE

# apply preprocessing layer to datasets
def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024)

    # # batch all datasets
    # ds = ds.batch(config.BS)

    # use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (aug(x, training=True), y), 
                    num_parallel_calls=AUTOTUNE)

    # use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


train_dataset = prepare(train_dataset, shuffle=True, augment=True)
val_dataset = prepare(val_dataset)
test_dataset = prepare(test_dataset)

# check shapes
for batch in train_dataset.take(1):
    print("x_batch_train shape: ", batch[0].shape, "y_batch_train shape: ", batch[1].shape)


############################################################################
#                          Training Loop                                   #
############################################################################

# init model object
model = CustomConvNet(num_classes=len(config.CLASSES))
model.build_graph(raw_input).summary()

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


# '@tf.function' decorator compiles a function into a callable tensorflow graph
@tf.function
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


@tf.function
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

for epoch in range(epochs):
    
    t = time.time()

    # iterate over the batches of the train dataset
    for train_batch_step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        train_batch_step = tf.convert_to_tensor(train_batch_step, dtype=tf.int64)
        train_loss_value = train_step(train_batch_step, x_batch_train, y_batch_train)

    # evaluation on validation set 
    # run a validation loop at the end of each epoch
    for test_batch_step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        test_batch_step = tf.convert_to_tensor(test_batch_step, dtype=tf.int64)
        val_loss_value = test_step(test_batch_step, x_batch_val, y_batch_val)

    template = 'ETA: {} - epoch: {} loss: {}  acc: {} val loss: {} val acc: {}\n'
    print(template.format(
        round((time.time() - t)/60, 2), epoch + 1,
        train_loss_value, float(train_acc_metric.result()),
        val_loss_value, float(val_acc_metric.result())
    ))
        
    # reset metrics at the end of each epoch
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()


############################################################################
#                          Model Evaluation                                #
############################################################################

# Multiclass ROC AUC score #
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

# ROC AUC score #
print("ROC AUC score: ", multiclass_roc_auc_score(y_test, preds.argmax(axis=1)))

# Classification report #
print("evaluating network...")
preds = model.predict(x_test, verbose=1)
print(sklearn.metrics.classification_report(y_test.argmax(axis=1), 
                                            preds.argmax(axis=1), 
                                            target_names=config.CLASSES))

# Confusion matrix #
preds = model.predict(x_test, verbose=2)
preds = np.argmax(preds, axis=1)
cm = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1), preds)
cm = pd.DataFrame(cm, range(10),range(10))
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True, annot_kws={"size": 12})
plt.savefig(config.CONFUSION_MATRIX)


############################################################################
#                          Saving                                          #
############################################################################

# The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture, 
# while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and 
# custom layers without requiring the orginal code
model.save('net', save_format='tf')
# A new folder 'net' will be created in the working directory: contains 'assets', 'saved_model.pb', 'variables'
# The model architecture and training configuration, including the optimizer, losses, and metrics are stored in saved_model.pb
# The weights are saved in the variables directory

# OR

# save only the trained weights
model.save_weights('net.h5')


############################################################################
#                          Loading                                         #
############################################################################

# When saving the model and its layers, the SavedModel format stores the class name, call function, losses, 
# and weights (and the config, if implemented). The call function defines the computation graph of the model/layer. 
# In the absence of the model/layer config, the call function is used to create a model that exists like the original model 
# which can be trained, evaluated, and used for inference.
new_model = tf.keras.models.load_model("net", compile=False)

# OR

# call the build method
new_model = CustomConvNet() 
new_model.build((x_train.shape))
# reload the weights 
new_model.load_weights('net.h5')


############################################################################
#                          Fine-tuning                                     #
############################################################################

# init base_model
base_model = VGGCnnModule(input_shape=(256,256,3))

# compile for the changes to the model to take affect
print("compiling model...")
loss = tf.keras.losses.CategoricalCrossentropy()

opt = tf.keras.optimizers.Adam(lr=config.INIT_LR, 
		                       decay=config.INIT_LR / config.FINETUNE_EPOCHS)

metric = tf.keras.metrics.CategoricalAccuracy()

base_model.compile(loss=loss, optimizer=opt, metrics=metric)

# # train the model again, fine-tuning the final CONV layers
# H = base_model.fit(,
# 	steps_per_epoch=,
# 	validation_data=,
# 	validation_steps=,
# 	epochs=)

# # serialize the model to disk using hdf5
# print("serializing network...")
# model.save(config.MODEL_PATH, save_format="h5")
