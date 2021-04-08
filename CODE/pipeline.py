import os
import PIL
import cv2
import PIL.Image
import glob
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Flatten, Dense, LSTM, Bidirectional, Input, GlobalAveragePooling2D, Activation, TimeDistributed
from attention import Attention
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import utility files
from models import *
import config
import utils


############################################################################
#                          Training Loop                                   #
############################################################################


# 1. Iterate over the number of epochs
# 2. For each epoch, iterate over the datasets, in batches (x, y)
# 3. For each batch, open GradientTape() scope
# 4. Inside this scope, call the model, the forward pass, compute the loss
# 5. Outside this scope, retrieve the gradients of the weights w.r.t loss
# 6. Next, use the optimizer to update the weights based on the gradients

# init optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=config.INIT_LR)

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


# init model object
time_model = VGG(np.ones((5,256,256,3)))
x = time_model.output
model = Model(time_model.input, x)
for layer in model.layers:
    print(layer, layer.input_shape, layer.output_shape)


#'@tf.function' decorator compiles a function into a callable tensorflow graph
@tf.function
def train_batch(step, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True) # forward pass
        loss_value = loss_fn(y, logits) # compute loss   
    # compute gradients
    grads = tape.gradient(loss_value, model.trainable_weights)
    # update weights
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # update metrics
    train_acc_metric.update_state(y, logits)
    # write training loss and accuracy to tensorboard
    with train_writer.as_default():
        tf.summary.scalar('train loss', loss_value, step=step)
        tf.summary.scalar('train accuracy', train_acc_metric.result(), step=step)
    return loss_value


@tf.function
def test_batch(step, x, y):
    # forward pass, no backprop, inference mode
    val_logits = model(x, training=False)
    # compute loss
    loss_value = loss_fn(y, val_logits)
    # update metrics
    val_acc_metric.update_state(y, val_logits)
    # write test loss and accuracy to tensorboard
    with test_writer.as_default():
        tf.summary.scalar('val loss', loss_value, step=step)
        tf.summary.scalar('val accuracy', val_acc_metric.result(), step=step)
    return loss_value


print("[INFO] Training model...")
epochs = config.EPOCHS
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    t = time.time()
    print("[INFO] Preparing training & validation generators...")
    val_dataset = tf.data.Dataset.from_generator(generator=utils.val_image_gen, 
                                            output_types=(tf.float32, tf.int32),) 

    train_dataset = tf.data.Dataset.from_generator(generator=utils.train_image_gen, 
                                            output_types=(tf.float32, tf.int32),)
                                                  

    # Iterate over the batches of the dataset
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        train_loss_value = train_batch(step, x_batch_train, y_batch_train)

        # Log every 200 batches
        if step % 200 == 0:
            print("Training loss (1 batch) at step %d: %.4f" % (step, float(train_loss_value)))
            

    # Run a validation loop at the end of each epoch
    for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
        step = tf.convert_to_tensor(step, dtype=tf.int64)
        val_loss_value = test_batch(step, x_batch_val, y_batch_val)

    template = 'ETA: {} - epoch: {} loss: {}  acc: {} val loss: {} val acc: {}\n'

    print(template.format(
        round((time.time() - t)/60, 2), epoch + 1,
        train_loss_value, float(train_acc_metric.result()),
        val_loss_value, float(val_acc_metric.result())
    ))

    # Reset training metrics at the end of each epoch
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
