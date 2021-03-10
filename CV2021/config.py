import os

# init the path to the original input dataset
ORIG_INPUT_DATASET = "videos" 

# init the base path to the *new* directory containing images
BASE_IMAGES_PATH = "images"
BASE_LABELS_PATH = "labels"

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_IMAGES_PATH, ""])
VAL_PATH = os.path.sep.join([BASE_IMAGES_PATH, "val_set"])
TEST_PATH = os.path.sep.join([BASE_IMAGES_PATH, ""])

TRAIN_LABELS = os.path.sep.join([BASE_LABELS_PATH, ""])
VAL_LABELS = os.path.sep.join([BASE_LABELS_PATH, "val_labels.csv"])
TEST_LABELS = os.path.sep.join([BASE_LABELS_PATH, ""])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.9
# the amount of validation data will be a percentage of training data
VAL_SPLIT = 0.05
# define the names of the classes
CLASSES = [""]
NUM_CLASSES = 226

# initialize the with, height and no. of channels
WIDTH = 16
HEIGHT = 16
DEPTH = 3

# initialize the number of epochs to train for
# initial learning rate and batch size
WARMUP_EPOCHS = 20
FINETUNE_EPOCHS = 10
INIT_LR = 1e-3
BS = 32

# path to output trained autoencoder
MODEL_PATH = "outputs/model.h5"
# path to output plot file
WARMUP_PLOT_PATH = "outputs/head_training.png"
UNFROZEN_PLOT_PATH = "outputs/fine_tuned.png"

# path to tensorboard logs
TENSORBOARD_TRAIN_WRITER = 'output/logs/train/'
TENSORBOARD_VAL_WRITER = 'output/logs/val/'

# path to confusion matrix fig
CONFUSION_MATRIX = 'cm.png'