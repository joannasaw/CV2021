import os
from pathlib import Path
# initilaize the path to the original input dataset
ORIG_INPUT_DATASET = ""

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = ""
dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = str(Path(dir_path).parents[0])

# initialize DATA directory
DATA_PATH = os.path.sep.join([BASE_PATH, "DATA"])

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used training
TRAIN_SPLIT = 0.9
# the amount of validation data will be a percentage of training data
VAL_SPLIT = 0.05
# define the names of the classes
CLASSES = [""]

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
