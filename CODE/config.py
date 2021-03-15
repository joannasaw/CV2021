import os
from pathlib import Path

# init the base path to the *new* directory 
BASE_PATH = ""
dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = str(Path(dir_path).parents[0])

# init DATA directory
DATA_PATH = os.path.sep.join([BASE_PATH, "DATA"])

# derive the raw videos directory
TRAIN_VIDS_PATH = os.path.sep.join([DATA_PATH, "videos/train"])
VAL_VIDS_PATH = os.path.sep.join([DATA_PATH, "videos/val"])

# derive the images directory
TRAIN_IMGS_PATH = os.path.sep.join([DATA_PATH, "images/train"])
VAL_IMGS_PATH = os.path.sep.join([DATA_PATH, "images/val"])

# derive the labels directory
TRAIN_LABELS = os.path.sep.join([DATA_PATH, "labels/train_labels.csv"])
VAL_LABELS = os.path.sep.join([DATA_PATH, "labels/val_labels.csv"])

CLASSES = [136, 130, 139, 222, 31, 60, 100, 168, 54, 225, 142, 170, 29, \
    118, 111, 48, 134, 43, 94, 86, 22, 38, 73, 32, 137, 149, 218, 77, 17, \
    18, 221, 146, 143, 220, 50, 34, 211, 27, 51, 197, 40, 175, 164, 208, 209, \
    72, 123, 36, 200, 158, 1, 176, 194, 90, 162, 128, 104, 150, 15, 3, 179, 156, \
    64, 106, 215, 191, 12, 112, 67, 216, 116, 20, 28, 46, 55, 140, 107, 202, 113, \
    223, 181, 110, 152, 103, 4, 63, 186, 203, 212, 119, 21, 65, 30, 214, 174, 85, \
    153, 75, 177, 184, 127, 114, 62, 165, 56, 166, 8, 23, 122, 180, 147, 101, 206, \
    121, 148, 68, 219, 79, 141, 144, 198, 2, 83, 131, 161, 66, 57, 70, 124, 74, 117, \
    99, 25, 91, 193, 185, 49, 37, 61, 11, 52, 81, 192, 44, 47, 97, 78, 89, 173, 87, \
    196, 195, 14, 69, 172, 129, 187, 205, 135, 183, 163, 16, 10, 125, 84, 41, 0, 102, \
    5, 71, 199, 58, 24, 108, 217, 13, 93, 189, 167, 182, 178, 115, 159, 105, 190, 145, \
    126, 39, 210, 109, 154, 35, 204, 42, 213, 207, 9, 133, 82, 120, 45, 53, 96, 26, 19, \
    80, 224, 138, 155, 7, 98, 132, 188, 59, 201, 33, 171, 88, 151, 76, 169, 157, 92, 160, 95, 6]

# num of classes
NUM_CLASSES = len(CLASSES)

# frames per video after padding
FRAMES_PADDED = 30

FPS = 10
VID_NUM_PADDED = 3
OUTPUT_FILE_TYPE = "png"

# initialize the width, height and no. of channels
WIDTH = 256
HEIGHT = 256
DEPTH = 3

# initialize the number of epochs to train for
# initial learning rate, batch size, finetuning epochs
BS = 1
EPOCHS = 20
INIT_LR = 1e-3
FINETUNE_EPOCHS = 10

# path to output trained autoencoder
MODEL_PATH = "outputs/model.h5"

# path to tensorboard logs
TENSORBOARD_TRAIN_WRITER = 'output/logs/train/'
TENSORBOARD_VAL_WRITER = 'output/logs/val/'

# path to confusion matrix fig
CONFUSION_MATRIX = 'cm.png'
