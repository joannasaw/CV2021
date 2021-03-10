import os
import importlib
from imutils import paths

import config
importlib.reload(config)

# CHANGE parameters here
INPUT_DATASET = os.path.sep.join([config.DATA_PATH, "val_set_SUBSET"])
OUTPUT_DATASET = os.path.sep.join([config.DATA_PATH, "val_out"])
fps = 10
num_padded = 3

# create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DATASET):
    os.makedirs(OUTPUT_DATASET)

vidPaths = list(paths.list_files(INPUT_DATASET))
for p in vidPaths:
    label = p.split(os.path.sep)[-1].strip(".mp4")
    #print(label)
    os.system("ffmpeg -i "+str(p)+" -vf fps="+str(fps)+" "+OUTPUT_DATASET+"/"+str(label)+"%0"+str(num_padded)+"d.png")
