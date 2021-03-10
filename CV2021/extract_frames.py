import os
import pandas as pd
import config
from imutils import paths

# CHANGE parameters here
INPUT_DATASET = "videos/val_set"
OUTPUT_DATASET = "images/val_set"
FPS = 10
NUM_PADDED = 3
OUTPUT_FILE_TYPE = "png"

# create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DATASET):
    os.makedirs(OUTPUT_DATASET)

# read validation_labels file to extract labels
val_df = pd.read_csv(config.VAL_LABELS, usecols=[0,1], names=['filename','label'])


vidPaths = list(paths.list_files(INPUT_DATASET))
for p in vidPaths:
    file_name = p.split(os.path.sep)[-1].split("_color")[-2]
    #print(file_name)
    label = val_df.loc[val_df['filename'] == file_name, 'label'].iloc[0]
    #print(label)
    filename = str(file_name)+"_"+str(label)
    #print(filename)
    os.system("ffmpeg -i "+str(p)+" -vf fps="+str(FPS)+" "+OUTPUT_DATASET+"/"+str(filename)+"_%0"+str(NUM_PADDED)+"d.png")
