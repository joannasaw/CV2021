import os
import pandas as pd
from imutils import paths

# import config file for params
import config

# CHANGE parameters here
INPUT_DATASETS = [config.VAL_VIDS_PATH]
OUTPUT_DATASETS = [config.VAL_IMGS_PATH]
LABELS_DATASETS = [config.VAL_LABELS]
FPS = 10
NUM_PADDED = 3
OUTPUT_FILE_TYPE = "png"


for vid_dir, img_dir, labels_file in zip(INPUT_DATASETS, OUTPUT_DATASETS, LABELS_DATASETS):

    # create the output dataset directory if it doesn't exist
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # read labels file to extract labels
    df = pd.read_csv(labels_file, usecols=[0,1], names=['filename','label'])

    # list all video paths in videos dataset directory
    vidPaths = list(paths.list_files(vid_dir))

    for p in vidPaths:

        file_name = p.split(os.path.sep)[-1].split("_color")[-2]
        #print(file_name)
        label = df.loc[df['filename'] == file_name, 'label'].iloc[0]
        #print(label)
        new_filename = str(file_name)+"_"+str(label)
        #print(new_filename)

        # perform system call to extract frames from each video
        os.system("ffmpeg -i "+str(p)+" -vf fps="+str(FPS)+" "+str(img_dir)+"/"+str(new_filename)+"_%0"+str(NUM_PADDED)+"d.png")

