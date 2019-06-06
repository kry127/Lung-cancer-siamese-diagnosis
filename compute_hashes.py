# this script purpose is for hash computing

import os
import sys
import keras
import numpy as np
import keras.backend as K
from datetime import datetime

from utility import getArgvKeyValue, isArgvKeyPresented, get_cancer_list, ct_folder, cancer_folder
from data_loader import load_train_data

def help():
    print("Usage: -L [model] -F [folder]")
    print("-L -- from where to load model")
    print("-F -- folder to save hashes")

def dummy_metric(y_true, y_pred):
    return K.constant(0)

model_weights_load_file = None

def check():
    global model_weights_load_file
    global hashes_save_folder
    
    model_weights_load_file = getArgvKeyValue("-L")
    if model_weights_load_file is None: #default value popup
        print("No weights file found specified at '-L' key!", file=os.sys.stderr)
        help()
        sys.exit(1)

    # check file exist
    isdir = os.path.isfile(model_weights_load_file)
    if not isdir:        
        print("File '{}' specified at '-L' key is not exist".format(model_weights_load_file)
                    , file=os.sys.stderr)
        help()
        sys.exit(2)

    hashes_save_folder = getArgvKeyValue("-F")
    if hashes_save_folder is None: #default value popup
        print("No hashes folder specified at '-F' key!", file=os.sys.stderr)
        help()
        sys.exit(3)

    
    exists = os.path.exists(hashes_save_folder)
    if not exists:
        try:
            os.makedirs(hashes_save_folder)
        except OSError:
            print("Cannot create folder '{}' for hashes",format(hashes_save_folder), file=os.sys.stderr)
            help()
            sys.exit(4)

check()


# check if user wants to preload existing weights
exists = os.path.isfile(model_weights_load_file)
if exists:
    model = keras.models.load_model(model_weights_load_file
            , custom_objects={'mean_distance': dummy_metric,
                            'mean_contradistance': dummy_metric,
                            'contrastive_loss': dummy_metric})

# get hashes to calculate
benign, malignant = get_cancer_list("img")
# get submodel
inner_model = model.layers[4]
# conversion window
cwnd = 64
M = len(benign)
if M < len(malignant):
    M = len(malignant)

hashes = None

def append_to_hashes(hashes, extra_hashes, hash_type = -1):
    if hashes is None:
        hashes = np.insert(extra_hashes, 0, hash_type, axis=1)
    else:
        hashes = np.append(hashes, np.insert(extra_hashes, 0, hash_type, axis=1), axis=0)
    return hashes
    

last_time = None
def print_progressbar(curr, M):
    global last_time
    chars = 30
    eq_count = curr*chars//M
    if eq_count > chars:
        eq_count = chars
    line = "="*eq_count
    if (eq_count < chars):
        line += ">"
    line += "."*(chars - eq_count - 1)

    time = datetime.now()
    diff_str = "unknown"
    if last_time is not None:
        delta = time - last_time
        diff_str = str(delta.seconds)
        last_time = time

    print("[{}], ETA: {}s".format(line, diff_str))

for k in range(0, M, cwnd):
    # get subindexes
    benign_cwnd = benign[k*cwnd:(k+1)*cwnd]
    malignant_cwnd = malignant[k*cwnd:(k+1)*cwnd]
    # load data
    data_benign_cwnd = load_train_data(ct_folder, benign_cwnd)
    data_malignant_cwnd = load_train_data(cancer_folder, malignant_cwnd)
    # compute hashes
    benign_hashes = inner_model.predict([np.expand_dims(data_benign_cwnd, axis=-1)])
    malignant_hashes = inner_model.predict([np.expand_dims(data_malignant_cwnd, axis=-1)])

    # stack hashes
    hashes = append_to_hashes(hashes, benign_hashes, 0) # type=0: train benign
    hashes = append_to_hashes(hashes, malignant_hashes, 1) # type=1: train malignant

    #print progressbar:
    print_progressbar(k, M//cwnd+1)


# save hashes to folder
print("Saving hashes to {}...".format(hashes_save_folder))
np.save(hashes_save_folder, hashes)
print("Saved!")

        