# this script purpose is for hash computing

import os
import sys
import keras
import numpy as np
import keras.backend as K
from datetime import datetime

from utility import getArgvKeyValue, get_cancer_list, ct_folder, cancer_folder
from data_loader import load_train_data

def _help():
    print("Usage: -L [model] -H [hash folder] -F [train folder]")
    print("-L -- file from where to load model weights")
    print("-H -- file where to save hashes")
    print("-F -- folder of training folder (training set is not included in hashes list)")

def dummy_metric(y_true, y_pred):
    return K.mean(y_pred)

model_weights_load_file = None
training_folder = None

def check():
    global model_weights_load_file
    global hashes_save_file
    global training_folder
    
    model_weights_load_file = getArgvKeyValue("-L")
    if model_weights_load_file is None: #default value popup
        print("No weights file found specified at '-L' key!", file=os.sys.stderr)
        _help()
        sys.exit(1)

    # check file exist
    isfile = os.path.isfile(model_weights_load_file)
    if not isfile:        
        print("File '{}' specified at '-L' key is not exist".format(model_weights_load_file)
                    , file=os.sys.stderr)
        _help()
        sys.exit(2)

    hashes_save_file = getArgvKeyValue("-H")
    if hashes_save_file is None: #default value popup
        print("No hashes file specified at '-H' key!", file=os.sys.stderr)
        _help()
        sys.exit(3)

    training_folder = getArgvKeyValue("-F")
    
    # check folder exists
    if training_folder is not None:
        isdir = os.path.isdir(training_folder)
        if not isdir:        
            print("File '{}' specified at '-F' key is not a directory".format(training_folder)
                        , file=os.sys.stderr)
            _help()
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
# filter hashes to calculate 
if training_folder is not None:
    train_benign = np.load(os.path.join(training_folder, "train_benign.npy"))
    train_malignant = np.load(os.path.join(training_folder, "train_malignant.npy"))

    benign = np.array(list(set(benign) - set(train_benign)))
    malignant = np.array(list(set(malignant) - set(train_malignant)))

# get submodel
inner_model = model.layers[4]
# conversion window
cwnd = 64
M = len(benign)
if M < len(malignant):
    M = len(malignant)

hashes = None

def append_to_hashes(hashes, extra_hashes, hash_type = -1):
    #print("append hash ndim={}".format(extra_hashes.ndim))
    if hashes is None:
        hashes = np.insert(extra_hashes, 0, hash_type, axis=1)
    else:
        hashes = np.append(hashes, np.insert(extra_hashes, 0, hash_type, axis=1), axis=0)
    return hashes
    

last_time = None
seconds_avg = 0.0
def print_progressbar(curr, M):
    global last_time
    global seconds_avg
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
        seconds_avg = (seconds_avg*(curr-1) + delta.seconds)/curr
        est_seconds = seconds_avg * (M - curr)
        diff_str = "{} s".format(int(est_seconds))
    last_time = time

    print("Step #{}: [{}], ETA: {}".format(curr, line, diff_str), flush=True)

for k in range(0, M//cwnd + 1, 1):
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
    if len(benign_cwnd) > 0:
        hashes = append_to_hashes(hashes, benign_hashes, 0) # type=0: train benign
    if len(malignant_cwnd) > 0:
        hashes = append_to_hashes(hashes, malignant_hashes, 1) # type=1: train malignant

    #print progressbar:
    print_progressbar(k, M//cwnd+1)


# save hashes to folder
print("Saving hashes to {}...".format(hashes_save_file))
np.save(hashes_save_file, hashes)
print("Saved!")

        