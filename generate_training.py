import os
import sys
import numpy as np
import utility

train_count = 100
validation_count = 30

def print_help():
    print("Usage: -F [folder] -t [train_count] -v [validation_count]")
    print("Defaults: -t {} -v {}".format(train_count, validation_count))

if utility.isArgvKeyPresented('-h') or utility.isArgvKeyPresented('--help'):
    print_help()
    exit(0)
        
folder = utility.getArgvKeyValue("-F")
if (folder is None):
    print("Folder -F is not specified!")
    print_help()
    exit(1)

# check folder exists
isdir = os.path.isdir(folder)
if not isdir:
    # os.path.join('.', folder) # ?
    os.mkdir(folder)
    
train_count = int(utility.getArgvKeyValue("-t", train_count))
validation_count = int(utility.getArgvKeyValue("-v", validation_count))


# https://luna16.grand-challenge.org/Data/
benign_set, malignant_set = utility.get_cancer_list()

# print found classes + masks
print("Img + masks. beingn: {}, malignant: {}".format(
      len(benign_set), len(malignant_set)))

benign_set, malignant_set = utility.get_cancer_list("img")

print("Img. beingn: {}, malignant: {}".format(
      len(benign_set), len(malignant_set)))

train_count = int(train_count/2)
validation_count = int(validation_count/2)

np.random.shuffle(malignant_set)
np.random.shuffle(benign_set)

train_malignant = malignant_set[:train_count]
train_benign = benign_set[:train_count]

validation_malignant = malignant_set[train_count:train_count+validation_count]
validation_benign = benign_set[train_count:train_count+validation_count]

np.save(os.path.join(folder, "train_malignant.npy"), train_malignant)
np.save(os.path.join(folder, "train_benign.npy"), train_benign)
np.save(os.path.join(folder, "validation_malignant.npy"), validation_malignant)
np.save(os.path.join(folder, "validation_benign.npy"), validation_benign)