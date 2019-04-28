import os
import sys
import numpy as np
import utility

ct_folder = utility.ct_folder
cancer_folder = utility.cancer_folder

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
ct_dataset = os.listdir(ct_folder)
cancer_dataset = os.listdir(cancer_folder)

ct_set = np.array(ct_dataset) # get set of all ct images and their masks
malignant_set = np.array(cancer_dataset) # get ct images containing cancer (call it malignant)
benign_set = np.setxor1d(ct_set, malignant_set) # make list of benign nodules

# filtering -- leave only images
def filter_data(dataset_list, prefix = "img"):
      ret = np.array([])
      for nodule in dataset_list: #go through benign examples
            valarr = nodule.split('_')
            if (valarr[1] == prefix):
                  ret = np.append(ret, [nodule])
      return ret

# print found classes + masks
print("Img + masks. beingn: {}, malignant: {}".format(
      len(benign_set), len(malignant_set)))

benign_set = filter_data(benign_set)
malignant_set = filter_data(malignant_set)

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