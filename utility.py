import os
import sys
import numpy as np

ct_folder = 'nocancer' # folder with non-cancerous computer tomography images
cancer_folder = 'cancer' # folder with cancerous tomography images

def getArgvKeyValue(key, default = None):
    try:
        k = sys.argv.index(key)
        return sys.argv[k+1]
    except ValueError:
        return default

def isArgvKeyPresented(key):
    try:
        sys.argv.index(key)
        return True
    except ValueError:
        return False


# filtering -- leave only images
def filter_data(dataset_list, prefix = "img"):
      ret = np.array([])
      for nodule in dataset_list: #go through benign examples
            valarr = nodule.split('_')
            if (valarr[1] == prefix):
                  ret = np.append(ret, [nodule])
      return ret

def get_cancer_list(prefix=None):
    ct_dataset = os.listdir(ct_folder)
    cancer_dataset = os.listdir(cancer_folder)

    ct_set = np.array(ct_dataset) # get set of all ct images and their masks
    malignant_set = np.array(cancer_dataset) # get ct images containing cancer (call it malignant)
    benign_set = ct_set #np.setxor1d(ct_set, malignant_set) # make list of benign nodules
    if prefix is None:
        return benign_set, malignant_set
    else:
        return filter_data(benign_set, prefix), filter_data(malignant_set, prefix)