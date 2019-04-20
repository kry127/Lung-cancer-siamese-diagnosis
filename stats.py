import os
import sys

def getArgvKeyValye(key, default = None):
    try:
        k = sys.argv.index(key)
        return sys.argv[k+1]
    except ValueError:
        return default

def isKeyPresented(key):
    try:
        sys.argv.index(key)
        return True
    except ValueError:
        return False

if (isKeyPresented("-h") or isKeyPresented("--help")):
    print("Usage: {} -b [path] -m [path] --img [suffix] --mask [suffix]".format(sys.argv[0]))
    print("Key description:")
    print(" -b -- specify path to benign CT images (or even all, including malignant)")
    print(" -m -- specify path to malignant CT images (and ONLY malignant)")
    print(" --img -- identifies substring in the name of file that represents CT image")
    print(" --mask -- identifies substring in the name of file that represents mask of the corresponding CT image")
    print(" -h, --help -- do nothing, print help and quit (with any combinations of keys)")
    sys.exit(0)


ct_folder = getArgvKeyValye('-b', 'all2') # folder with all computer tomography images
cancer_folder = getArgvKeyValye('-m', 'cancer') # folder with cancerous tomography images

img_suffix = getArgvKeyValye('--img', 'img')
mask_suffix = getArgvKeyValye('--mask', 'mask')

# https://luna16.grand-challenge.org/Data/
ct_dataset = os.listdir(ct_folder)
cancer_dataset = os.listdir(cancer_folder)

ct_set = set(ct_dataset) # get set of all ct images and their masks
malignant_set = set(cancer_dataset) # get ct images containing cancer (call it malignant)
benign_set = ct_set - malignant_set # make list of benign nodules

# check all correct
print("All scans+masks count: {}, malignant: {}, union: {}, intersection: {}".format(
      len(ct_set), len(malignant_set), len(ct_set | malignant_set), len(ct_set & malignant_set)))

# load data 
Nb = 0
Nb_mask = 0
for benign in benign_set: #go through benign examples
    valarr = benign.split('_')
    if (valarr[1] == img_suffix):
        Nb += 1
    if (valarr[1] == mask_suffix):
        Nb_mask += 1

        
Nm = 0 
Nm_mask = 0
for malignant in malignant_set: #go through malignant examples
    valarr = malignant.split('_')
    if (valarr[1] == img_suffix):
        Nm += 1
    if (valarr[1] == mask_suffix):
        Nm_mask += 1


print("{} benign images, {} malignant images, {} overall".format(Nb, Nm, Nb + Nm))
print("{} benign masks, {} malignant masks, {} overall".format(Nb_mask, Nm_mask, Nb_mask + Nm_mask))
