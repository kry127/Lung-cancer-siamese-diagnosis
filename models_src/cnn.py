# a script for classic CNN implementation of classification task


import os
import keras
import utility
import data_loader

import numpy as np

from models_src.model_ResNet import inner_model
from utility import getArgvKeyValue, isArgvKeyPresented
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

ct_folder = utility.ct_folder # folder with all computer tomography images
cancer_folder = utility.cancer_folder # folder with cancerous tomography images

def print_help():
    pass

def check_folder(folder):
    if (folder is None):
        print("Folder -F is not specified!")
        #print_help()
        exit(1)

    # check folder exists
    isdir = os.path.isdir(folder)
    if not isdir:
        print("Folder -F is not exist!")
        #print_help()
        exit(1)

# loading training and validation set, setting parameters
training_folder = getArgvKeyValue("-F") # folder for data loading
check_folder(training_folder)
batch_size = int(getArgvKeyValue("-bs", 100)) # how many pairs form loss function in every training step
epochs_all = int(getArgvKeyValue("-e", 300)) # every epoch training pairs shuffled
steps_per_epoch = int(getArgvKeyValue("-s", 3)) # how many steps per epoch available
learning_rate = float(getArgvKeyValue("-lr", 0.06)) # the learning rate should correspond to loss magnitude
learning_rate_reduce_factor = getArgvKeyValue("-rf")
if learning_rate_reduce_factor is not None:
    learning_rate_reduce_factor = float(learning_rate_reduce_factor)
augmentation = isArgvKeyPresented("-aug")

# loss function parameters
margin = float(getArgvKeyValue("-m", 10)) # margin defines how strong dissimilar values are pushed from each other (contrastive loss)
lambda1 = float(getArgvKeyValue("-l1", 1)) # lambda1
lambda2 = float(getArgvKeyValue("-l2", 1)) # lambda2

visualisation = isArgvKeyPresented("-vis")
visualisation_folder = getArgvKeyValue("-V")
model_weights_load_file = getArgvKeyValue("-L") # can be none
model_weights_save_file = getArgvKeyValue("-S", "./lung_cancer_siamese_conv3D.model") # with default value

print("\n")
print ("+-----+-------------------------+---------+")
print ("| Key | Parameter name          | Value   |")
print ("+-----+-------------------------+---------+")
print ("|        Training parameters table        |")
print ("+-----+-------------------------+---------+")
print ("| -F  | Training folder         | {0:<7} |".format(training_folder))
print ("| -bs | Batch size              | {0:<7} |".format(batch_size))
print ("| -e  | Epochs all              | {0:<7} |".format(epochs_all))
print ("| -s  | Steps per epoch         | {0:<7} |".format(steps_per_epoch))
print ("| -lr | Learing rate            | {0:<7} |".format(learning_rate))
print ("| -rf | LR (-lr) recude factor  | {0:<7} |".format(str(learning_rate_reduce_factor)))
print ("| -aug| Augmentation            | {0:<7} |".format(str(augmentation)))
print ("+-----+-------------------------+---------+")
print ("|        Loss function parameters         |")
print ("+-----+-------------------------+---------+")
print ("| -m  | margin                  | {0:<7} |".format(margin))
print ("| -l1 | lambda1                 | {0:<7} |".format(lambda1))
print ("| -l2 | lambda2                 | {0:<7} |".format(lambda2))
print ("+-----+-------------------------+---------+")
print ("|            Other parameters             |")
print ("+-----+-------------------------+---------+")
print ("| -vis| Make visualisation data | {0:<7} |".format(str(visualisation)))
print ("| -V  | Visualisation folder    | {0:<7} |".format(str(visualisation_folder)))
print ("| -L  | Model weights load file | {0:<7} |".format(str(model_weights_load_file)))
print ("| -S  | Model weights save file | {0:<7} |".format(model_weights_save_file))
print ("+-----+-------------------------+---------+")
print("\n", flush = True)

    
# init loader class. Same benign set to True or False, it doesn't matter for simple cnn
loader = data_loader.Loader(training_folder, ct_folder, cancer_folder, False, augmentation)

# make learning rate schedule
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

optimizer = keras.optimizers.Adam(lr = lr_schedule(0))
#optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.3)
#optimizer = keras.optimizers.RMSprop(lr = learning_rate)
inner_model.compile(
    optimizer=optimizer,
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

# check if user wants to preload existing weights
def preload_weights():
    global inner_model
    if (isArgvKeyPresented("-L")):
        if (model_weights_load_file != None):
            exists = os.path.isfile(model_weights_load_file)
            if exists:
                inner_model = keras.models.load_model(model_weights_load_file)
                return
        print("No weights file found specified at '-L' key!", file=os.sys.stderr)

preload_weights()

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=model_weights_save_file,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]