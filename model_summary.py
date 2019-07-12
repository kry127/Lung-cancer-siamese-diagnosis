import keras
from keras import backend as K
import os
from utility import getArgvKeyValue, isArgvKeyPresented

model_weights_load_file = getArgvKeyValue("-L") # can be none

print ("+-----+-------------------------+---------+")
print ("|            Other parameters             |")
print ("+-----+-------------------------+---------+")
print ("| -L  | Model weights load file | {0:<7} |".format(str(model_weights_load_file)))
print ("+-----+-------------------------+---------+")

def mean_distance(y_true, y_pred):
    Dw = 1 * y_pred
    return K.sum((1 - y_true) * Dw)/K.sum(1 - y_true)
      
def mean_contradistance(y_true, y_pred):
    Cw = 1 * K.maximum(1 - y_pred, 0)
    return K.sum(y_true * Cw) / K.sum(y_true)

# Model is ready, let's compile it with quality function and optimizer
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    #y_pred = y_pred / K.sum(y_pred)
    Dw = 1 * K.square(y_pred)
    Cw = 1 * K.square(K.maximum(1 - y_pred, 0))
    return K.mean((1 - y_true) * Dw + y_true * Cw)

# check if user wants to preload existing weights
def preload_weights():
    global model
    if (isArgvKeyPresented("-L")):
        if (model_weights_load_file != None):
            exists = os.path.isfile(model_weights_load_file)
            if exists:

                # we should load it with custom objects
                # https://github.com/keras-team/keras/issues/5916
                model = keras.models.load_model(model_weights_load_file
                        , custom_objects={'mean_distance': mean_distance,
                                        'mean_contradistance': mean_contradistance,
                                        'contrastive_loss': contrastive_loss})
                return
        print("No weights file found specified at '-L' key!", file=os.sys.stderr)

preload_weights()

model.summary()