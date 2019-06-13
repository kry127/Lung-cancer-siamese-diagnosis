import keras
from keras import backend as K
import re
import os
import numpy as np
import time
import utility
import data_loader
from utility import getArgvKeyValue, isArgvKeyPresented
from models_src.model_ResNet import model
from models_src.custom_layers import distance_layer

vis_regexp = 'vis_(\d\d\d\d).npy'
vis_filename = "vis_{:04d}.npy"
visualisation_index = 0

#starting time count
time_start = time.time()

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
same_benign = isArgvKeyPresented("-sb") # do we need benign-benign pairs in training and validation set?
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
lambda3 = float(getArgvKeyValue("-l3", 1)) # lambda3
lambda4 = float(getArgvKeyValue("-l4", 1)) # lambda4
lambda5 = float(getArgvKeyValue("-l5", 1)) # lambda5
lambda6 = float(getArgvKeyValue("-l6", 1)) # lambda6

# knn parameters
k = int(getArgvKeyValue("-k", 5)) # pick k = 5 nearest neibourgs
sigma = float(getArgvKeyValue("-si", 1)) # sigma parameter for distance
threshold = float(getArgvKeyValue("-th", 10)) # distance for both siamese accuracy and knn distance filter

knn = isArgvKeyPresented("-knn")
if (knn):
    print("Warning! key '-knn' is deprecated and should not be used. It will be switched off")
    knn = False
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
print ("| -sb | Form benign-benign pair | {0:<7} |".format(str(same_benign)))
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
print ("| -l3 | lambda3                 | {0:<7} |".format(lambda3))
print ("| -l4 | lambda4                 | {0:<7} |".format(lambda4))
print ("| -l5 | lambda5                 | {0:<7} |".format(lambda5))
print ("| -l6 | lambda6                 | {0:<7} |".format(lambda6))
print ("+-----+-------------------------+---------+")
print ("|             k-nn parameters             |")
print ("+-----+-------------------------+---------+")
print ("| -k  | k                       | {0:<7} |".format(k))
print ("| -si | sigma                   | {0:<7} |".format(sigma))
print ("| -th | threshold               | {0:<7} |".format(threshold))
print ("+-----+-------------------------+---------+")
print ("|            Other parameters             |")
print ("+-----+-------------------------+---------+")
print ("| -knn| Apply knn stage         | {0:<7} |".format(str(knn)))
print ("| -vis| Make visualisation data | {0:<7} |".format(str(visualisation)))
print ("| -V  | Visualisation folder    | {0:<7} |".format(str(visualisation_folder)))
print ("| -L  | Model weights load file | {0:<7} |".format(str(model_weights_load_file)))
print ("| -S  | Model weights save file | {0:<7} |".format(model_weights_save_file))
print ("+-----+-------------------------+---------+")
print("\n", flush = True)

# param preprocessing
if (isArgvKeyPresented("-V")):
    if (visualisation_folder != None):
        exists = os.path.exists(visualisation_folder)
        if not exists:
            try:
                os.makedirs(visualisation_folder)
            except OSError:
                print("Cannot create visualisation folder", file=os.sys.stderr)
                visualisation_folder = None
            
        exists = os.path.exists(visualisation_folder)
        if exists:
            vis_regexp = re.compile(vis_regexp)
            filelist = os.listdir(visualisation_folder)
            vislist = filter(lambda s: vis_regexp.match(s) != None, filelist)
            visnum = list(map(lambda s: int(vis_regexp.match(s).group(1)), vislist))
            if (len(visnum) > 0):
                visualisation_index = max(visnum) + 1
            print("Current visualisation index = {}".format(visualisation_index))
        
    else:
        print("No path specified at -V key", file=os.sys.stderr)

# init loader class
loader = data_loader.Loader(training_folder, ct_folder, cancer_folder, same_benign, augmentation)
# Making siamese network for nodules comparison

# importing model
# as follows: "from models_src.model_simple import model"

# parallelizing model on two GPU's
#model = keras.utils.multi_gpu_model(model, gpus=2)

# mean_distance for cancers
def mean_distance(y_true, y_pred):
    Dw = lambda1 * y_pred
    return K.sum((1 - y_true) * Dw)/K.sum(1 - y_true)
      
def mean_contradistance(y_true, y_pred):
    Cw = lambda2 * K.maximum(margin - y_pred, 0)
    return K.sum(y_true * Cw) / K.sum(y_true)

# Model is ready, let's compile it with quality function and optimizer
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    #y_pred = y_pred / K.sum(y_pred)
    Dw = lambda1 * K.square(y_pred)
    Cw = lambda2 * K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * Dw + y_true * Cw)

# custom metrics
# obsolete, just abuse of computational resources
def siamese_accuracy(y_true, y_pred):
    #https://github.com/tensorflow/tensorflow/issues/23133
    '''Compute custom classification accuracy.
    '''
    #m = mean_distance(y_true, y_pred)
    #return K.mean(K.equal(y_true, K.cast(y_pred > threshold, y_true.dtype)))
    Dw = lambda1 * y_pred
    Dw_accuracy = 1 - K.sum(K.cast((1 - y_true) * Dw > threshold, "float32"))/K.sum(1 - y_true)
    Cw = lambda2 * K.maximum(margin - y_pred, 0)
    Cw_accuracy = (K.sum(K.cast(K.equal(y_true * Cw, 0), "float32")) - K.sum(1 - y_true)) / K.sum(y_true)
    return (Dw_accuracy + Cw_accuracy) / 2

    
#https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
optimizer = keras.optimizers.Adam(lr = learning_rate)
#optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.3)
model.compile(
    optimizer=optimizer,
    loss=contrastive_loss,
    metrics=[mean_distance, mean_contradistance]
)

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

# creating sequencer
training_batch_generator = loader.get_training_pair_generator(batch_size)

# saving model is easy
#https://stackoverflow.com/questions/52553593/tensorflow-keras-model-save-raise-notimplementederror
def save_weights(epoch, logs):
      global model
      exists = os.path.isfile(model_weights_save_file)
      if exists:
            print('Overwriting model {}'.format(model_weights_save_file))
      else:
            print('Saving model {}'.format(model_weights_save_file))

      model.save(model_weights_save_file)
      return


# functions for final testing (validation)
def calc_diff_distance(hash_benign, hash_malignant):
    rho_benign_malignant = np.array([])
    for hash_benign_nodule in hash_benign:
        for hash_malignant_nodule in hash_malignant:
            rho = distance_layer(
                        [np.expand_dims(hash_benign_nodule, axis=0),
                        np.expand_dims(hash_malignant_nodule, axis=0)]
                        )
            rho_benign_malignant = np.append(rho_benign_malignant, K.get_value(rho)[0])
    return rho_benign_malignant

def calc_same_distance(hash):
    rho_arr = np.array([])
    for i in range(0, len(hash)):
        nodule1 = hash[i]
        mask = np.ones(len(hash), np.bool)
        mask[i] = 0
        for nodule2 in hash[mask]:
            rho = distance_layer(
                        [np.expand_dims(nodule1, axis=0),
                        np.expand_dims(nodule2, axis=0)]
                        )
            rho_arr = np.append(rho_arr, K.get_value(rho)[0])
    return rho_arr
            

def knn_for_nodule_hash(hash_nodule, hash_benign, hash_malignant, k, threshold, sigma):
    # ввести арбитраж на основе расстояния
    # например, на основе экспонентациальной функции (e^-x)
    # Для вычисления расстояния использовать sqr_distance_layer
    rho_benign = np.array([])
    rho_malignant = np.array([])
    for hash_benign_nodule in hash_benign:
          rho = distance_layer(
                  [np.expand_dims(hash_nodule, axis=0),
                  np.expand_dims(hash_benign_nodule, axis=0)]
                  )
          rho_benign = np.append(rho_benign, K.get_value(rho)[0])
    for hash_malignant_nodule in hash_malignant:
          rho = distance_layer(
                  [np.expand_dims(hash_nodule, axis=0),
                  np.expand_dims(hash_malignant_nodule, axis=0)]
                  )
          rho_malignant = np.append(rho_malignant, K.get_value(rho)[0])

    # учитывая расстояние threshold, отсекаем и сортируем данные
    rho_benign = np.sort(rho_benign[np.where(rho_benign < threshold)])
    rho_malignant = np.sort(rho_malignant[np.where(rho_malignant < threshold)])

    # insufficient amount of neigbours
    if (len(rho_benign) + len(rho_malignant) < 1):
      return None

    # далее, необходимо ввести экспонентациальную зависимость (e^-x) от каждого ближайшего соседа
    # (гауссово распределение)
    # по закону трёх сигм: sigma = threshold / 3. СТОИТ ЛИ ВВОДИТЬ?
    weighter = np.vectorize(lambda x: np.sign(x)/sigma*np.e ** -(((x/sigma)**2)/2) )
    #weighter = np.vectorize(lambda x: x )

    rho = np.append(-rho_benign, rho_malignant)
    rho_weights = weighter(rho)
    
    # TODO sort descending by module
    rho_abs_weights = np.vectorize(lambda val: np.abs(val))(rho_weights)
    rho_abs_weights_id_sorted = np.argsort(rho_abs_weights)
    rho_weights = rho_weights[rho_abs_weights_id_sorted]

    # TODO choose k biggest by module weights
    # TODO sum chosen weights and pass to heavyside function
    result = np.sum(rho_weights[-k:])

    if (result > 0):
          return 1 # malignant
    elif (result <= 0):
          return 0 # benign

def end_checks(epoch = None, logs = None):
    if not knn and not visualisation:
        return
    # common data for final checks
    inner_model = model.layers[4]
    # apply inner_model to validation set
    global train_hash_benign
    global train_hash_malignant
    global validation_hash_benign
    global validation_hash_malignant
    
    train_hash_benign = inner_model.predict([np.expand_dims(loader.data_benign, axis=-1)])
    train_hash_malignant = inner_model.predict([np.expand_dims(loader.data_malignant, axis=-1)])
    validation_hash_benign = inner_model.predict([np.expand_dims(loader.data_validation_benign, axis=-1)])
    validation_hash_malignant = inner_model.predict([np.expand_dims(loader.data_validation_malignant, axis=-1)])
    if knn:
        knn_check()
    if visualisation:
        vis()

def knn_check():
    #calculate accuracies
    true_benign = 0
    benign_predictions = np.array([])
    for i in range(0, len(validation_hash_benign)):
        nodule = validation_hash_benign[i]
        #mask = np.ones(len(validation_hash_benign), np.bool)
        #mask[i] = 0
        #res = knn_for_nodule_hash(nodule, validation_hash_benign[mask], validation_hash_malignant, k, threshold, sigma)
        res = knn_for_nodule_hash(nodule, train_hash_benign, train_hash_malignant, k, threshold, sigma)
        benign_predictions = np.append(benign_predictions, [res])
        if res == 0:
                true_benign += 1
    true_malignant = 0
    malignant_predictions = np.array([])
    for i in range(0, len(validation_hash_malignant)):
        nodule = validation_hash_malignant[i]
        #mask = np.ones(len(validation_hash_benign), np.bool)
        #mask[i] = 0
        #res = knn_for_nodule_hash(nodule, validation_hash_benign, validation_hash_malignant[mask], k, threshold, sigma)
        res = knn_for_nodule_hash(nodule, train_hash_benign, train_hash_malignant, k, threshold, sigma)
        malignant_predictions = np.append(malignant_predictions, [res])
        if res == 1:
                true_malignant += 1
    
    benign_accuracy = true_benign / len(validation_hash_benign)
    malignant_accuracy = true_malignant / len(validation_hash_malignant)

    print("Computing knn distance-weighted accuracy on validation set")
    print("Knn params: k={}, threshold={}, sigma = {}".format(k, threshold, sigma))
    print("benign_accuracy = {}".format(benign_accuracy))
    print("malignant_accuracy = {}".format(malignant_accuracy))
    print("accuracy = {}".format((benign_accuracy + malignant_accuracy)/2))
    print("")
    print("Benign prediction array: {}".format(benign_predictions))
    print("Malignant prediction array: {}".format(malignant_predictions))


    dist_diff = calc_diff_distance(validation_hash_benign, validation_hash_malignant)
    dist_benign = calc_same_distance(validation_hash_benign)
    dist_malignant = calc_same_distance(validation_hash_malignant)
    print("Distances    benign-benign.    Min={}, Max={}, Mean={}".format(np.min(dist_benign), np.max(dist_benign), np.mean(dist_benign)))
    print("Distances    benign-malignant. Min={}, Max={}, Mean={}".format(np.min(dist_diff), np.max(dist_diff), np.mean(dist_diff)))
    print("Distances malignant-malignant. Min={}, Max={}, Mean={}".format(np.min(dist_malignant), np.max(dist_malignant), np.mean(dist_malignant)))

def vis():
    global visualisation_index
    
    inner_model = model.layers[4]
    # apply inner_model to validation set
    

    print()
    print("Visualisation data production stage")

    hash_type = 0 # train benign
    hashes = np.insert(train_hash_benign, 0, hash_type, axis=1)
    hash_type = 1 # train malignant
    hashes = np.append(hashes, np.insert(train_hash_malignant, 0, hash_type, axis=1), axis=0)
    hash_type = 2 # validation benign
    hashes = np.append(hashes, np.insert(validation_hash_benign, 0, hash_type, axis=1), axis=0)
    hash_type = 3 # validation malignant
    hashes = np.append(hashes, np.insert(validation_hash_malignant, 0, hash_type, axis=1), axis=0)

    if visualisation_folder != None:
        filepath = os.path.join(visualisation_folder, vis_filename.format(visualisation_index))
        np.savetxt(filepath, hashes)
        visualisation_index += 1


# creating model checkpoints
save_callback = keras.callbacks.LambdaCallback(on_epoch_end=save_weights)
end_checks_callback = keras.callbacks.LambdaCallback(on_epoch_end=end_checks)
callbacks = [save_callback, end_checks_callback,
      keras.callbacks.TerminateOnNaN(),
]

if learning_rate_reduce_factor != None:
      callbacks += [keras.callbacks.ReduceLROnPlateau(monitor='loss',
            factor=learning_rate_reduce_factor, verbose=1,
            patience=10, mode='min', cooldown=50, min_lr=0.00000001)]

# The model is ready to train!
if steps_per_epoch <= 0:
      steps_per_epoch = None
      if epochs_all <= 0:
          # no actual training, launch knn and vis if needed
          end_checks()

model.fit_generator(generator=training_batch_generator,
      epochs=epochs_all,
      steps_per_epoch = steps_per_epoch,
      verbose=1,
      shuffle=True,
      callbacks = callbacks)
