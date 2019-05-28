import keras
import tensorflow as tf
from keras import backend as K
import re
import os
import numpy as np
import time
import utility
import data_loader
from utility import getArgvKeyValue
from utility import isArgvKeyPresented

from keras.layers import Conv3D, MaxPooling3D, Activation, ReLU, add
from keras import regularizers

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
train_pair_count = int(getArgvKeyValue("-tp", 800)) # we take 1500 pairs every training step (75% training)
validation_pair_count = int(getArgvKeyValue("-vp", 100)) # we take 500 pairs for validation (25% validation)
same_benign = isArgvKeyPresented("-sb") # do we need benign-benign pairs in training and validation set?
batch_size = int(getArgvKeyValue("-bs", 100)) # how many pairs form loss function in every training step (2 recomended)
epochs_all = int(getArgvKeyValue("-e", 300)) # global epochs (with pair change)
steps_per_epoch = int(getArgvKeyValue("-s", 3)) # how many steps per epoch available (0.96 acc: 120 for 2 batch size, 300 for 128 batch size)
learning_rate = float(getArgvKeyValue("-lr", 0.000006))
learning_rate_reduce_factor = getArgvKeyValue("-rf")
if learning_rate_reduce_factor is not None:
      learning_rate_reduce_factor = float(learning_rate_reduce_factor)
augmentation = isArgvKeyPresented("-aug")

k = int(getArgvKeyValue("-k", 5)) # knn parameter -- pick k = 5 nearest neibourgs
sigma = float(getArgvKeyValue("-si", 1)) # sigma parameter for distance
lambda1 = float(getArgvKeyValue("-l", 0.0002)) # lambda1
threshold = float(getArgvKeyValue("-th", 10)) # distance for both siamese accuracy and knn distance filter
margin = float(getArgvKeyValue("-m", 1000)) # margin defines how strong dissimilar values are pushed from each other (contrastive loss)

knn = isArgvKeyPresented("-knn")
visualisation = isArgvKeyPresented("-vis")
visualisation_folder = getArgvKeyValue("-V")
model_weights_load_file = getArgvKeyValue("-L") # can be none
model_weights_save_file = getArgvKeyValue("-S", "./lung_cancer_siamese_conv3D.model") # with default value

print("\n")
print ("+-----+-------------------------+---------+")
print ("| Key | Parameter name          | Value   |")
print ("+-----+-------------------------+---------+")
print ("|         Tuning parameters table         |")
print ("+-----+-------------------------+---------+")
print ("| -F  | Training folder         | {0:<7} |".format(training_folder))
print ("| -tp | Train pair count        | {0:<7} |".format(train_pair_count))
print ("| -vp | Validation pair count   | {0:<7} |".format(validation_pair_count))
print ("| -sb | Form benign-benign pair | {0:<7} |".format(str(same_benign)))
print ("| -bs | Batch size              | {0:<7} |".format(batch_size))
print ("| -e  | Epochs all              | {0:<7} |".format(epochs_all))
print ("| -s  | Steps per epoch         | {0:<7} |".format(steps_per_epoch))
print ("| -lr | Learing rate            | {0:<7} |".format(learning_rate))
print ("| -rf | LR (-lr) recude factor  | {0:<7} |".format(str(learning_rate_reduce_factor)))
print ("| -aug| Augmentation            | {0:<7} |".format(str(augmentation)))
print ("+-----+-------------------------+---------+")
print ("| -k  | k                       | {0:<7} |".format(k))
print ("| -si | sigma                   | {0:<7} |".format(sigma))
print ("| -l  | lambda1                 | {0:<7} |".format(lambda1))
print ("| -th | threshold               | {0:<7} |".format(threshold))
print ("| -m  | margin                  | {0:<7} |".format(margin))
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
loader = data_loader.Loader(training_folder, ct_folder, same_benign, augmentation)
# Making siamese network for nodules comparison

# More info about Keras Layers: https://keras.io/layers/core/, https://keras.io/layers/convolutional/
# Good presentation of Mail.ru https://logic.pdmi.ras.ru/~sergey/teaching/dl2017/DLNikolenko-MailRu-05.pdf
# ResNet: https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/

# First of all, let's create two input layers.
ct_img1 = keras.layers.Input(shape=(16,64,64))
ct_img2 = keras.layers.Input(shape=(16,64,64))
# We should reshape input for 1 depth color channel, to feed into Conv3D layer
ct_img1_r = keras.layers.Reshape((16,64,64,1))(ct_img1)
ct_img2_r = keras.layers.Reshape((16,64,64,1))(ct_img2)

# building sequential type of model

#ResNet example: https://github.com/raghakot/keras-resnet/blob/master/resnet.py
inner_model_input = keras.layers.Input(shape=(16,64,64,1))

# here types of model:
# https://www.quora.com/What-is-the-VGG-neural-network
# https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
# Try big sizes of kernel : 11-16
# trying ResNet model

uni_init = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)

# function for creating an identity residual module
# function for creating an identity or projection residual module
def residual_module(layer_in, n_filters):
      merge_input = layer_in
      # check if the number of filters needs to be increase, assumes channels last format
      if layer_in.shape[-1] != n_filters:
            merge_input = Conv3D(n_filters, (1,1,1), padding='same', kernel_initializer='he_normal')(layer_in)
            merge_input = ReLU(negative_slope=0.1)(merge_input)
      # conv1
      conv1 = Conv3D(n_filters, (3,3,3), padding='same', kernel_initializer='he_normal')(layer_in)
      conv1 = ReLU(negative_slope=0.1)(conv1)
      # conv2
      conv2 = Conv3D(n_filters, (3,3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
      # add filters, assumes filters/channels last
      layer_out = add([conv2, merge_input])
      # activation function
      layer_out = ReLU(negative_slope=0.1)(layer_out)
      return layer_out
      
block1 = residual_module(inner_model_input, 16) # (16, 64, 64)
block1 = residual_module(block1, 16)
block1 = residual_module(block1, 16)
block1 = residual_module(block1, 16)
block1 = residual_module(block1, 16)
block1 = residual_module(block1, 16)
block1 = residual_module(block1, 16)
block1 = residual_module(block1, 16)
block2 = keras.layers.MaxPooling3D(pool_size=2)(block1) # (8, 32, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block2 = residual_module(block2, 32)
block3 = keras.layers.MaxPooling3D(pool_size=2)(block2) # (4, 16, 16)
block3 = residual_module(block3, 64)
block3 = residual_module(block3, 64)
block3 = residual_module(block3, 64)
block3 = residual_module(block3, 64)
block3 = residual_module(block3, 64)
block3 = residual_module(block3, 64)
block4 = keras.layers.MaxPooling3D(pool_size=2)(block3) # (2, 8, 8)
block4 = residual_module(block4, 128)
block4 = residual_module(block4, 128)
block4 = residual_module(block4, 128)
block4 = residual_module(block4, 128)
block4 = residual_module(block4, 128)
block4 = residual_module(block4, 128)
block5 = keras.layers.MaxPooling3D(pool_size=2)(block4) # (1, 4, 4)
block5 = residual_module(block5, 256)
block5 = residual_module(block5, 256)
block5 = residual_module(block5, 256)
block5 = residual_module(block5, 256)
block6 = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(block5) # (1, 2, 2)
block6 = residual_module(block6, 512)
block6 = residual_module(block6, 512)
block6 = residual_module(block6, 512)
block6 = residual_module(block6, 512)
flatten = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(block6) # (1, 1, 1)

fc = keras.layers.Flatten()(flatten)
fc = keras.layers.Dense(512, kernel_initializer=uni_init)(fc)
fc = ReLU(negative_slope=0.1)(fc)
fc = keras.layers.Dense(512, kernel_initializer=uni_init, activation=tf.nn.sigmoid)(fc)
fc = keras.layers.Dense(3  , kernel_initializer=uni_init, activation='linear')(fc)

# Next, we should twin this network, and make a layer, that calculates energy between output of two networks
inner_model = keras.Model(inner_model_input, fc)
ct_img_model1 = inner_model(ct_img1_r)
ct_img_model2 = inner_model(ct_img2_r)

# for training
def sqr_distance_layer(tensors):
    # https://github.com/tensorflow/tensorflow/issues/12071
    # print (K.sqrt(K.mean(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True)))
    return K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True)

# for knn
def distance_layer(tensors):
    return K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True))

def difference_layer(tensors):
    # https://github.com/tensorflow/tensorflow/issues/12071
    # print (K.sqrt(K.mean(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True)))
    return K.abs(tensors[0] - tensors[1])

merge_layer_lambda = keras.layers.Lambda(sqr_distance_layer)
#merge_layer_lambda = keras.layers.Lambda(difference_layer)
merge_layer = merge_layer_lambda([ct_img_model1, ct_img_model2])
# add FC layer to make similarity score
#merge_layer = keras.layers.Dense(1, activation=keras.activations.sigmoid)(merge_layer)

# Finally, creating model with two inputs 'mnist_img' 1 and 2 and output 'final layer'
model = keras.Model([ct_img1, ct_img2], merge_layer)
#model.summary()

# parallelizing model on two GPU's
#model = keras.utils.multi_gpu_model(model, gpus=2)

# mean_distance for cancers
def mean_distance(y_true, y_pred):
      return K.mean(y_pred)
      
def mean_contradistance(y_true, y_pred):
    margin_square = lambda1 * K.square(K.maximum(margin - y_pred, 0))
    return K.sum(y_true * margin_square) / K.sum(y_true)

# Model is ready, let's compile it with quality function and optimizer
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    #y_pred = y_pred / K.sum(y_pred)
    square_pred = K.square(y_pred)
    margin_square = lambda1 * K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)

# custom metrics
def siamese_accuracy(y_true, y_pred):
    #https://github.com/tensorflow/tensorflow/issues/23133
    '''Compute classification accuracy with a dynamic threshold on distances.
    '''
    m = mean_distance(y_true, y_pred)
    return K.mean(K.equal(y_true, K.cast(y_pred > m, y_true.dtype)))

    
#https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
#optimizer = keras.optimizers.Adam(lr = learning_rate)
optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.3)
model.compile(
    optimizer=optimizer,
    loss=contrastive_loss,
    metrics=[mean_distance, mean_contradistance, siamese_accuracy]
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
                              , custom_objects={'siamese_accuracy': siamese_accuracy,
                                                'mean_distance': mean_distance,
                                                'mean_contradistance': mean_contradistance,
                                                'contrastive_loss': contrastive_loss})
                        return
            print("No weights file found specified at '-L' key!", file=os.sys.stderr)

preload_weights()

# creating sequencer
training_batch_generator = loader.get_training_generator(batch_size)

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
