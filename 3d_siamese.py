import keras
import tensorflow as tf
from keras import backend as K
import os
import numpy as np
import time
import utility
import data_loader
from utility import getArgvKeyValue
from utility import isArgvKeyPresented

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
learning_rate = float(getArgvKeyValue("-lr",0.000006))

k = int(getArgvKeyValue("-k", 5)) # knn parameter -- pick k = 5 nearest neibourgs
sigma = float(getArgvKeyValue("-si", 1)) # sigma parameter for distance
lambda1 = float(getArgvKeyValue("-l", 0.0002)) # lambda1
threshold = float(getArgvKeyValue("-th", 1)) # distance for both siamese accuracy and knn distance filter
margin = float(getArgvKeyValue("-m", 1000)) # margin defines how strong dissimilar values are pushed from each other (contrastive loss)

knn = isArgvKeyPresented("-knn")
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
print ("| -L  | Model weights load file | {0:<7} |".format(str(model_weights_load_file)))
print ("| -S  | Model weights save file | {0:<7} |".format(model_weights_save_file))
print ("+-----+-------------------------+---------+")
print("\n", flush = True)

# init loader class
loader = data_loader.Loader(training_folder, ct_folder, same_benign)
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
inner_model = keras.models.Sequential()

# here types of model:
# https://www.quora.com/What-is-the-VGG-neural-network
# https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
# Try big sizes of kernel : 11-16

inner_model.add(keras.layers.Conv3D(64, kernel_size=11,
            activation=tf.nn.relu,
            strides=1, kernel_initializer = "he_normal",
            input_shape=(16,64,64,1))) # (6, 54, 54)
inner_model.add(keras.layers.Conv3D(128, kernel_size=(5, 5, 5),
            strides=1, kernel_initializer = "he_normal",
            activation=tf.nn.relu)) # (2, 50, 50)
inner_model.add(keras.layers.BatchNormalization())
inner_model.add(keras.layers.Activation("relu"))
inner_model.add(keras.layers.SpatialDropout3D(0.1))
inner_model.add(keras.layers.MaxPooling3D(pool_size=(2, 2, 2))) # (1, 25, 25)


inner_model.add(keras.layers.Conv3D(256, kernel_size=(1, 6, 6),
            strides=1, kernel_initializer = "he_normal",
            activation=tf.nn.relu)) # (1, 20, 20)
inner_model.add(keras.layers.Conv3D(512, kernel_size=(1, 5, 5),
            strides=1, kernel_initializer = "he_normal",
            activation=tf.nn.relu)) # (1, 16, 16)
inner_model.add(keras.layers.BatchNormalization())
inner_model.add(keras.layers.Activation("relu"))
inner_model.add(keras.layers.SpatialDropout3D(0.1))
inner_model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2))) # (1, 8, 8)

inner_model.add(keras.layers.Conv3D(1024, kernel_size=(1, 5, 5),
            strides=1, kernel_initializer = "he_normal",
            activation=tf.nn.relu)) # (1, 4, 4)
inner_model.add(keras.layers.Conv3D(2048, kernel_size=(1, 3, 3),
            strides=1, kernel_initializer = "he_normal",
            activation=tf.nn.relu)) # (1, 2, 2)
inner_model.add(keras.layers.BatchNormalization())
inner_model.add(keras.layers.Activation("relu"))
inner_model.add(keras.layers.SpatialDropout3D(0.1))
inner_model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2))) # (1, 1, 1)

# Then, we should flatten last layer
# Avoid OOM!
# https://stackoverflow.com/questions/53658501/out-of-memory-oom-error-of-tensorflow-keras-model
inner_model.add(keras.layers.Flatten())
inner_model.add(keras.layers.Dense(2048, activation=tf.nn.relu, kernel_initializer = "he_normal",))
inner_model.add(keras.layers.Dense(256, activation=tf.nn.relu,  kernel_initializer = "he_normal",))
inner_model.add(keras.layers.Dense(64, activation=keras.activations.sigmoid,  kernel_initializer = "he_normal",))

# Next, we should twin this network, and make a layer, that calculates energy between output of two networks

ct_img_model1 = inner_model(ct_img1_r)
ct_img_model2 = inner_model(ct_img2_r)

def sqr_distance_layer(tensors):
    # https://github.com/tensorflow/tensorflow/issues/12071
    # print (K.sqrt(K.mean(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True)))
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
model = keras.utils.multi_gpu_model(model, gpus=2)

# mean_distance for cancers
def mean_distance(y_true, y_pred):
      return K.mean(y_pred)

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
def siamese_accuracy_far(y_true, y_pred):
    #https://github.com/tensorflow/tensorflow/issues/23133
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # about Keras backend: https://stackoverflow.com/questions/49950130/logical-and-or-in-keras-backend
    y_true = K.cast(y_true, 'bool')
    dist_bool_mask = K.cast(y_pred > threshold, 'bool')
    tp = K.sum(K.cast(keras.backend.all(keras.backend.stack([y_true, dist_bool_mask], axis=0), axis=0), 'float32'))
    return tp / K.sum(K.cast(y_true, 'float32'))

def siamese_accuracy_close(y_true, y_pred):
    #https://github.com/tensorflow/tensorflow/issues/23133
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    # about Keras backend: https://stackoverflow.com/questions/49950130/logical-and-or-in-keras-backend
    y_false = K.cast(K.equal(y_true, K.cast(False, y_true.dtype)), 'bool')
    dist_bool_mask = K.cast(y_pred <= threshold, 'bool')
    fp = K.sum(K.cast(keras.backend.all(keras.backend.stack([y_false, dist_bool_mask], axis=0), axis=0), 'float32'))
    return fp / K.sum(K.cast(y_false, 'float32'))

def knn_for_nodule(nodule, k, threshold, sigma):
    # ввести арбитраж на основе расстояния
    # например, на основе экспонентациальной функции (e^-x)
    rho_benign = model.predict([np.tile(nodule, (len(loader.data_benign), 1, 1, 1)), loader.data_benign])
    rho_malignant = model.predict([np.tile(nodule, (len(loader.data_malignant), 1, 1, 1)), loader.data_malignant])

    # учитывая расстояние threshold, отсекаем и сортируем данные
    rho_benign = np.sort(rho_benign[np.where(rho_benign < threshold)])
    rho_malignant = np.sort(rho_malignant[np.where(rho_malignant < threshold)])

    # insufficient amount of neigbours
    if (len(rho_benign) + len(rho_malignant) < k):
      return None

    # далее, необходимо ввести экспонентациальную зависимость (e^-x) от каждого ближайшего соседа
    # (гауссово распределение)
    # по закону трёх сигм: sigma = threshold / 3. СТОИТ ЛИ ВВОДИТЬ?
    weighter = np.vectorize(lambda x: np.sign(x)/sigma*np.e ** -(((x/sigma)**2)/2) )

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


def knn_benign_accuracy(k, threshold, sigma):
      N = 0
      t = 0
      for benign_nodule in loader.data_validation_benign:
            result = knn_for_nodule(benign_nodule, k, threshold, sigma)
            N += 1
            if result == 0:
                  t += 1
      return t / N

def knn_malignant_accuracy(k, threshold, sigma):
      N = 0
      t = 0
      for malignant_nodule in loader.data_validation_malignant:
            result = knn_for_nodule(malignant_nodule, k, threshold, sigma)
            N += 1
            if result == 1:
                  t += 1
      return t / N

    
#https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
optimizer = keras.optimizers.Adam(lr = learning_rate)
#optimizer = keras.optimizers.SGD(lr=0.0005, momentum=0.3)
model.compile(
    optimizer=optimizer,
    loss=contrastive_loss,
    metrics=[mean_distance, siamese_accuracy_far, siamese_accuracy_close]
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
                              , custom_objects={'siamese_accuracy_far': siamese_accuracy_far,
                                                'siamese_accuracy_close': siamese_accuracy_close,
                                                'mean_distance': mean_distance,
                                                'contrastive_loss': contrastive_loss})
                        return
            print("No weights file found specified at '-L' key!", file=os.sys.stderr)

preload_weights()
    

# forming pairs from validation
time_start_load = time.time()
print("Start forming validation tuples at {0:.3f} seconds".format(time_start_load - time_start))
validation_tuple = loader.form_pairs(validation_pair_count,
                  loader.data_validation_benign, loader.data_validation_malignant)
t_end = time.time()
print("Validation tuples formed at {0:.3f} in {1:.3f} sec.".format(t_end - time_start, t_end - time_start_load))
print()

# The model is ready to train!
for N in range(1, epochs_all+1):
    form_pairs_start_time = time.time()
    pairs, pairs_y = loader.form_pairs(train_pair_count,
                        loader.data_benign, loader.data_malignant)
    print("Pairs formation: {0:.3f} seconds".format(time.time() - form_pairs_start_time))
    print("Epoch #{}/{} ".format(str(N), epochs_all))
    model.fit(pairs, pairs_y, epochs = steps_per_epoch, verbose=2, batch_size=batch_size
                  , validation_data = validation_tuple)
    #print("Batch {}, validation accuracy: {}".format(str(N), knn_accuracy(threshold)))
    # лучше сделать подсчёт accuracy по ПАРАМ на валидационной выборке
    # knn_accuracy сделать на ТЕСТОВОЙ

# saving model is easy
#https://stackoverflow.com/questions/52553593/tensorflow-keras-model-save-raise-notimplementederror
def save_weights():
      global model
      exists = os.path.isfile(model_weights_save_file)
      if exists:
            print('Overwriting model {}'.format(model_weights_save_file))
      else:
            print('Saving model {}'.format(model_weights_save_file))

      model.save(model_weights_save_file)
      return

save_weights()

# final testing
if knn:
      print("Computing knn distance-weighted accuracy on validation set")
      print("Knn params: k={}, threshold={}, sigma = {}".format(k, threshold, sigma))
      benign_accuracy = knn_benign_accuracy(k, threshold, sigma)
      malignant_accuracy = knn_malignant_accuracy(k, threshold, sigma)
      print("benign_accuracy = {}".format(benign_accuracy))
      print("malignant_accuracy = {}".format(malignant_accuracy))
      print("accuracy = {}".format((benign_accuracy + malignant_accuracy)/2))
