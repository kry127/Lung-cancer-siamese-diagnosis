import tensorflow as tf
from keras import backend as K
import os
import numpy as np
import time

#starting time count
time_start = time.time()

# Keras partial weight loading info:
# https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras

ct_folder = 'all2' # folder with all computer tomography images
cancer_folder = 'cancer' # folder with cancerous tomography images

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

# forming training and validation set
train_count = 100 # should be 100
validation_count = 30 # should be 50
train_pair_count = 800 # we take 1500 pairs every training step (75% training)
validation_pair_count = 100 # we take 500 pairs for validation (25% validation)
batch_size = 100 # how many pairs form loss function in every training step (2 recomended)
epochs_all = 300 # global epochs (with pair change)
steps_per_epoch = 1 # how many steps per epoch available (0.96 acc: 120 for 2 batch size, 300 for 128 batch size)

print("Training batch size = {}".format(batch_size))

#halven the train count and validation count (for two classes)
train_count = int(train_count/2)
validation_count = int(validation_count/2)

np.random.shuffle(malignant_set)
np.random.shuffle(benign_set)

train_malignant = malignant_set[:train_count]
train_benign = benign_set[:train_count]

validation_malignant = malignant_set[train_count:train_count+validation_count]
validation_benign = benign_set[train_count:train_count+validation_count]

# TODO make save and load method of the list of training data

# print found classes
print("train_beingn: {}, train_malignant: {}, validation_benign: {}, validation_malignant: {}".format(
      len(train_benign), len(train_malignant), len(validation_malignant), len(validation_benign)))

# load data 
def load_train_data(dataset_list, augment = None):
      data_nodules = np.ndarray((0,16,64,64))
      for nodule in dataset_list: #go through benign examples
            valarr = nodule.split('_')
            if (valarr[1] == "img"):
                  data = np.load(os.path.join(ct_folder, nodule))
                  data_nodules = np.append(data_nodules, [data], axis = 0)
                  # data augmentation: there are 8 flips of image
                  if augment:
                        data_nodules = np.append(data_nodules, [np.flip(data, (0))], axis = 0)
                        data_nodules = np.append(data_nodules, [np.flip(data, (1))], axis = 0)
                        data_nodules = np.append(data_nodules, [np.flip(data, (2))], axis = 0)
                        data_nodules = np.append(data_nodules, [np.flip(data, (0, 1))], axis = 0)
                        data_nodules = np.append(data_nodules, [np.flip(data, (1, 2))], axis = 0)
                        data_nodules = np.append(data_nodules, [np.flip(data, (0, 2))], axis = 0)
                        data_nodules = np.append(data_nodules, [np.flip(data, (0, 1, 2))], axis = 0)
      return data_nodules


time_start_load = time.time()
print("Start loading data at {} sec.".format(time_start_load - time_start))
data_benign = load_train_data(train_benign, augment = True)
data_malignant = load_train_data(train_malignant, augment = True)
t_end = time.time()
print("Training data loaded at {} sec. in {} sec.".format(t_end - time_start, t_end - time_start_load))

time_start_load = time.time()
data_validation_benign = load_train_data(validation_benign)
data_validation_malignant = load_train_data(validation_malignant)
t_end = time.time()
print("Validation data loaded at {} in {} sec.".format(t_end - time_start, t_end - time_start_load))

# Making siamese network for nodules comparison

# More info about Keras Layers: https://keras.io/layers/core/, https://keras.io/layers/convolutional/
# Good presentation of Mail.ru https://logic.pdmi.ras.ru/~sergey/teaching/dl2017/DLNikolenko-MailRu-05.pdf
# ResNet: https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/

# First of all, let's create two input layers.
ct_img1 = tf.keras.layers.Input(shape=(16,64,64))
ct_img2 = tf.keras.layers.Input(shape=(16,64,64))
# We should reshape input for 1 depth color channel, to feed into Conv3D layer
ct_img1_r = tf.keras.layers.Reshape((16,64,64,1))(ct_img1)
ct_img2_r = tf.keras.layers.Reshape((16,64,64,1))(ct_img2)

# building sequential type of model
model = tf.keras.models.Sequential()

# Let's create simple Conv2D layer, 28 batches, 10x10 kernels each
model.add(tf.keras.layers.Conv3D(128, kernel_size=9,
            activation=tf.nn.relu, input_shape=(16,64,64,1))) # (8, 56, 56)
# Then, let's add a subsampling layer (https://keras.io/layers/pooling/)
model.add(tf.keras.layers.MaxPooling3D(pool_size=2)) # (4, 28, 28)

# add layer #2, assume 8 new features from each level 1 feature, filter 6x6
model.add(tf.keras.layers.Conv3D(128, kernel_size=4, activation=tf.nn.relu)) # (1, 25, 25)

# adding reshape for 3D-data conversion to 2D-data
#model.add(tf.keras.layers.Reshape((25, 25, 1)))
# for some reason it didn't work

# Here, we can make 3x3 floating filters with max pool, (this converged to (N, 1, 1) dimension)
model.add(tf.keras.layers.Conv3D(256, kernel_size=(1, 8, 8), activation=tf.nn.relu)) # (18, 18)
model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))) # (9, 9)
model.add(tf.keras.layers.Conv3D(512, kernel_size=(1, 4, 4), activation=tf.nn.relu)) # (6, 6)
model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2))) # (3, 3)
model.add(tf.keras.layers.Conv3D(1024, kernel_size=(1, 3, 3), activation=tf.nn.relu)) # (1, 1)

# Then, we should flatten last layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.sigmoid))

# Next, we should twin this network, and make a layer, that calculates energy between output of two networks

ct_img_model1 = model(ct_img1_r)
ct_img_model2 = model(ct_img2_r)

def lambda_layer(tensors):
    # https://github.com/tensorflow/tensorflow/issues/12071
    # print (K.sqrt(K.mean(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True)))
    return K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims = True)

merge_layer_lambda = tf.keras.layers.Lambda(lambda_layer)
merge_layer = merge_layer_lambda([ct_img_model1, ct_img_model2])

# Finally, creating model with two inputs 'mnist_img' 1 and 2 and output 'final layer'
model = tf.keras.Model([ct_img1, ct_img2], merge_layer)
#model.summary()

# Model is ready, let's compile it with quality function and optimizer
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 3 # margin defines how strong dissimilar values are pushed from each other
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)

# custom metrics
def siamese_accuracy(y_true, y_pred):
    threshold = 1
    #https://github.com/tensorflow/tensorflow/issues/23133
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred > threshold, y_true.dtype)))

def knn_for_nodule(nodule, k = 5, threshold = 1):
    # ввести арбитраж на основе расстояния
    # например, на основе экспонентациальной функции (e^-x)
    rho_benign = model.predict([np.tile(nodule, (len(data_benign), 1, 1, 1)), data_benign])
    rho_malignant = model.predict([np.tile(nodule, (len(data_malignant), 1, 1, 1)), data_malignant])

    # учитывая расстояние threshold, отсекаем и сортируем данные
    rho_benign = np.sort(rho_benign[np.where(rho_benign < threshold)])
    rho_malignant = np.sort(rho_malignant[np.where(rho_malignant < threshold)])

    # insufficient amount of neigbours
    if (len(rho_benign) + len(rho_malignant) < 5):
      return None

    # далее, необходимо ввести экспонентациальную зависимость (e^-x) от каждого ближайшего соседа
    # (гауссово распределение)
    # по закону трёх сигм: sigma = threshold / 3. СТОИТ ЛИ ВВОДИТЬ?
    weighter = np.vectorize(lambda x: np.sign(x)*np.e ** -np.abs(x))

    rho = np.append(-rho_benign, rho_malignant)
    rho_weights = weighter(rho)
    
    # TODO sort descending by module
    rho_abs_weights = np.vectorize(lambda val: np.abs(val))(rho_weights)
    rho_abs_weights_id_sorted = np.argsort(rho_abs_weights)
    rho_weights = rho_weights[rho_abs_weights_id_sorted]

    # TODO choose 5 biggest by module weights
    # TODO sum chosen weights and pass to heavyside function
    result = np.sum(rho_weights[-5:])

    if (result > 0):
          return 1 # malignant
    elif (result <= 0):
          return 0 # benign

    
#https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
optimizer = tf.keras.optimizers.Adam(lr = 0.000006)
#optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.3)
model.compile(
    optimizer=optimizer,
    loss=contrastive_loss,
    metrics=['accuracy', siamese_accuracy]
)

def knn_accuracy(k = 5, threshold = 1):
      N = 0
      t = 0
      for benign_nodule in data_validation_benign:
            result = knn_for_nodule(benign_nodule, k, threshold)
            N += 1
            if result == 0:
                  t += 1
      for malignant_nodule in data_validation_malignant:
            result = knn_for_nodule(malignant_nodule, k, threshold)
            N += 1
            if result == 1:
                  t += 1
      return t / N

# automatically forming training pairs
# N is half of of the batch size
def form_pairs_auto(Nhalf, benign, malignant):
    pairs = np.ndarray((0,2,16,64,64))
    pairs_y = np.ndarray((0, 1))
    for i in range (0, Nhalf):
      # replace = False ~ no repeats
      benign_index = np.random.choice(benign.shape[0], 2, replace=False)
      malignant_index = np.random.choice(malignant.shape[0], 2, replace=False)
      A = benign[benign_index[0],:,:,:]
      B = malignant[malignant_index[0],:,:,:]
      C = benign[benign_index[1],:,:,:]
      D = malignant[malignant_index[1],:,:,:]
      # different
      pairs = np.append(pairs, [np.array([A, B])], axis=0)
      pairs_y = np.append(pairs_y, 1) 
      pairs = np.append(pairs, [np.array([C, D])], axis=0)
      pairs_y = np.append(pairs_y, 1) 
      # same
      pairs = np.append(pairs, [np.array([A, C])], axis=0)
      pairs_y = np.append(pairs_y, 0)
      pairs = np.append(pairs, [np.array([B, D])], axis=0)
      pairs_y = np.append(pairs_y, 0)

    pairs = np.swapaxes(pairs, 0, 1)
    return list(pairs), pairs_y

    

# forming pairs from validation
time_start_load = time.time()
print("Start forming validation tuples at {} seconds".format(time_start_load - time_start))
validation_tuple = form_pairs_auto(int(np.ceil(validation_pair_count/4)),
                  data_validation_benign, data_validation_malignant)
t_end = time.time()
print("Validation tuples formed at {} in {} sec.".format(t_end - time_start, t_end - time_start_load))

# The model is ready to train!
for N in range(1, epochs_all+1):
    pairs, pairs_y = form_pairs_auto(int(np.ceil(train_pair_count/4)),
                  data_benign, data_malignant)
    print("Epoch #{}/{} ".format(str(N), epochs_all))
    model.fit(pairs, pairs_y, epochs = steps_per_epoch, verbose=2, batch_size=batch_size
                  , validation_data = validation_tuple)
    #print("Batch {}, validation accuracy: {}".format(str(N), knn_accuracy(threshold = 1)))
    # лучше сделать подсчёт accuracy по ПАРАМ на валидационной выборке
    # knn_accuracy сделать на ТЕСТОВОЙ

# saving model is easy
#https://stackoverflow.com/questions/52553593/tensorflow-keras-model-save-raise-notimplementederror
print('Saving model')
model.save('./lung_cancer_siamese_conv3D.model')

# loading model is also simple
#new_model = tf.keras.models.load_model('lung_cancer_siamese_conv3D.model')

# final testing
print("Computing knn distance-weighted accuracy on validation set")
k = 5
threshold = 1
#accuracy = knn_accuracy(k, threshold)
#print("accuracy(k={}, threshold={}) = {}".format(k, threshold, accuracy))

for k in range(5, 12, 2):
#      for threshold in [0.5, 1, 2]:
            accuracy = knn_accuracy(k, threshold)
            print("accuracy(k={}, threshold={}) = {}".format(k, threshold, accuracy))
