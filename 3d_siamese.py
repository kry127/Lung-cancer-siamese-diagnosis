import tensorflow as tf
from keras import backend as K
import os
import numpy as np

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

# print found classes
print("beingn: {}, malignant: {}".format(
      len(benign_set), len(malignant_set)))

# forming training and validation set
train_count = 4 #should be 70
validation_count = 8 #should be 30

np.random.shuffle(malignant_set)
np.random.shuffle(benign_set)

train_malignant = malignant_set[:train_count]
train_benign = benign_set[:train_count]

validation_malignant = malignant_set[train_count:train_count+validation_count]
validation_benign = benign_set[train_count:train_count+validation_count]

# print found classes
print("test_beingn: {}, test_malignant: {}, validation_benign: {}, validation_malignant: {}".format(
      len(train_benign), len(train_malignant), len(validation_malignant), len(validation_benign)))

# load data 
data_benign = np.ndarray((0,16,64,64))
for benign in train_benign: #go through benign examples
    valarr = benign.split('_')
    if (valarr[1] == "img"):
        data = np.load(os.path.join(ct_folder, benign))
        data_x = np.append(data_benign, [data], axis = 0)
        # data augmentation: there are 8 flips of image
        data_benign = np.append(data_benign, [np.flip(data, (0))], axis = 0)
        data_benign = np.append(data_benign, [np.flip(data, (1))], axis = 0)
        data_benign = np.append(data_benign, [np.flip(data, (2))], axis = 0)
        data_benign = np.append(data_benign, [np.flip(data, (0, 1))], axis = 0)
        data_benign = np.append(data_benign, [np.flip(data, (1, 2))], axis = 0)
        data_benign = np.append(data_benign, [np.flip(data, (0, 2))], axis = 0)
        data_benign = np.append(data_benign, [np.flip(data, (0, 1, 2))], axis = 0)

        
data_malignant = np.ndarray((0,16,64,64))
for malignant in train_malignant: #go through malignant examples
    valarr = malignant.split('_')
    if (valarr[1] == "img"):
        data = np.load(os.path.join(cancer_folder, malignant))
        data_malignant = np.append(data_malignant, [data], axis = 0)
        # data augmentation: there are 8 flips of image
        data_malignant = np.append(data_malignant, [np.flip(data, (0))], axis = 0)
        data_malignant = np.append(data_malignant, [np.flip(data, (1))], axis = 0)
        data_malignant = np.append(data_malignant, [np.flip(data, (2))], axis = 0)
        data_malignant = np.append(data_malignant, [np.flip(data, (0, 1))], axis = 0)
        data_malignant = np.append(data_malignant, [np.flip(data, (1, 2))], axis = 0)
        data_malignant = np.append(data_malignant, [np.flip(data, (0, 2))], axis = 0)
        data_malignant = np.append(data_malignant, [np.flip(data, (0, 1, 2))], axis = 0)

# Making siamese network for nodules comparison

# More info about Keras Layers: https://keras.io/layers/core/, https://keras.io/layers/convolutional/

# First of all, let's create two input layers.
ct_img1 = tf.keras.layers.Input(shape=(16,64,64))
ct_img2 = tf.keras.layers.Input(shape=(16,64,64))
# We should reshape input for 1 depth color channel, to feed into Conv3D layer
ct_img1_r = tf.keras.layers.Reshape((16,64,64,1))(ct_img1)
ct_img2_r = tf.keras.layers.Reshape((16,64,64,1))(ct_img2)

# building sequential type of model
model = tf.keras.models.Sequential()

# Let's create simple Conv2D layer, 28 batches, 10x10 kernels each
model.add(tf.keras.layers.Conv3D(32, kernel_size=9,
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

# TODO -- Remove last layer from 
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

def knn_accuracy(y_true, y_pred):
    threshold = 1

#https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
optimizer = tf.keras.optimizers.Adam(lr = 0.000006)
#optimizer = tf.keras.optimizers.SGD(lr=0.0005, momentum=0.3)
model.compile(
    optimizer=optimizer,
    loss=contrastive_loss,
    metrics=['accuracy', siamese_accuracy]
)

# automatically forming training pairs
# N is half of of the batch size
def form_pairs_auto(Nhalf):
    pairs = np.ndarray((0,2,16,64,64))
    pairs_y = np.ndarray((0, 1))
    for i in range (0, Nhalf):
      A = data_benign[np.random.randint(data_benign.shape[0]),:,:,:]
      B = data_malignant[np.random.randint(data_malignant.shape[0]),:,:,:]
      C = data_benign[np.random.randint(data_benign.shape[0]),:,:,:]
      D = data_malignant[np.random.randint(data_malignant.shape[0]),:,:,:]
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
    return pairs, pairs_y

# The model is ready to train!
for k in range(1, 100):
    batch_size_quarter = 5
    pairs, pairs_y = form_pairs_auto(batch_size_quarter)
    print("Batch " + str(k))
    model.fit([pairs[0], pairs[1]], pairs_y, epochs = 10, batch_size=4*batch_size_quarter)

# saving model is easy
#https://stackoverflow.com/questions/52553593/tensorflow-keras-model-save-raise-notimplementederror
model.save('./lung_cancer_siamese_conv3D.model')

# loading model is also simple
#new_model = tf.keras.models.load_model('lung_cancer_siamese_conv3D.model')