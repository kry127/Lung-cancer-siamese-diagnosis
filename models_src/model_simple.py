import keras

from keras import backend as K
from keras.layers import Conv3D, SpatialDropout3D, MaxPooling3D, Dropout, Flatten, Dense, ReLU
from keras import regularizers

from custom_layers import sqr_distance_layer

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

# initializer for last weights
uni_init = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)

# building sequential type of model
inner_model = keras.models.Sequential()

# trying VGG-like model
# https://www.quora.com/What-is-the-VGG-neural-network
# VGG is bad :(
# here another types:
# https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
# Try big sizes of kernel : 11-16
inner_model.add(Conv3D(512, kernel_size=12,
            strides=4, input_shape=(16,64,64,1))) # (2, 14, 14)
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(SpatialDropout3D(0.1))
inner_model.add(MaxPooling3D(pool_size=(1, 2, 2))) # (1, 7, 7)


inner_model.add(Conv3D(1024, kernel_size=(1, 4, 4))) # (1, 4, 4)
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(SpatialDropout3D(0.1))
inner_model.add(MaxPooling3D(pool_size=(1, 2, 2))) # (1, 2, 2)

# Then, we should flatten last layer
# Avoid OOM!
# https://stackoverflow.com/questions/53658501/out-of-memory-oom-error-of-tensorflow-keras-model
inner_model.add(Flatten())
inner_model.add(Dense(4096, kernel_initializer=uni_init))
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Dropout(0.1))
inner_model.add(Dense(1024, kernel_initializer=uni_init))
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Dropout(0.1, kernel_initializer=uni_init))
inner_model.add(Dense(256, activation='linear'))

# Next, we should twin this network, and make a layer, that calculates energy between output of two networks

ct_img_model1 = inner_model(ct_img1_r)
ct_img_model2 = inner_model(ct_img2_r)

merge_layer_lambda = keras.layers.Lambda(sqr_distance_layer)
merge_layer = merge_layer_lambda([ct_img_model1, ct_img_model2])

# Finally, creating model with two inputs 'mnist_img' 1 and 2 and output 'final layer'
model = keras.Model([ct_img1, ct_img2], merge_layer)
#model.summary()