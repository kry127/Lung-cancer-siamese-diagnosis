import keras

from keras import backend as K
from keras.layers import Conv3D, MaxPooling3D, Activation, ReLU, add
from keras import regularizers

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
fc = keras.layers.Dense(512, kernel_initializer=uni_init, activation=keras.activations.sigmoid)(fc)
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