import keras

from keras.layers import Conv3D, MaxPooling3D, Activation, ReLU, add, BatchNormalization
from keras import regularizers

from models_src.custom_layers import sqr_distance_layer

# Making siamese network for nodules comparison

# More info about Keras Layers: https://keras.io/layers/core/, https://keras.io/layers/convolutional/
# Good presentation of Mail.ru https://logic.pdmi.ras.ru/~sergey/teaching/dl2017/DLNikolenko-MailRu-05.pdf
# ResNet: https://neurohive.io/ru/vidy-nejrosetej/resnet-34-50-101/

# First of all, let's create two input layers.
ct_img1 = keras.layers.Input(shape=(16,16,16))
ct_img2 = keras.layers.Input(shape=(16,16,16))
# We should reshape input for 1 depth color channel, to feed into Conv3D layer
ct_img1_r = keras.layers.Reshape((16,16,16,1))(ct_img1)
ct_img2_r = keras.layers.Reshape((16,16,16,1))(ct_img2)

#ResNet example: https://github.com/raghakot/keras-resnet/blob/master/resnet.py
inner_model_input = keras.layers.Input(shape=(16,16,16,1))

# here types of model:
# https://www.quora.com/What-is-the-VGG-neural-network
# https://medium.com/@sidereal/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5
# trying VGG model

uni_init = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)


# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
      # add convolutional layers
      for _ in range(n_conv):
            layer_in = Conv3D(n_filters, (3,3, 3), padding='same', kernel_initializer='he_normal')(layer_in)
            layer_in = ReLU(negative_slope=0.1)(layer_in)
      # add max pooling layer
      layer_in = MaxPooling3D((2,2,2), strides=2)(layer_in)
      return layer_in
      
block = Conv3D(64, (7,7,7), strides=1, padding='same',
      kernel_initializer='he_normal')(inner_model_input)
merblock1ge_input = ReLU(negative_slope=0.1)(block) #(16, 16, 16)

block = vgg_block(block, 64 , 8) #(8, 8, 8)
block = vgg_block(block, 128, 8) #(4, 4, 4)
block = vgg_block(block, 256, 15) #(2, 2, 2)
block = vgg_block(block, 512, 3) #(1, 1, 1)

fc = keras.layers.Flatten()(block)
fc = keras.layers.Dense(1024, kernel_initializer=uni_init, activation=keras.activations.sigmoid)(fc)
fc = keras.layers.Dense(1024, kernel_initializer=uni_init)(fc)
fc = ReLU(negative_slope=0.1)(fc)
fc = keras.layers.Dense(16, kernel_initializer=uni_init, activation=keras.activations.linear)(fc)

# Next, we should twin this network, and make a layer, that calculates energy between output of two networks
inner_model = keras.Model(inner_model_input, fc)
ct_img_model1 = inner_model(ct_img1_r)
ct_img_model2 = inner_model(ct_img2_r)

merge_layer_lambda = keras.layers.Lambda(sqr_distance_layer)
#merge_layer_lambda = keras.layers.Lambda(difference_layer)
merge_layer = merge_layer_lambda([ct_img_model1, ct_img_model2])
# add FC layer to make similarity score
#merge_layer = keras.layers.Dense(1, activation=keras.activations.sigmoid)(merge_layer)

# Finally, creating model with two inputs 'mnist_img' 1 and 2 and output 'final layer'
model = keras.Model([ct_img1, ct_img2], merge_layer)
#model.summary()