import keras

from keras import backend as K
from keras.layers import Conv3D, SpatialDropout3D, MaxPooling3D, AvgPool3D, Dropout, Flatten, Dense, ReLU
from keras import regularizers

from models_src.custom_layers import distance_layer

ct_img1 = keras.layers.Input(shape=(16,16,16))
ct_img2 = keras.layers.Input(shape=(16,16,16))
ct_img1_r = keras.layers.Reshape((16,16,16,1))(ct_img1)
ct_img2_r = keras.layers.Reshape((16,16,16,1))(ct_img2)

# initializer for last weights
uni_init = keras.initializers.RandomUniform(minval=-0.7, maxval=0.7, seed=None)


# building sequential type of inner_model
inner_model = keras.models.Sequential()

inner_model.add(Conv3D(64, kernel_size=5, input_shape=(16,16,16,1))) # (12, 12, 12)
inner_model.add(ReLU(negative_slope=0.1))

inner_model.add(Conv3D(128, kernel_size=3)) # (10, 10, 10)
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Conv3D(192, kernel_size=3)) # (8, 8, 8)
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Conv3D(792, kernel_size=3))  # (6, 6, 6)
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Conv3D(1536, kernel_size=3))  # 4
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Conv3D(3084, kernel_size=3))  # 2
inner_model.add(AvgPool3D(pool_size=(1, 2, 2))) # (1, 1, 1)
inner_model.add(ReLU(negative_slope=0.1))

# Then, we should flatten last layer
inner_model.add(Flatten())
inner_model.add(Dense(2048, kernel_initializer=uni_init))
inner_model.add(ReLU(negative_slope=0.1))
inner_model.add(Dense(2048, kernel_initializer=uni_init, activation=keras.activations.sigmoid))
inner_model.add(Dense(16,  kernel_initializer=uni_init, activation='linear'))

ct_img_model1 = inner_model(ct_img1_r)
ct_img_model2 = inner_model(ct_img2_r)

merge_layer_lambda = keras.layers.Lambda(distance_layer)
merge_layer = merge_layer_lambda([ct_img_model1, ct_img_model2])

# Finally, creating model with two inputs 'mnist_img' 1 and 2 and output 'final layer'
model = keras.Model([ct_img1, ct_img2], merge_layer)