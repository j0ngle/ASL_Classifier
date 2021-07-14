import tensorflow as tf
from tensorflow import keras

IMG_SIZE        = 64
LEAKY_SLOPE     = 0.2
DROPOUT         = 0.4
CODINGS_SIZE    = 128
SCALE           = 16
WEIGHT_STD      = 0.02
WEIGHT_MEAN     = 0

scaled_size = IMG_SIZE // SCALE
weight_init = tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_STD, mean=WEIGHT_MEAN, seed=42)

def conv_transpose(model, out_channels, k_size, s_size, batch_normalize=True):
  model.add(keras.layers.Conv2DTranspose(out_channels, kernel_size=(k_size, k_size), 
                                         strides=(s_size, s_size), padding='same', use_bias=False,
                                         kernel_initializer=weight_init))
  if batch_normalize is True:
    model.add(keras.layers.BatchNormalization(momentum=0.9))

  #model.add(keras.layers.ReLU())
  model.add(keras.layers.LeakyReLU(LEAKY_SLOPE))

  return model


def conv(model, out_channels, k_size, s_size, batch_normalize=True):
  model.add(keras.layers.Conv2D(out_channels, kernel_size=(k_size, k_size), 
                                strides=(s_size, s_size), padding='same',
                                kernel_initializer=weight_init))
  if batch_normalize is True:
    model.add(keras.layers.BatchNormalization(momentum=0.9))

  model.add(keras.layers.LeakyReLU(LEAKY_SLOPE))

  return model


def generator():
  model = keras.models.Sequential()

  model.add(keras.layers.Dense(scaled_size * scaled_size * 128, input_shape=(CODINGS_SIZE,), kernel_initializer=weight_init))
  model.add(keras.layers.Reshape([scaled_size, scaled_size, 128]))
  
  model = conv_transpose(model, 512, k_size=5, s_size=1)
  model.add(keras.layers.Dropout(DROPOUT))

  model = conv_transpose(model, 256, k_size=5, s_size=2)
  model.add(keras.layers.Dropout(DROPOUT))

  model = conv_transpose(model, 128, k_size=5, s_size=2)
  model.add(keras.layers.Dropout(DROPOUT))

  model = conv_transpose(model, 64, k_size=5, s_size=2)
  model.add(keras.layers.Dropout(DROPOUT))

  model = conv_transpose(model, 32, k_size=5, s_size=2)

  model.add(keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(1, 1), 
                                         padding='same', activation='tanh', 
                                         use_bias=True, kernel_initializer=weight_init))
  # model.add(keras.layers.Dense(3, activation='tanh', kernel_initializer=weight_init))

  return model


def discriminator():
  model = keras.models.Sequential()

  model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), 
                              padding='same', activation=keras.layers.LeakyReLU(LEAKY_SLOPE), 
                              input_shape=[64, 64, 3], kernel_initializer=weight_init)) 

  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 64, k_size=5, s_size=2)

  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 128, k_size=5, s_size=2)

  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 256, k_size=5, s_size=2)

  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 512, k_size=5, s_size=2)

  #model.add(keras.layers.Lambda(minibatch_discrimination))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  return model