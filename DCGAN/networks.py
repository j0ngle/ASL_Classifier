import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer

IMG_SIZE        = 32
LEAKY_SLOPE     = 0.2
DROPOUT         = 0.4
CODINGS_SIZE    = 128
SCALE           = 16
WEIGHT_STD      = 0.02
WEIGHT_MEAN     = 0

scaled_size = IMG_SIZE // SCALE
weight_init = tf.keras.initializers.TruncatedNormal(stddev=WEIGHT_STD, mean=WEIGHT_MEAN, seed=42)

class Minibatch(Layer):
    def __init__(self, **kwargs):
        super(Minibatch, self).__init__(**kwargs)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        avg_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch = tf.tile(avg_stddev, (shape[0], shape[1], shape[2], 1))
        
        return tf.concat([inputs, minibatch], axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1

        return tuple(input_shape)

def conv_transpose(model, out_channels, k_size, s_size, batch_normalize=True):
  model.add(keras.layers.Conv2DTranspose(out_channels, kernel_size=(k_size, k_size), 
                                         strides=(s_size, s_size), padding='same', use_bias=False,
                                         kernel_initializer=weight_init))
  if batch_normalize is True:
    model.add(keras.layers.BatchNormalization(momentum=0.9))

  #model.add(keras.layers.ReLU())
  model.add(keras.layers.ReLU())

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


  #2x2
  model = conv_transpose(model, 1024, k_size=5, s_size=2)
  model.add(keras.layers.Dropout(DROPOUT))

  #4x4
  model = conv_transpose(model, 512, k_size=5, s_size=2)
  model.add(keras.layers.Dropout(DROPOUT))

  #8x8
  model = conv_transpose(model, 256, k_size=5, s_size=2)
  model.add(keras.layers.Dropout(DROPOUT))

  #16x16
  model = conv_transpose(model, 128, k_size=5, s_size=2)

  #32x32
  model.add(keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(1, 1), 
                                         padding='same', activation='tanh', 
                                         use_bias=True, kernel_initializer=weight_init))
  # model.add(keras.layers.Dense(3, activation='tanh', kernel_initializer=weight_init))

  return model


def discriminator():
  model = keras.models.Sequential()

  #32x32
  model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), 
                              padding='same', activation=keras.layers.LeakyReLU(LEAKY_SLOPE), 
                              input_shape=[IMG_SIZE, IMG_SIZE, 3], kernel_initializer=weight_init)) 

  

  #16x16
  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 256, k_size=5, s_size=2)

  #8x8
  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 512, k_size=5, s_size=2)

  #4x4
  #model.add(keras.layers.Dropout(DROPOUT))
  model = conv(model, 1024, k_size=5, s_size=2)

  #2x2
  #model.add(keras.layers.Lambda(minibatch_discrimination))

  # model.add(Minibatch())

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(1, activation='sigmoid'))

  return model