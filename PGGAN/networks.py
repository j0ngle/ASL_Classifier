import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Add
from keras import backend
from helpers import smooth_labels, noisy_labels
from train import FILTERS, LEAKY_SLOPE, CODINGS_SIZE
from train import scaled_size, cross_entropy

######################
#   CUSTOM CLASSES   #
######################

class WeightScaling(Layer):
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        shape = np.asarray(shape)
        shape = tf.constant(shape, dtype=tf.float32)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain * tf.math.rsqrt(fan_in)
    
    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        return inputs * self.wscale
    
    def compute_output_shape(self, input_shape):
        return input_shape

class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        
        return inputs * l2
    
    def compute_output_shape(self, input_shape):
        return input_shape

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

class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))

        return output

#################
#   LAYER DEF   #
#################

def WS_Dense(model, filters, gain, use_pixelnorm=False, activation=None):
    init = keras.initializers.RandomNormal(mean=0, stddev=1.)
    in_filters = backend.int_shape(model)[-1]

    model = keras.layers.Dense(filters, use_bias=True, kernel_initializer=init, dtype='float32')(model)
    model = WeightScaling(shape=(in_filters), gain=gain)(model)

    if activation == 'LeakyReLU':
        model = keras.layers.LeakyReLU(LEAKY_SLOPE)(model)
    elif activation == 'tanh':
        model = keras.layers.activation('tanh')(model)

    if use_pixelnorm:
        model = PixelNormalization()(model)

    return model

def WS_Conv(model, filters, k_size, strides, gain, use_pixelnorm, activation):
    init = keras.initializers.RandomNormal(mean=0, stddev=1.)
    in_filters = backend.int_shape(model)[-1]

    model = keras.layers.Conv2D(filters, kernel_size=k_size, strides=strides, 
                            use_bias=True, padding='same', kernel_initializer=init,
                            dtype='float32')(model)
    model = WeightScaling(shape=(k_size[0], k_size[1], in_filters), gain=gain)(model)

    if activation == 'LeakyReLU':
        model = keras.layers.LeakyReLU(LEAKY_SLOPE)(model)
    elif activation == 'tanh':
        model = keras.layers.Activation('tanh')(model)

    if use_pixelnorm:
        model = PixelNormalization()(model)

    return model

#################
#   GENERATOR   #
#################

def init_generator():
    input_ = keras.layers.Input(shape=(CODINGS_SIZE,))

    model = PixelNormalization()(input_)

    model = WS_Dense(model, 4*4*FILTERS[0], gain=np.sqrt(2)/4, 
                    activation='LeakyReLU', use_pixelnorm=True)

    model = keras.layers.Reshape([scaled_size, scaled_size, FILTERS[0]])(model)

    model = WS_Conv(model, filters=FILTERS[0], k_size=(4, 4), strides=(1, 1), 
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=True)
    model = WS_Conv(model, filters=FILTERS[0], k_size=(3, 3), strides=(1, 1), 
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=True)


    model = WS_Conv(model, filters=3, k_size=(1, 1), strides=(1, 1), gain=1,
                    activation='tanh', use_pixelnorm=False)   #toRGB

    generator = keras.Model(input_, model, name='generator')

    return generator

def fade_G(generator, depth):
    #Double generated image size
    block = generator.layers[-4].output               #pixel_normalization_7
    block = keras.layers.UpSampling2D((2, 2))(block)  #Doubling using upscaling

    #Grabbing old output
    old_G = generator.layers[-3](block)               #conv2d_8        
    old_G = generator.layers[-2](old_G)               #weight_scaling_11
    old_G = generator.layers[-1](old_G)               #activation_1 

    #New block
    new_G = WS_Conv(block, filters=FILTERS[depth], k_size=(3, 3), strides=(1, 1),
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=True)
    new_G = WS_Conv(new_G, filters=FILTERS[depth], k_size=(3, 3), strides=(1, 1),
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=True)
    new_G = WS_Conv(new_G, filters=3, k_size=(1, 1), strides=(1, 1),
                    gain=1., activation='tanh', use_pixelnorm=False)  #linear in paper

    G_stable = keras.models.Model(generator.input, new_G, name='generator')

    #New generator that outputs the weighted sum of both old and new G
    new_generator = WeightedSum()([old_G, new_G])

    #Compile
    new_generator = keras.models.Model(generator.input, new_generator, name='generator')

    return new_generator, G_stable

def generator_loss(fake_output, apply_smoothing=True):
    fake_output_mod = fake_output

    if apply_smoothing:
        fake_output_mod = smooth_labels(fake_output_mod, 'negative')

    return cross_entropy(tf.ones_like(fake_output_mod), fake_output)

#####################
#   DISCRIMINATOR   #
#####################

def init_discriminator():
    input_ = keras.layers.Input(shape=(4, 4, 3))
    # input_ = tf.cast(input_, tf.float32)

    model = WS_Conv(input_, FILTERS[0], k_size=(1, 1), strides=(1, 1), 
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=False) #fromRGB

    model = Minibatch()(model)

    model = WS_Conv(model, FILTERS[0], k_size=(3, 3), strides=(1, 1), 
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=False)
    model = WS_Conv(model, FILTERS[0], k_size=(4, 4), strides=(4, 4), 
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=False)

    model = keras.layers.Flatten()(model)

    model = WS_Dense(model, filters=1, gain=1.)

    discriminator = keras.Model(input_, model, name='discriminator')

    return discriminator

def fade_D(discriminator, depth):
    #Double out input shape
    input_shape = list(discriminator.input.shape)
    input_shape = (input_shape[1]*2, input_shape[2]*2, input_shape[3])

    new_input_ = keras.layers.Input(shape=input_shape)
    # new_input_ = tf.cast(new_input_, tf.float32)

    #Grab old input from D
    old_D = keras.layers.AveragePooling2D()(new_input_) #Downscale input
    old_D = discriminator.layers[1](old_D)              #conv2d_3
    old_D = discriminator.layers[2](old_D)              #weight_scaling_4
    old_D = discriminator.layers[3](old_D)              #leaky_re_lu_3

    #Define new D with double input
    new_D = WS_Conv(new_input_, filters=FILTERS[depth], k_size=(1, 1), strides=(1, 1),
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=False)
    new_D = WS_Conv(new_D, filters=FILTERS[depth], k_size=(3, 3), strides=(1, 1),
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=False)
    new_D = WS_Conv(new_D, filters=FILTERS[depth-1], k_size=(3, 3), strides=(1, 1),
                    gain=np.sqrt(2), activation='LeakyReLU', use_pixelnorm=False)
    new_D = keras.layers.AveragePooling2D()(new_D)

    #New D with weightedSum
    new_discriminator = WeightedSum()([old_D, new_D])

    #Rebuild stable D
    for i in range(4, len(discriminator.layers)):
        new_D = discriminator.layers[i](new_D)
    D_stable = keras.models.Model(new_input_, new_D, name='discriminator')

    #Connect new D with remaining layers from old
    for i in range(4, len(discriminator.layers)):
        new_discriminator = discriminator.layers[i](new_discriminator)
    new_discriminator = keras.models.Model(new_input_, new_discriminator, name='discriminator')

    return new_discriminator, D_stable

def discriminator_loss(real_output, fake_output, apply_smoothing=True, apply_noise=True):
    real_output_mod = real_output
    fake_output_mod = fake_output

    if apply_noise:
        real_output_mod = noisy_labels(tf.ones_like(real_output_mod), 0.1)
        fake_output_mod = noisy_labels(tf.zeros_like(fake_output_mod), 0.1)

    if apply_smoothing:
        real_output_mod = smooth_labels(real_output_mod, 'positive')
        fake_output_mod = smooth_labels(fake_output_mod, 'negative')

    #Instead of cross_entropy, try:
    #fake mean - real mean
    #Gradient penalty
    #Drift for regularization

    #Or try cross entropy
    real_loss = cross_entropy(tf.ones_like(real_output_mod), real_output) #labels, outputs
    fake_loss = cross_entropy(tf.zeros_like(fake_output_mod), fake_output)

    return real_loss + fake_loss