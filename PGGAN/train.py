import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Add
from keras import backend
from networks import *
from helpers import *

#######################
#   HYPERPARAMETERS   #
#######################

SAMPLE_SIZE      = 1000
BATCH_SIZE       = 32
LEAKY_SLOPE      = 0.2
DROPOUT          = 0.4
CODINGS_SIZE     = 128 #Might increase size later
FILTERS          = [512, 256, 128, 64, 32, 16, 8, 4]
WEIGHT_STD       = 0.02
WEIGHT_MEAN      = 0
LEARNING_RATE_G  = 0.0001
LEARNING_RATE_D  = 0.0002
scaled_size      = 4

gen_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=.5)
disc_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=0.5)
#disc_optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE_D)
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)

#################
#   FUNCTIONS   #
#################

def train_step(images, generator, discriminator, d_pretrain=5, smooth=False, noise=False):
  #No pretraining
  if d_pretrain == 0:
    noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      #Generate Images
      generated_images = generator(noise, training=True)

      #Send real and fake images through D
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      #Calculate loss
      gen_loss = generator_loss(fake_output,
                                apply_smoothing=smooth)
      disc_loss = discriminator_loss(real_output, 
                                     fake_output, 
                                     apply_smoothing=smooth, 
                                     apply_noise=noise)

    #Get gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    #Train
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

  #With pretrianing
  else:
    #Pretrain D for d_pretrain steps
    for i in range(d_pretrain):
      noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])

      with tf.GradientTape() as disc_tape:
        #Generate images
        generated_images = generator(noise, training=True)

        #Send real and fake images through D
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        #get D loss
        disc_loss = discriminator_loss(real_output, 
                                       fake_output, 
                                       apply_smoothing=smooth, 
                                       apply_noise=False)
        
        #Get and apply D gradients
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    #Train G
    with tf.GradientTape() as gen_tape:
      #Generate images
      generated_images = generator(noise, training=True)

      #Get D output for fake images
      fake_output = discriminator(generated_images, training=True)

      #G loss
      gen_loss = generator_loss(fake_output,
                                apply_smoothing=smooth)
      
      #Get and apply gradients
      gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
      gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

  return gen_loss, disc_loss

def update_alpha(a, generator, discriminator):
  for layer in generator.layers:
    if isinstance(layer, WeightedSum):
      backend.set_value(layer.alpha, a)
  for layer in discriminator.layers:
    if isinstance(layer, WeightedSum):
      backend.set_value(layer.alpha, a)



def train_gan(path, generator, discriminator, epochs=50, plot_step=1):

    checkpoint_dir = '/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                  discriminator_optimizer=disc_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

    #Metrics

    for depth in range(1, 7):
        dataset = prepare_dataset(path, FILTERS[7-depth])

        generator, generator_stable = fade_G(generator, depth)
        discriminator, discriminator_stable = fade_D(discriminator, depth)

        #Metrics
        print("Starting depth {}...\n".format(depth))
        for epoch in range(epochs):
            print("Starting epoch {}/{}...\n".format(epoch, epochs))

            update_alpha(epoch / epochs, generator, discriminator)

            #Metrics

            for batch in dataset:
                g_loss, d_loss = train_step(batch,
                                            generator, 
                                            discriminator, 
                                            d_pretrain=3)

                #More metrics

            
            #Swtich to stablized models
            print("Stablizing...\n")
            generator = generator_stable
            discriminator = discriminator_stable

            #Metrics

            for batch in dataset:
                g_loss, d_loss = train_step(batch, 
                                            generator, 
                                            discriminator, 
                                            d_pretrain=3)

            #Metrics

            #Print epoch metrics

            #Generator test images and display if epoch % plot_step == 0
    

if __name__ == "__main__":
  path = ""
  dataset = process_batch(path, 4)

  generator = init_generator()
  discriminator = init_discriminator()

  train_gan(path, generator, discriminator, epochs=50, plot_step=1)