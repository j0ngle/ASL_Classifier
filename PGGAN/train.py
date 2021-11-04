import numpy as np
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, Add
from keras import backend
from networks import *
from helpers import *

#######################
#   HYPERPARAMETERS   #
#######################



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
                                     apply_smoothing=False, 
                                     apply_noise=False)

    #Get gradients
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    #Train
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))



  #Pretrain D
  elif d_pretrain > 0:
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



  #Pretrain G
  elif d_pretrain < 0:
    noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])

    with tf.GradientTape() as disc_tape:
      #Generate images
      generated_images = generator(noise, training=False)

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

    for i in range(-d_pretrain):
      with tf.GradientTape() as gen_tape:
        #Generate images
        generated_images = generator(noise, training=True)

        #Get D output for fake images
        fake_output = discriminator(generated_images, training=False)

        #G loss
        gen_loss = generator_loss(fake_output,
                                  apply_smoothing=smooth)
        
        #Get and apply gradients
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))


  return gen_loss, disc_loss

def update_alpha(a, generator, discriminator):
  print("Updating alpha (a={})...\n".format(a))

  for layer in generator.layers:
    if isinstance(layer, WeightedSum):
      backend.set_value(layer.alpha, a)
  for layer in discriminator.layers:
    if isinstance(layer, WeightedSum):
      backend.set_value(layer.alpha, a)



def train_gan(path, generator, discriminator, epochs=50, plot_step=1, ckpt_step=1):

  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                discriminator_optimizer=disc_optimizer,
                                generator=generator,
                                discriminator=discriminator)

  for depth in range(1, 7):
    dataset = prepare_dataset(path, FILTERS[7-depth], BATCH_SIZE, SAMPLE_SIZE)

    generator, generator_stable = fade_G(generator, depth)
    discriminator, discriminator_stable = fade_D(discriminator, depth)

    print(generator.summary())
    print(generator_stable.summary())

    depth_loss_G = np.array([])
    depth_loss_D = np.array([])

    print("\nStarting depth {}...\n".format(depth))
    for epoch in range(epochs):
      print("\nStarting epoch {}/{}...\n".format(epoch, epochs))

      update_alpha(epoch / epochs, generator, discriminator)

      epoch_loss_G = []
      epoch_loss_D = []

      for batch in dataset:
        g_loss, d_loss = train_step(batch,
                                    generator, 
                                    discriminator, 
                                    d_pretrain=3)

        epoch_loss_G.append(g_loss)
        epoch_loss_D.append(d_loss)

      depth_loss_G = np.append(depth_loss_G, np.array([epoch_loss_G]))
      depth_loss_D = np.append(depth_loss_D, np.array([epoch_loss_D]))

      
      #Swtich to stablized models
      print("Stablizing...\n")
      generator = generator_stable
      discriminator = discriminator_stable

      stable_epoch_loss_G = []
      stable_epoch_loss_D = []

      for batch in dataset:
        g_loss, d_loss = train_step(batch, 
                                    generator, 
                                    discriminator, 
                                    d_pretrain=3)

        stable_epoch_loss_G.append(g_loss)
        stable_epoch_loss_D.append(d_loss)

      print_statistics(epoch_loss_G, "Generator")
      print_statistics(epoch_loss_D, "Discriminator")
      print_statistics(stable_epoch_loss_G, "Stable Generator")
      print_statistics(stable_epoch_loss_D, "Stable Discriminator")

      if ((epoch + 1) % plot_step == 0) or epoch == 0:
        noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])
        generated_images = generator(noise, training=False)
        plot_multiple_images(generated_images, epoch+1, 'epoch_grids', 8)
        # plt.show()

        plot_metrics('GD_epoch_loss_D{}'.format(depth), "iterations", "loss", epoch,
                      epoch_loss_G, "Generator", 
                      epoch_loss_D, "Discriminator")

        plot_metrics('S_GD_epoch_loss_D{}'.format(depth), "iterations",
                      "loss", epoch,
                      stable_epoch_loss_G, "Generator", 
                      stable_epoch_loss_D, "Discriminator")

        plot_metrics('GD_total_depth_loss_D{}'.format(depth), "iterations", "loss", epoch,
                      depth_loss_G, "Generator", 
                      depth_loss_D, "Discriminator")

        #TODO: Show_single image
        # noise = tf.random.normal([1, CODINGS_SIZE])
        # image = generator(noise, training=False)
        # show_image(image, epoch)

      if (epoch + 1) % ckpt_step == 0:
        print("Saving checkpoint...\n")
        checkpoint.save(file_prefix = checkpoint_prefix)

    print_statistics(depth_loss_G, "Depth {}: Generator".format(depth))
    print_statistics(depth_loss_D, "Depth {}: Discriminator".format(depth))

    #Save images at n depth
    noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])
    generated_images = generator(noise, training=False)
    plot_multiple_images(generated_images, epoch, "depth_grids", 8)
    # plt.show()

    # noise = tf.random.normal([1, CODINGS_SIZE])
    # image = generator(noise, training=False)
    # show_image(image, epoch)

    #Save Model at n depth
    generator.save_weights("/depth_checkpoints/depth_{}.ckpt".format(depth))

  #Save model
  generator.save_weights("final_model.ckpt")

def train_without_growth(path, generator, discriminator, epochs=50, plot_step=1, ckpt_step=1):
  dataset = prepare_dataset(path, FILTERS[7-depth], BATCH_SIZE, SAMPLE_SIZE)

  total_loss_G = np.array([])
  total_loss_D = np.array([])

  for epoch in len(epochs):
    epoch_loss_G = []
    epoch_loss_D = []

    for batch in dataset:
      g_loss, d_loss = train_step(batch,
                                  generator, 
                                  discriminator, 
                                  d_pretrain=3)

      epoch_loss_G.append(g_loss)
      epoch_loss_D.append(d_loss)

    total_loss_G = np.append(total_loss_G, np.array([epoch_loss_G]))
    total_loss_D = np.append(total_loss_D, np.array([epoch_loss_D]))

    print_statistics(epoch_loss_G, "Generator")
    print_statistics(epoch_loss_D, "Discriminator")

    if ((epoch + 1) % plot_step == 0) or epoch == 0:
      noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])
      generated_images = generator(noise, training=False)
      plot_multiple_images(generated_images, epoch+1, 'epoch_grids', 8)
      # plt.show()

      plot_metrics('GD_epoch_loss_D{}'.format(depth), "iterations", "loss", epoch,
                    epoch_loss_G, "Generator", 
                    epoch_loss_D, "Discriminator")

      plot_metrics('GD_total_depth_loss_D{}'.format(depth), "iterations", "loss", epoch,
                    total_loss_G, "Generator", 
                    total_loss_D, "Discriminator")

############
#   MAIN   #
############

if __name__ == "__main__":
  PATH = "D:/School/landscape/"
  # dataset = prepare_dataset(PATH, 4, BATCH_SIZE, SAMPLE_SIZE)

  generator = init_generator()
  discriminator = init_discriminator()

  # train_gan(PATH, generator, discriminator, epochs=2, plot_step=1, ckpt_step=10)
  train_without_growth(PATH, generator, discriminator, epochs=5, plot_step=1, ckpt_step=10)
  