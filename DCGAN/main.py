import numpy as np
import cv2
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import urllib
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from helpers import *
import helpers
import networks

def process():
    print("Processing images...")

    PATH = 'landscapes_scaled.csv'

    landscapes_flat = helpers.get_csv_path(PATH, '')
    length = len(landscapes_flat['images'])
    landscapes = [None] * length

    for i in range(length):

        if i % 500 == 0:
            print("   ", (i / length) * 100, "% complete")

        #Grab image
        img = landscapes_flat['images'][i]
        
        #Get substring (omitting brackets)
        img = img[1:len(img) - 1]

        #Convert to array, reshape, and add to new list
        landscapes[i] = np.fromstring(img, sep=' ').reshape(64, 64, 3)

    

    landscapes = np.asarray(landscapes)

    # helpers.plot_multiple_images(landscapes[:40], n_cols=8)
    # plt.show()

    dataset = tf.data.Dataset.from_tensor_slices(landscapes).shuffle(SAMPLE_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(1)

    print("Processing complete!")

    return dataset
    

def train_step(images):
  noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = helpers.generator_loss(fake_output, apply_smoothing=True)
    disc_loss = helpers.discriminator_loss(real_output, fake_output, apply_smoothing=True, apply_noise=True)

  gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
  disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
  disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

  return gen_loss, disc_loss


def train_gan(dataset, epochs=50, plot_step=1, ckpt_step=20):
    print("Training...")

    all_gl = np.array([])
    all_dl = np.array([])
    G_mean = np.array([])
    D_mean = np.array([])

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))     

        G_loss = []
        D_loss = []

        for batch in dataset:
            g_loss, d_loss = train_step(batch)
            G_loss.append(g_loss)
            D_loss.append(d_loss)

        all_gl = np.append(all_gl, np.array([G_loss]))
        all_dl = np.append(all_dl, np.array([D_loss]))

        #Print
        print_statistics(G_loss, "Generator loss")
        print_statistics(D_loss, "Discriminator loss")
        # print("Generator Loss Mean:", np.mean(G_loss), "Std:", np.std(G_loss))
        # print("Discriminator Loss Mean:", np.mean(D_loss), "Std:", np.std(D_loss))
        print()

        G_mean = np.append(G_mean, np.mean(G_loss))
        D_mean = np.append(D_mean, np.mean(D_loss))

        if ((epoch + 1) % plot_step == 0) or epoch == 0:
            #Generate test images
            noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])
            generated_images = generator(noise, training=False)

            plot_metrics("G-D loss", "Iterations", "Loss", epoch, G_loss, "G", D_loss, "D")
            plot_metrics("G-D_loss_total", "Iterations", "Loss", epoch, all_gl, "G", all_dl, "D")

            save_grid(generated_images, epoch+1, 'grids', 8) 

        if (epoch + 1) % ckpt_step == 0:
            #Save checkpoint
            checkpoint.save(file_prefix = checkpoint_prefix)


if __name__ == '__main__':
    #Hyperparameters
    SAMPLE_SIZE     = 1000
    BATCH_SIZE      = 32
    CODINGS_SIZE    = 128
    IMG_SIZE        = 32
    LEARNING_RATE_G = 0.0003
    LEARNING_RATE_D = 0.0001

    PATH = "D:/School/landscape/"

    dataset = prepare_dataset(PATH, IMG_SIZE, BATCH_SIZE, SAMPLE_SIZE)

    generator = networks.generator()
    discriminator = networks.discriminator()

    gen_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, beta_1=.5)
    disc_optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, beta_1=0.5)  
    #disc_optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE_D)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                    discriminator_optimizer=disc_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    train_gan(dataset, epochs=300, plot_step=20, ckpt_step=50)

    noise = tf.random.normal(shape=[BATCH_SIZE, CODINGS_SIZE])
    final_images = generator(noise, training=False)

    helpers.plot_multiple_images(final_images, 200, 8)

    