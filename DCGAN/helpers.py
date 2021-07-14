import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras


#PROCESSING HELPERS
IMG_SIZE = 64

def get_csv_path(filename, path=''):
  """Reads csv file specified filepath

  Keyword arguments:
  filename (str) -- Desired csv to read

  path (str, optional) -- Location of csv (default is None)
  """

  csv_path = os.path.join(path, filename)
  return pd.read_csv(csv_path)


def process_image(img_path):
  #Convert Image to numpy array
  img = cv2.imread(img_path)

  #Resize the image
  img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

  #Normalize image
  img = (img - 127.5) / 127.5
  # img = img / 255.

  return img.astype(np.float32)


def smooth_labels(y, label_type):
  """Applies label smoothing to discriminator elements

  Prevents the discriminator from being either over confident or
  underconfident in its predictions by converting labels from a binary
  to a range of valid numbers per label type

  Keywork arguments:
  y (arraylike) -- Labels to smooth
  label_type (str) -- Either 'positive' or 'negative' depedning on which type 
    of labelthe function is smoothing
  """

  if label_type == 'positive':
    return y - 0.3 + (np.random.random(y.shape) * 0.5)
  elif label_type == 'negative':
    return y + np.random.random(y.shape) * 0.3
  else:
    raise ValueError('Expected "positive" or "negative" for label_type. Recieved:', label_type)


def noisy_labels(y, p_flip):
  """Applies noise to the discriminator labels

  Adds a small amount of error to the labels, which makes the
  learning process easier for the generator

  Keyword arguments:
  y (arraylike) -- Labels to add noise to
  p_flip (float) -- probablity that a label will flip from positive
    to negative
  """


  length = int(y.shape[0])

  num_labels = int(p_flip * length)

  i_flip = np.random.choice([i for i in range(length)], size=num_labels)

  op_list = []
  for i in range(length):
    if i in i_flip:
      op_list.append(tf.subtract(1, y[i]))
    else:
      op_list.append(y[i])

  outputs = tf.stack(op_list)
  return outputs


#PLOTTING HELPERS

def show_image(image):
  """Displays image using matplotlib"""

  #Readjusted pixel values (convert from [-1, 1] to [0, 1]) 
  image_adjusted = (image * 127.5 + 127.5) / 255.
  plt.imshow(image_adjusted, cmap='binary')
  plt.axis('off')
  plt.show()


def plot_multiple_images(images, epoch, n_cols=None):
  """Displays multiple images in grid format"""

  n_cols = n_cols or len(images)
  n_rows = (len(images) - 1) // n_cols + 1
  if images.shape[-1] == 1:
      images = np.squeeze(images, axis=-1)
  plt.figure(figsize=(n_cols, n_rows))
  for index, image in enumerate(images):

      image_adjusted = (image * 127.5 + 127.5) / 255.

      plt.subplot(n_rows, n_cols, index + 1)
      plt.imshow(image_adjusted, cmap='binary')
      plt.axis("off")
    
  print("[UPDATE] Saving image grid")
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  print("[UPDATE] Grid saved\n")


def plot_losses(G_loss, D_loss, G_loss_total, D_loss_total, G_mean, D_mean, epoch):
  """Displays loss graphs
  
  Graphs are displayed for both epoch loss and lifetime loss

  Keyword arguments:
  G_loss (arraylike) -- Epoch generator loss
  D_loss (arraylike) -- Epoch discriminator loss
  G_loss_total (arraylike) -- Lifetime generator loss
  D_loss_total (arraylike) -- Lifetime discriminator loss
  G_mean (arraylike, deprecated) -- Lifetime generator loss mean
  D_mean (arraylike, deprecated) -- Lifetime discriminator loss mean
  epoch (int) -- Current epoch
  """

  plt.figure(figsize=(10, 5))
  plt.title("Generator and Discriminator Loss - EPOCH {}".format(epoch + 1))
  plt.plot(G_loss, label="Generator")
  plt.plot(D_loss, label="Discriminator")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.legend()
  ymax = plt.ylim()[1]
  plt.draw()


  plt.figure(figsize=(10, 5))
  plt.plot(np.arange(len(G_loss_total)), G_loss_total, label='G')
  plt.plot(np.arange(len(D_loss_total)), D_loss_total, label='D')
  plt.legend()
  plt.title("All Time Loss")
  plt.draw()


#TRAINING HELPERS

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output, apply_smoothing=True, apply_noise=True):
  """Calculates discriminator loss using cross entropy

  Keyword arguments:
  real_output (arraylike) -- Discriminator output from actual images
  fake_output (arraylike) -- Discriminator output from generated images
  apply_smoothing (boolean, optional) -- Applies label smoothing (Default is True)
  apply_noise (boolean, optional) -- Applies noise to labels (Default is True)
  """

  real_output_mod = real_output
  fake_output_mod = fake_output

  if apply_noise:
    real_output_mod = noisy_labels(tf.ones_like(real_output_mod), 0.1)
    fake_output_mod = noisy_labels(tf.zeros_like(fake_output_mod), 0.1)
  
  if apply_smoothing:
    real_output_mod = smooth_labels(real_output_mod, 'positive')
    fake_output_mod = smooth_labels(fake_output_mod, 'negative')

  real_loss = cross_entropy(tf.ones_like(real_output_mod), real_output) #labels, outputs
  fake_loss = cross_entropy(tf.zeros_like(fake_output_mod), fake_output)

  return real_loss + fake_loss


def generator_loss(fake_output, apply_smoothing=True):
  """Calculated generator loss using cross entropy

  Keyword arguments:
  fake_output (arraylike) -- Discriminator output from generated images
  apply_smoothing (boolean, optional) -- Applies label smoothing (Default is True)
  """
  fake_output_mod = fake_output

  if apply_smoothing:
    fake_output_mod = smooth_labels(fake_output_mod, 'negative')

  return cross_entropy(tf.ones_like(fake_output_mod), fake_output)