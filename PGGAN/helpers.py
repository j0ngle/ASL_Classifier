import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

###########################
#   Image Proprocessing   #
###########################

#Funct: Process_image
#Descr: Process single image from a given directory to a desired square size
#Param: str path: os directory to images, int d_size: desired side length in pixels
def process_image(path, d_size):
    #Convert image to numpy array
    img = cv2.imread(path)

    #Resize
    img = cv2.resize(img, dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC)

    #Normalize image
    img = (img - 127.5) / 127.5

    return img.astype(np.float32)


#Funct: process_batch
#Descr: Process all images from a given directory to a desired square size
#Param: str path: os directory to images, int d_size: desired side length in pixels
def process_batch(path, d_size):
    processed = []
    i = 0

    print("[PREPROCESSING] Processing all images in {}...".format(path))
    print("[PREPROCESSING] Scaling images to size {}x{}".format(d_size, d_size))

    for filename in os.listdir(path):
        if filename.endswith('jpg'):
            if i == 10000:
                break

            if (i % 500 == 0):
                print("[PREPROCESSING] Processed {} images".format(i))

            img = None
            img = process_image(path + filename, d_size)

            processed.append(img)
            i += 1

    return np.asarray(processed)

#Funct: prepare_dataset
#Descr: Convert images ina  given directory to a tensorflow dataset
#Param: str path: os directory to images, int d_size: desired side length in pixels
def prepare_dataset(path, img_size, batch_size, sample_size):
    images = process_batch(path, img_size)
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(sample_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset.prefetch(1)

    return dataset


################
#   TRAINING   #
################

def smooth_labels(y, label_type):
    if label_type == 'positive':
        return y - 0.3 + (np.random.random(y.shape) * 0.5)
    elif label_type == 'negative':
        return y + np.random.random(y.shape) * 0.3
    else:
        raise ValueError('Expected "positive" or "negative" for label_type. Recieved:', label_type)

def noisy_labels(y, p_flip):
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

###############
#   METRICS   #
###############

def show_image(image, epoch):
    print("Showing image")
    #Readjusted pixel values (convert from [-1, 1] to [0, 1]) 
    image_adjusted = (image * 127.5 + 127.5) / 255.
    # plt.imshow(image_adjusted, cmap='binary')
    plt.axis('off')
    # plt.show()

    print("Saving single image")
    plt.savefig("image_at_epoch_{:04}.png".format(epoch))
    print("Image saved\n")
 
def plot_multiple_images(images, epoch, path, n_cols=None):
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

    print("\nSaving images grid")
    #TODO: Save to folder bc right now it isn't working for some reason
    filename = "PGGAN/" + path + "/grid_at_epoch_{:04}.png".format(epoch)
    # dir = os.path.join(filename)
    plt.savefig(filename)
    print("Grid saved\n")

    plt.close()

def print_statistics(list, title):
    print(title+" loss mean: ", np.mean(list), 
    "Std: ", np.std(list))

def plot_metrics(title, x_label, y_label, epoch, list1, list1_label, list2=None, list2_label=None):
    # title = title + "_at_epoch_{:04}".format(epoch)

    plt.figure(figsize=(10, 3)) 
    plt.title(title)
    plt.plot(list1, label=list1_label)
    plt.plot(list2, label=list2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    # plt.show()

    #TODO: Save to folder bc right now it isn't working for some reason
    filename = "PGGAN/metrics/" + title + "_epoch_{:04}.png".format(epoch)
    # dir = os.path.join("metrics/"+filename)
    
    plt.savefig(filename)
    

def plot_losses(G_loss, D_loss, G_loss_total, D_loss_total, G_mean, D_mean, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss - EPOCH {}".format(epoch + 1))
    plt.plot(G_loss, label="Generator")
    plt.plot(D_loss, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    ymax = plt.ylim()[1]
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(G_loss_total)), G_loss_total, label='G')
    plt.plot(np.arange(len(D_loss_total)), D_loss_total, label='D')
    plt.legend()
    plt.title("All Time Loss")
    # plt.show()