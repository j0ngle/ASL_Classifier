import tensorflow as tf
from tensorflow import keras

def minimax_loss_D(real_output, fake_output, apply_smoothing=True, apply_noise=True):
  '''
  Discriminator wants to MAXIMIZE this function, so we should take the negative
  Ex[log(D(x))] + Ez[log(1-D(G(z)))], where:
      Ex = expected value over all real data instances (1s)
      D(x) = Discriminator's estimate of the probability that (real) x is real
      Ez   = expected value over all fake data instances (0s)
      G(z) = Generator's outputs
      D(G(z)) = Discriminator's estimate of the probability that (fake) z is 
  '''

  Ex = tf.ones_like(real_output)
  sig_real = tf.math.sigmoid(real_output)
  real_loss = -Ex * tf.math.log(sig_real)

  Ez = tf.zeros_like(fake_output)
  sig_fake = tf.math.sigmoid(fake_output)
  fake_loss = -Ez * tf.math.log(1 - sig_fake)
  
  return real_loss + fake_loss

def minimax_loss_G(fake_output, apply_smoothing=True):
  Ez = tf.zeros_like(fake_output)
  sig_fake = tf.math.sigmoid(fake_output)
  fake_loss = Ez * tf.math.log(1 - sig_fake)