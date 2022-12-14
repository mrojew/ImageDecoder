import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# dataset1_fromrgb.tf OR dataset1_grayscale.tf
dataset = tf.data.experimental.load("dataset1_grayscale.tf")

def augment(image, label):
  image = tf.expand_dims(image, axis=-1)
  image = tf.image.random_crop(image, size=(224, 224, 1))
  image = tf.image.random_contrast(image, 0.2, 0.5)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_brightness(image, max_delta=0.5)

  return image, label

augmented_dataset = dataset.map(augment)
shuffled_dataset = dataset.shuffle(buffer_size = 1024)


for img, label in shuffled_dataset:
    plt.imshow(img, cmap='gray')
    plt.title(label.numpy())
    plt.show()

batched_dataset = shuffled_dataset.batch(32)

#model = tf.keras.models.Sequential([
#])

#model.compile(
#)

#model.fit(batched_dataset)