import tensorflow as tf
import os
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

GOOD_DIRECTORY = "good_imgs/"
BAD_DIRECTORY  = "bad_format_images/"
GOOD = 1
BAD = 0

def augment(image, label):
  image = tf.expand_dims(image, axis=-1)
  image = tf.image.random_crop(image, size=(224, 224, 1))
  image = tf.image.random_contrast(image, 0.2, 0.5)
  image = tf.image.random_flip_left_right(image)
  image = tf.image.random_flip_up_down(image)
  image = tf.image.random_brightness(image, max_delta=0.5)  

files = os.listdir(GOOD_DIRECTORY)

i = 1
tensor_list = []
label_list = []

#####LOADING THE GOOD DATA#####################################################################
for file in files:
    with open(GOOD_DIRECTORY + file, 'rb') as f:
        #print(i) # To check progress - approx 24000 imgs
        raw_bytes = f.read()
        match = re.search(r"_(?P<width>\d+)x(?P<height>\d+)_(?P<format>\w+)\.", file)
        if match:
            width = match.group("width")
            height = match.group("height")
            file_format = match.group("format")
        else:
            match1 = re.search(r"_(?P<width>\d+)x(?P<height>\d+)\.",file)
            width = match1.group("width")
            height = match1.group("height")
            file_format = "rgb24"
        i = i + 1

        width = int(width)
        height = int(height)

        print(f"width: {width}, height: {height}, format: {file_format}")

        match file_format:
            case "rgb24":
                mul = 3
            case "rgb332":
                mul = 1
            case "rgb565":
                mul = 2
            case "rgba32":
                mul = 4
            case "abgr444":
                mul = 2
            case "abgr555":
                mul = 2
            case "uyvy":
                mul = 4 
            case "gray":
                mul = 1
    ########## WORK ON SINGLE TENSOR ###############################################
        tensor = tf.io.decode_raw(raw_bytes, tf.uint8)
        tensor = tf.reshape(tensor, (width*mul,height))
        tensor = tf.expand_dims(tensor, axis=-1)
            
        ## HERE ADD CROP OR RESIZE TO MAKE TENSORS MATCH
        tensor = tf.image.resize(tensor, [256,256])


    ################################################################################

        ## Append label
        tensor_list.append(tensor)
        label_list.append(GOOD)
        

#####LOADING THE BAD DATA#####################################################################
j = 0
files = os.listdir(BAD_DIRECTORY)
for file in files:
    j = j +1
    image_string = open(BAD_DIRECTORY + file, 'rb').read()
########## WORK ON SINGLE TENSOR ###############################################

    bad_tensor = tf.io.decode_image(image_string, channels=1, dtype=tf.uint8)
    
    ## HERE ADD CROP OR RESIZE TO MAKE TENSORS MATCH
    bad_tensor = tf.image.resize(bad_tensor, [256,256])

################################################################################

    ## Append label
    tensor_list.append(bad_tensor)
    label_list.append(BAD)


        


########CONSTS#####################################################
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((tensor_list,label_list))
#dataset = dataset.batch(BATCH_SIZE)


#example 
#for img, label in dataset:
    
 