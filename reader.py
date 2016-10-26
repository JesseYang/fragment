import tensorflow as tf
from tensorflow.contrib import ffmpeg
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import os
import json

def preprocess_images(height, width_pad):
	for (dirpath, dirnames, filenames) in os.walk('training_set/images'):
		for filename in filenames:
			image = misc.imread('training_set/images/' + filename, mode='L')
			cur_height = image.shape[0]
			cur_width = image.shape[1]
			if height > cur_height:
				# should pad in the vertical direction
				top_pad = (height - cur_height) / 2
				bottom_pad = (height - cur_height) - top_pad
				pad_image = np.pad(array=image,
								   pad_width=((top_pad, bottom_pad), (width_pad, width_pad)),
								   mode='constant',
								   constant_values=(255, 255))
			else:
				# should resise to fit the vertical direction
				print height
				resized_image = misc.imresize(image, (height, image.shape[1]))
				print resized_image.shape
				pad_image = np.pad(array=resized_image,
								   pad_width=((0, 0), (width_pad, width_pad)),
								   mode='constant',
								   constant_values=(255, 255))
			misc.imsave('training_set/pad_images/' + filename, pad_image)

def image_label_list():
	image_name_list = [ ]
	label_name_list = [ ]
	for (dirpath, dirnames, filenames) in os.walk('training_set/pad_images'):
		label_name_list.extend(map(lambda x: 'training_set/labels/' + x.replace('.png', '.dat'), filenames))
		image_name_list.extend(map(lambda x: 'training_set/pad_images/' + x, filenames))
		break
	return image_name_list, label_name_list

def create_inputs(input_channel, dilations):
	receptive_field = 0
	for dilation in dilations:
		receptive_field = receptive_field + dilation
	width_pad = receptive_field
	receptive_field = receptive_field * 2 + 1
	height = receptive_field
	preprocess_images(height, width_pad)

	image_name_list, label_name_list = image_label_list()

	seed = np.random.randint(1000)

	image_name_queue = tf.train.string_input_producer(image_name_list, seed=seed)
	label_name_queue = tf.train.string_input_producer(label_name_list, seed=seed)

	image_reader = tf.WholeFileReader()
	_, image_content = image_reader.read(image_name_queue)
	image_tensor = tf.image.decode_png(image_content, channels=input_channel)

	label_reader = tf.WholeFileReader()
	_, label_content = label_reader.read(label_name_queue)
	label_tensor = tf.decode_raw(label_content, tf.uint8)

	return image_tensor, label_tensor
