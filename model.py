import tensorflow as tf

def create_variable(name, shape):
	initializer = tf.contrib.layers.xavier_initializer_conv2d()
	variable = tf.Variable(initializer(shape=shape), name=name)
	return variable

def create_bias_variable(name, shape):
	initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
	return tf.Variable(initializer(shape=shape), name)


class FragModel(object):
	def __init__(self,
				 input_channel,
				 klass,
				 batch_size,
				 kernel_size,
				 dilations,
				 channels):
		self.input_channel = input_channel
		self.klass = klass
		self.batch_size = batch_size
		self.dilations = dilations
		self.kernel_size = kernel_size
		channels.insert(0, self.input_channel)
		channels.append(self.klass)
		self.channels = channels
		self.variables = self._create_variables()

	def _create_variables(self):
		var = dict()

		var['filters'] = list()
		for i, dilation in enumerate(self.dilations):
			var['filters'].append(create_variable('filter',
												  [self.kernel_size[i],
												   self.kernel_size[i],
												   self.channels[i],
												   self.channels[i + 1]]))
		var['biases'] = list()
		for i, channel in enumerate(self.channels):
			if i == 0:
				continue
			var['biases'].append(create_bias_variable('bias', [channel]))
		return var

	def _preprocess(self, input_data, generate=False):
		if generate == True:
			image = input_data
			image = tf.cast(tf.expand_dims(tf.expand_dims(image, 2), 0), tf.float32)
			label = None
		else:
			image = input_data[0]
			image = tf.cast(tf.expand_dims(image, 0), tf.float32)
			label = input_data[1]
			label = tf.reshape(label, [-1])
			# value for the elements in label before preprocess can be:
			#	i (i >= 0): the i-th class (0-based)
			label = tf.cast(label, tf.int32)
		# tf.nn.conv2d(padding='SAME') always pads 0 to the input tensor,
		# thus make the value of the white pixels in the image 0
		image = 1.0 - image / 255.0
		return image, label

	def _create_network(self, input_data):
		current_layer = input_data
		for layer_idx, dilation in enumerate(self.dilations):
			conv = tf.nn.atrous_conv2d(value=current_layer,
									   filters=self.variables['filters'][layer_idx],
									   rate=self.dilations[layer_idx],
									   padding='VALID')
			with_bias = tf.nn.bias_add(conv, self.variables['biases'][layer_idx])
			if layer_idx == len(self.dilations) - 1:
				current_layer = with_bias
			else:
				current_layer = tf.nn.relu(with_bias)
		return current_layer

	def loss(self, input_data):
		image, label = self._preprocess(input_data)

		output = self._create_network(image)

		output = tf.reshape(output, [-1, self.klass])

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
															  labels=label)
		reduced_loss = tf.reduce_mean(loss)
		tf.scalar_summary('loss', reduced_loss)
		return reduced_loss

	def generate(self, image):
		image, _ = self._preprocess(input_data=image,
									generate=True)
		output = self._create_network(image)
		output_image = tf.argmax(input=output,
								 dimension=3)
		return output_image
