import tensorflow as tf

def init_weight(shape):
	w = tf.truncated_normal(shape= shape, mean=0, stddev =0.1)
	return tf.Variable(w)

def init_bias(shape):
	b = tf.zeros(shape)
	return tf.Variable(b)

class Lenet3D(object):
	def __init__(self,
		num_classes,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0):
		"""
		Implements Lenet in 3D
		:param num_classes: Number of output classes.
		:param is_training: Set network in training mode
		:param activation_fn: The activation function.
		:param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
		"""
		self.num_classes = num_classes
		self.is_training = is_training
		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		else:
			print("Invalid activation function")
			exit()
		self.keep_prob = keep_prob

	def ConvPool3d_block(self, input_tensor, filterShape, is_training=True, activation_fn="relu"):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = [1,1,1,1,1], padding ='VALID') + conv_B
		
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		if activation_fn == "relu":
			conv = tf.nn.relu(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		pool = tf.nn.max_pool3d(conv, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding = 'VALID')
		return pool

	def GetNetwork(self, input_image):
		input_channels = int(input_image.get_shape()[-1])
		conv1Filter_shape = [5,5,5,input_channels,20]
		pool1 = self.ConvPool3d_block(input_image, conv1Filter_shape, self.is_training, self.activation_fn)
		pool1_channels = int(pool1.get_shape()[-1])
		conv1Filter_shape = [5,5,5,pool1_channels,50]
		pool2 = self.ConvPool3d_block(pool1, conv1Filter_shape, self.is_training , self.activation_fn)

		with tf.variable_scope('lenet3d/output_layer'):
			flatten = tf.reshape(pool2, [-1, pool2.get_shape()[1]*pool2.get_shape()[2]*pool2.get_shape()[3]*pool2.get_shape()[4]])
			dense0 = tf.layers.dense(inputs=flatten,units=768, activation=self.activation_fn)
			dense1 = tf.layers.dense(inputs=dense0,units=500, activation=self.activation_fn)
			logits = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)
			# logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

		return logits

