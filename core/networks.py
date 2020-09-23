import tensorflow as tf

def init_weight(shape):
	w = tf.truncated_normal(shape= shape, mean=0, stddev =0.1)
	return tf.Variable(w)

def init_bias(shape):
	b = tf.zeros(shape)
	return tf.Variable(b)

class Lenet2D(object):
	def __init__(self,
		num_classes,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0):
		"""
		Implements Lenet in 2D
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

	def ConvPool2d_block(self, input_tensor, filterShape, is_training=True, activation_fn="relu"):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = [1,1,1,1], padding ='VALID') + conv_B
		
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		if activation_fn == "relu":
			conv = tf.nn.relu(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		pool = tf.nn.max_pool2d(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
		return pool

	def GetNetwork(self, input_image):
		with tf.variable_scope('lenet2d'):
			input_channels = int(input_image.get_shape()[-1])
			conv1Filter_shape = [5,5,input_channels,20]
			pool1 = self.ConvPool2d_block(input_image, conv1Filter_shape, self.is_training, self.activation_fn)
			pool1_channels = int(pool1.get_shape()[-1])
			conv1Filter_shape = [5,5,pool1_channels,50]
			pool2 = self.ConvPool2d_block(pool1, conv1Filter_shape, self.is_training , self.activation_fn)

			flatten = tf.reshape(pool2, [-1, pool2.get_shape()[1]*pool2.get_shape()[2]*pool2.get_shape()[3]])
			dense0 = tf.layers.dense(inputs=flatten,units=768, activation=self.activation_fn)
			dense1 = tf.layers.dense(inputs=dense0,units=500, activation=self.activation_fn)
			logits = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)
			# logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

		return logits

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
		with tf.variable_scope('lenet3d'):
			input_channels = int(input_image.get_shape()[-1])
			conv1Filter_shape = [5,5,5,input_channels,20]
			pool1 = self.ConvPool3d_block(input_image, conv1Filter_shape, self.is_training, self.activation_fn)
			pool1_channels = int(pool1.get_shape()[-1])
			conv1Filter_shape = [5,5,5,pool1_channels,50]
			pool2 = self.ConvPool3d_block(pool1, conv1Filter_shape, self.is_training , self.activation_fn)

			flatten = tf.reshape(pool2, [-1, pool2.get_shape()[1]*pool2.get_shape()[2]*pool2.get_shape()[3]*pool2.get_shape()[4]])
			dense0 = tf.layers.dense(inputs=flatten,units=768, activation=self.activation_fn)
			dense1 = tf.layers.dense(inputs=dense0,units=500, activation=self.activation_fn)
			logits = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)
			# logits = tf.layers.batch_normalization(logits, momentum=0.99, epsilon=0.001,center=True, scale=True,training=self.is_training)

		return logits

class Alexnet2D(object):
	def __init__(self,
		num_classes,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0):
		"""
		Implements Lenet in 2D
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

	def ConvPool2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		pool = tf.nn.max_pool2d(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
		return pool

	def Conv2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		return conv

	def GetNetwork(self, input_image):
		print("Input image: ", input_image.get_shape() )

		with tf.variable_scope('alexnet2d'):
			input_channels = int(input_image.get_shape()[-1])
			conv1Filter_shape = [5,5,input_channels,96]
			layer1_stride = [1,1,1,1,1]
			paddings = tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]])
			input_image = tf.pad(input_image, paddings, "CONSTANT") 
			pool1 = self.ConvPool2d_block(input_image, conv1Filter_shape, strides = layer1_stride,is_training = self.is_training)
			pool1_channels = int(pool1.get_shape()[-1])

			conv2Filter_shape = [5,5,pool1_channels,256]
			pool1 = tf.pad(pool1, paddings, "CONSTANT") 
			pool2 = self.ConvPool2d_block(pool1, conv2Filter_shape, is_training = self.is_training)
			pool2_channels = int(pool2.get_shape()[-1])

			paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
			pool2 = tf.pad(pool2, paddings, "CONSTANT") 
			conv3Filter_shape = [3,3,pool2_channels,384]
			pool3 = self.Conv2d_block(pool2, conv3Filter_shape, is_training = self.is_training)
			pool3_channels = int(pool3.get_shape()[-1])

			pool3 = tf.pad(pool3, paddings, "CONSTANT") 
			conv4Filter_shape = [3,3, pool3_channels,384]
			pool4 = self.Conv2d_block(pool3, conv4Filter_shape, is_training = self.is_training)
			pool4_channels = int(pool4.get_shape()[-1])

			pool4 = tf.pad(pool4, paddings, "CONSTANT") 
			conv5Filter_shape = [3,3, pool4_channels, 256]
			pool5 = self.ConvPool2d_block(pool4, conv5Filter_shape, is_training = self.is_training)

			flatten = tf.reshape(pool5, [-1, pool5.get_shape()[1]*pool5.get_shape()[2]*pool5.get_shape()[3]*pool5.get_shape()[4]])
			dense0 = tf.layers.dense(inputs=flatten,units=768, activation=self.activation_fn)
			dense1 = tf.layers.dense(inputs=dense0,units=500, activation=self.activation_fn)
			logits = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)

		return logits

class Alexnet3D(object):
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

	def ConvPool3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		pool = tf.nn.max_pool3d(conv, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding = 'VALID')
		return pool

	def Conv3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		return conv

	def GetNetwork(self, input_image):
		with tf.variable_scope('alexnet3d'):
			input_channels = int(input_image.get_shape()[-1])
			conv1Filter_shape = [5,5,5,input_channels,96]
			layer1_stride = [1,1,1,1,1]
			paddings = tf.constant([[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]])
			input_image = tf.pad(input_image, paddings, "CONSTANT") 
			pool1 = self.ConvPool3d_block(input_image, conv1Filter_shape, strides = layer1_stride,is_training = self.is_training)
			pool1_channels = int(pool1.get_shape()[-1])

			conv2Filter_shape = [5,5,5,pool1_channels,256]
			pool1 = tf.pad(pool1, paddings, "CONSTANT") 
			pool2 = self.ConvPool3d_block(pool1, conv2Filter_shape, is_training = self.is_training)
			pool2_channels = int(pool2.get_shape()[-1])

			paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
			pool2 = tf.pad(pool2, paddings, "CONSTANT") 
			conv3Filter_shape = [3,3,3,pool2_channels,384]
			pool3 = self.Conv3d_block(pool2, conv3Filter_shape, is_training = self.is_training)
			pool3_channels = int(pool3.get_shape()[-1])

			pool3 = tf.pad(pool3, paddings, "CONSTANT") 
			conv4Filter_shape = [3,3,3, pool3_channels,384]
			pool4 = self.Conv3d_block(pool3, conv4Filter_shape, is_training = self.is_training)
			pool4_channels = int(pool4.get_shape()[-1])

			pool4 = tf.pad(pool4, paddings, "CONSTANT") 
			conv5Filter_shape = [3,3,3, pool4_channels, 256]
			pool5 = self.ConvPool3d_block(pool4, conv5Filter_shape, is_training = self.is_training)

			flatten = tf.reshape(pool5, [-1, pool5.get_shape()[1]*pool5.get_shape()[2]*pool5.get_shape()[3]*pool5.get_shape()[4]])
			dense0 = tf.layers.dense(inputs=flatten,units=768, activation=self.activation_fn)
			dense1 = tf.layers.dense(inputs=dense0,units=500, activation=self.activation_fn)
			logits = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)

		return logits

class Resnet2D(object):
	def __init__(self,
		num_classes,
		num_channels=64,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0,
		init_conv_shape=7,
		init_pool=True,
		module_config=[3,3,5,2]):
		"""
		Implements Resnet3D in 3D
		:param num_classes: Number of output classes.
		:param num_channels: Number of feature channels.
		:param is_training: Set network in training mode.
		:param activation_fn: The activation function.
		:param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
		:param init_conv_shape: First layer convolution kernel size
		:param init_pool: Choose whether to use pooling after first layer convolution
		:param module_config: Number of residual blocks that separates by subsampling convolution layers
		"""
		self.num_classes = num_classes
		self.is_training = is_training
		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		else:
			print("Invalid activation function")
			exit()
		self.keep_prob = keep_prob
		self.init_conv_shape = init_conv_shape
		self.init_pool = init_pool
		self.module_config = module_config
		self.num_channels = num_channels

	def ConvPool2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		pool = tf.nn.max_pool2d(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
		return pool

	def ConvActivate2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		return conv

	def Conv2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		return conv

	def residual_block(self, input_tensor, channels, is_training=True):
		paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
		input_tensor_padded = tf.pad(input_tensor, paddings, "CONSTANT")
		input_channels = int(input_tensor.get_shape()[-1])
		conv1Filter_shape = [3,3,input_channels,channels]
		conv1 = self.ConvActivate2d_block(input_tensor_padded, conv1Filter_shape, is_training = self.is_training)
		conv1 = tf.pad(conv1, paddings, "CONSTANT")
		conv2Filter_shape = [3,3,channels,channels]
		conv2 = self.Conv2d_block(conv1, conv2Filter_shape, is_training = self.is_training)
		input_tensor_norm = tf.layers.batch_normalization(input_tensor, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		output = tf.add(conv2, input_tensor_norm)
		output = tf.layers.batch_normalization(output, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		output = self.activation_fn(output)
		output = tf.nn.dropout(output, self.keep_prob)
		return output

	def residual_shortcut_block(self, input_tensor, channels, is_training=True):
		paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
		input_tensor_padded = tf.pad(input_tensor, paddings, "CONSTANT")
		input_channels = int(input_tensor.get_shape()[-1])
		conv1Filter_shape = [3,3,input_channels,channels]
		conv1 = self.ConvActivate2d_block(input_tensor_padded, conv1Filter_shape, strides = [1,2,2,1], is_training = self.is_training)
		conv1 = tf.pad(conv1, paddings, "CONSTANT")
		conv2Filter_shape = [3,3,channels,channels]
		conv2 = self.Conv2d_block(conv1, conv2Filter_shape, is_training = self.is_training)

		shortcut_filterShape = [1,1,input_channels,channels]
		shortcut_conv_W = init_weight(shortcut_filterShape)
		shortcut_conv_B = init_bias(shortcut_filterShape[3])
		shortcut = tf.nn.conv2d(input_tensor, shortcut_conv_W, strides = [1,2,2,1], padding ='VALID') + shortcut_conv_B
		shortcut_norm = tf.layers.batch_normalization(shortcut, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)

		output = tf.add(conv2, shortcut_norm)
		output = tf.layers.batch_normalization(output, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		output = self.activation_fn(output)
		output = tf.nn.dropout(output, self.keep_prob)

		return output

	def GetNetwork(self, input_image):
		with tf.variable_scope('Resnet2D/init_conv'):
			input_channels = int(input_image.get_shape()[-1])
			init_conv_filter_shape = [self.init_conv_shape,self.init_conv_shape,input_channels,self.num_channels]
			if self.init_pool:
				x = self.ConvPool2d_block(input_image, init_conv_filter_shape, strides=[1,1,1,1],is_training=self.is_training)
			else:
				x = self.ConvActivate2d_block(input_image, init_conv_filter_shape, strides = [1,1,1,1], is_training=self.is_training)

		for module in range(len(self.module_config)):
			with tf.variable_scope('Resnet2D/module' + str(module+1)):
				if module > 0:
					x = self.residual_shortcut_block(x, self.num_channels*(1+module), self.is_training)
				for block in range(self.module_config[module]):
					with tf.variable_scope('block' + str(block+1)):
						x = self.residual_block(x,self.num_channels*(1+module), self.is_training)

		avgPool = tf.layers.average_pooling2d(x, pool_size=[2,2], strides=[2,2], padding = 'valid')

		flatten = tf.reshape(avgPool, [-1, avgPool.get_shape()[1]*avgPool.get_shape()[2]*avgPool.get_shape()[3]])
		dense = tf.layers.dense(inputs=flatten,units=1000, activation=self.activation_fn)
		logits = tf.layers.dense(inputs=dense,units=self.num_classes, activation=None)

		return logits

class GoogLeNet2D(object):
	def __init__(self,
		num_classes,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0,
		version=1):
		"""
		Implements GoogLeNet in 2D
		:param num_classes: Number of output classes.
		:param is_training: Set network in training mode.
		:param activation_fn: The activation function.
		:param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
		:param version: GoogLeNet version, from 1 to 4
		"""
		self.num_classes = num_classes
		self.is_training = is_training
		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		else:
			print("Invalid activation function")
			exit()
		self.keep_prob = keep_prob
		self.version = version

		print("GoogLeNet version {}".format(str(version)))

	def ConvActivate2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], padding='VALID', is_training=True, name=""):
		input_channels = int(input_tensor.get_shape()[-1])
		
		with tf.variable_scope(name):
			conv_W = init_weight(filterShape)
			conv_B = init_bias(filterShape[3])
			conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding = padding) + conv_B
			conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
			conv = self.activation_fn(conv)
			conv = tf.nn.dropout(conv, self.keep_prob)
		return conv

	def inception_module(self, 
			input_tensor,
			channels_1x1,
			channels_3x3_reduce,
			channels_3x3,
			channels_5x5_reduce,
			channels_5x5,
			channels_pool,
			version=1,
			name=""
			):
		input_channels = int(input_tensor.get_shape()[-1])

		with tf.variable_scope(name):
			conv_1x1 = self.ConvActivate2d_block(input_tensor, [1,1,input_channels,channels_1x1],padding='SAME',name="1x1")

			conv_3x3_reduce = self.ConvActivate2d_block(input_tensor, [1,1,input_channels,channels_3x3_reduce],padding='SAME',name="3x3_reduce")
			conv_3x3 = self.ConvActivate2d_block(conv_3x3_reduce, [3,3,channels_3x3_reduce,channels_3x3],padding='SAME',name="3x3")

			if version > 1:
				pass
			else:
				conv_5x5_reduce = self.ConvActivate2d_block(input_tensor, [1,1,input_channels,channels_5x5_reduce],padding='SAME',name="5x5_reduce")
				conv_5x5 = self.ConvActivate2d_block(conv_5x5_reduce, [3,3,channels_5x5_reduce,channels_5x5],padding='SAME',name="5x5")

			pool = tf.nn.max_pool2d(input_tensor, ksize=[1,3,3,1], strides=[1,1,1,1], padding = 'SAME',name='pool')
			pool = self.ConvActivate2d_block(pool, [1,1,input_channels,channels_pool],padding='SAME',name="pool_conv")

			output = tf.concat([conv_1x1,conv_3x3,conv_5x5,pool],axis=3,name="output")

		return output

	def auxillary_output(self,input_image, keep_prob=1.0,units=1024,name=""):
		with tf.variable_scope(name):
			x = tf.nn.avg_pool(input_image,[1,5,5,1],strides=[1,3,3,1],padding='SAME')
			x = self.ConvActivate2d_block(x,[1,1,int(input_image.get_shape()[-1]),128],padding='SAME')
			x = tf.reshape(x, [-1, x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3]])
			x = tf.layers.dense(inputs=x,units=units, activation=self.activation_fn)
			x = tf.nn.dropout(x, keep_prob)
			x = tf.layers.dense(inputs=x,units=self.num_classes, activation=None)
		return x

	def GetNetwork(self, input_image):
		keep_prob = self.keep_prob if self.is_training else 1.0
		input_channels = int(input_image.get_shape()[-1])

		aux = []

		with tf.variable_scope('GoogLeNet2D'):
			with tf.variable_scope('input'):
				x = self.ConvActivate2d_block(input_image,[7,7,input_channels,64],strides=[1,2,2,1],padding='SAME',is_training=self.is_training, name="conv_1")
				x = tf.nn.max_pool2d(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'SAME',name='pool1')
				x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75,name='pool1_lrn')

				x = self.ConvActivate2d_block(x,[1,1,64,64],padding='SAME',is_training=self.is_training, name="conv_2a")
				x = self.ConvActivate2d_block(x,[3,3,64,192],padding='SAME',is_training=self.is_training, name="conv_2b")
				x = tf.nn.max_pool2d(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'SAME',name='pool2')
				x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75,name='pool2_lrn')

			# block 3
			with tf.variable_scope('block3'):
				x = self.inception_module(x,64,96,128,16,32,32,name="inception_3a",version=self.version)
				x = self.inception_module(x,128,128,192,32,96,64,name="inception_3b",version=self.version)
				x = tf.nn.max_pool2d(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'SAME',name='pool3')

			# block 4
			with tf.variable_scope('block4'):
				x = self.inception_module(x,192,96,208,16,48,64,name="inception_4a",version=self.version)
				aux_1 = self.auxillary_output(x,keep_prob=keep_prob,name="auxillary_output_1")
				aux.append(aux_1)
				x = self.inception_module(x,160,112,224,24,64,64,name="inception_4b",version=self.version)
				x = self.inception_module(x,128,128,256,24,64,64,name="inception_4c",version=self.version)
				x = self.inception_module(x,112,144,288,32,64,64,name="inception_4d",version=self.version)
				aux_2 = self.auxillary_output(x,keep_prob=keep_prob,name="auxillary_output_2")
				aux.append(aux_2)
				x = self.inception_module(x,256,160,320,32,128,128,name="inception_4e",version=self.version)
				x = tf.nn.max_pool2d(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'SAME',name='pool3')

			# block 5
			with tf.variable_scope('block5'):
				x = self.inception_module(x,256,160,320,32,128,128,name="inception_5a",version=self.version)
				x = self.inception_module(x,128,128,256,24,64,64,name="inception_5b",version=self.version)

			# output
			with tf.variable_scope('output'):
				aux_3 = tf.nn.avg_pool(x,[1,7,7,1],strides=[1,1,1,1],padding='SAME')
				aux_3 = tf.reshape(aux_3, [-1, aux_3.get_shape()[1]*aux_3.get_shape()[2]*aux_3.get_shape()[3]])
				aux_3 = tf.layers.dense(inputs=aux_3,units=1024, activation=self.activation_fn)
				aux_3 = tf.nn.dropout(aux_3, keep_prob)
				aux_3 = tf.layers.dense(inputs=aux_3,units=self.num_classes, activation=None)
				aux.append(aux_3)

				# linear combination of auxiliary outputs
				output = tf.concat(aux,axis=-1)
				logits = tf.layers.dense(inputs=output,units=self.num_classes, activation=None)

		return logits

class Vgg2D(object):
	def __init__(self,
		num_classes,
		num_channels=64,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0,
		module_config=[2,2,3,3,3],
		batch_norm_momentum=0.99,
		fc_channels=[4096,4096]):
		"""
		Implements Vgg in 2D
		:param num_classes: Number of output classes.
		:param num_channels: Number of feature channels.
		:param is_training: Set network in training mode.
		:param activation_fn: The activation function.
		:param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
		:param module_config: Number of residual blocks that separates by subsampling convolution layers
		:param batch_norm_momentum: Momentum for batch normalization layer
		:param fc_channels: Channel number for fully connected layers except last one
		"""
		self.num_classes = num_classes
		self.is_training = is_training
		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		else:
			print("Invalid activation function")
			exit()
		self.keep_prob = keep_prob
		self.module_config = module_config
		self.num_channels = num_channels
		self.train_phase = tf.placeholder(tf.bool, name="train_phase_placeholder")
		self.batch_norm_momentum = batch_norm_momentum
		self.fc_channels = fc_channels

	def ConvActivate2d_block(self, input_tensor, filterShape, strides = [1,1,1,1], is_training=True):
		keep_prob = self.keep_prob if self.is_training else 1.0
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[3])
		conv = tf.nn.conv2d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=self.batch_norm_momentum, epsilon=0.001,center=True, scale=True,training=self.train_phase)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, keep_prob)
		return conv

	def GetNetwork(self, input_image):
		keep_prob = self.keep_prob if self.is_training else 1.0

		x = input_image
		for module in range(len(self.module_config)):
			with tf.variable_scope('Vgg/module' + str(module+1)):
				for layer in range(self.module_config[module]):
					with tf.variable_scope('conv_' + str(layer+1)):
						paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])
						x = tf.pad(x, paddings, "CONSTANT")
						input_channels = int(x.get_shape()[-1])
						convFilter_shape = [3,3,input_channels, self.num_channels*(1+module)]
						x = self.ConvActivate2d_block(x, convFilter_shape, is_training = self.is_training)

				with tf.variable_scope('max_pool'):
					x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

		with tf.variable_scope('Vgg/fully_connected'):
			x = tf.reshape(x, [-1, x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3]])

			for channel in self.fc_channels:
				x = tf.layers.dense(inputs=x,units=channel, activation=self.activation_fn)
				x = tf.layers.batch_normalization(x, momentum=self.batch_norm_momentum, epsilon=0.001,center=True, scale=True,training=self.train_phase)
		
			logits = tf.layers.dense(inputs=x,units=self.num_classes, activation=None)

		return logits

class Resnet3D(object):
	def __init__(self,
		num_classes,
		num_channels=64,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0,
		init_conv_shape=7,
		init_pool=True,
		module_config=[3,3,5,2]):
		"""
		Implements Resnet3D in 3D
		:param num_classes: Number of output classes.
		:param num_channels: Number of feature channels.
		:param is_training: Set network in training mode.
		:param activation_fn: The activation function.
		:param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
		:param init_conv_shape: First layer convolution kernel size
		:param init_pool: Choose whether to use pooling after first layer convolution
		:param module_config: Number of residual blocks that separates by subsampling convolution layers
		"""
		self.num_classes = num_classes
		self.is_training = is_training
		if (activation_fn == "relu"):
			self.activation_fn = tf.nn.relu
		else:
			print("Invalid activation function")
			exit()
		self.keep_prob = keep_prob
		self.init_conv_shape = init_conv_shape
		self.init_pool = init_pool
		self.module_config = module_config
		self.num_channels = num_channels

	def ConvPool3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		pool = tf.nn.max_pool3d(conv, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding = 'VALID')
		return pool

	def ConvActivate3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		conv = self.activation_fn(conv)
		conv = tf.nn.dropout(conv, self.keep_prob)
		return conv

	def Conv3d_block(self, input_tensor, filterShape, strides = [1,1,1,1,1], is_training=True):
		input_channels = int(input_tensor.get_shape()[-1])

		conv_W = init_weight(filterShape)
		conv_B = init_bias(filterShape[4])
		conv = tf.nn.conv3d(input_tensor, conv_W, strides = strides, padding ='VALID') + conv_B
		conv = tf.layers.batch_normalization(conv, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		return conv

	def residual_block(self, input_tensor, channels, is_training=True):
		paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
		input_tensor_padded = tf.pad(input_tensor, paddings, "CONSTANT")
		input_channels = int(input_tensor.get_shape()[-1])
		conv1Filter_shape = [3,3,3,input_channels,channels]
		conv1 = self.ConvActivate3d_block(input_tensor_padded, conv1Filter_shape, is_training = self.is_training)
		conv1 = tf.pad(conv1, paddings, "CONSTANT")
		conv2Filter_shape = [3,3,3,channels,channels]
		conv2 = self.Conv3d_block(conv1, conv2Filter_shape, is_training = self.is_training)
		input_tensor_norm = tf.layers.batch_normalization(input_tensor, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		output = tf.add(conv2, input_tensor_norm)
		output = tf.layers.batch_normalization(output, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		output = self.activation_fn(output)
		output = tf.nn.dropout(output, self.keep_prob)
		return output

	def residual_shortcut_block(self, input_tensor, channels, is_training=True):
		paddings = tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
		input_tensor_padded = tf.pad(input_tensor, paddings, "CONSTANT")
		input_channels = int(input_tensor.get_shape()[-1])
		conv1Filter_shape = [3,3,3,input_channels,channels]
		conv1 = self.ConvActivate3d_block(input_tensor_padded, conv1Filter_shape, strides = [1,2,2,2,1], is_training = self.is_training)
		conv1 = tf.pad(conv1, paddings, "CONSTANT")
		conv2Filter_shape = [3,3,3,channels,channels]
		conv2 = self.Conv3d_block(conv1, conv2Filter_shape, is_training = self.is_training)

		shortcut_filterShape = [1,1,1,input_channels,channels]
		shortcut_conv_W = init_weight(shortcut_filterShape)
		shortcut_conv_B = init_bias(shortcut_filterShape[4])
		shortcut = tf.nn.conv3d(input_tensor, shortcut_conv_W, strides = [1,2,2,2,1], padding ='VALID') + shortcut_conv_B
		shortcut_norm = tf.layers.batch_normalization(shortcut, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)

		output = tf.add(conv2, shortcut_norm)
		output = tf.layers.batch_normalization(output, momentum=0.99, epsilon=0.001,center=True, scale=True,training=is_training)
		output = self.activation_fn(output)
		output = tf.nn.dropout(output, self.keep_prob)

		return output

	def GetNetwork(self, input_image):
		with tf.variable_scope('Resnet3D/init_conv'):
			input_channels = int(input_image.get_shape()[-1])
			init_conv_filter_shape = [self.init_conv_shape,self.init_conv_shape,self.init_conv_shape,input_channels,self.num_channels]
			if self.init_pool:
				x = self.ConvPool3d_block(input_image, init_conv_filter_shape, strides=[1,1,1,1,1],is_training=self.is_training)
			else:
				x = self.ConvActivate3d_block(input_image, init_conv_filter_shape, strides = [1,1,1,1,1], is_training=self.is_training)

		for module in range(len(self.module_config)):
			with tf.variable_scope('Resnet3D/module' + str(module+1)):
				if module > 0:
					x = self.residual_shortcut_block(x, self.num_channels*(1+module), self.is_training)
				for block in range(self.module_config[module]):
					with tf.variable_scope('block' + str(block+1)):
						x = self.residual_block(x,self.num_channels*(1+module), self.is_training)

		avgPool = tf.layers.average_pooling3d(x, pool_size=[2,2,2], strides=[2,2,2], padding = 'valid')

		flatten = tf.reshape(avgPool, [-1, avgPool.get_shape()[1]*avgPool.get_shape()[2]*avgPool.get_shape()[3]*avgPool.get_shape()[4]])
		dense = tf.layers.dense(inputs=flatten,units=1000, activation=self.activation_fn)
		logits = tf.layers.dense(inputs=dense,units=self.num_classes, activation=None)

		return logits