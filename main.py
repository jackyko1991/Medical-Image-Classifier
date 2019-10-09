import argparse
import SimpleITK as sitk
import datetime
import tensorflow as tf
import numpy as np

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

def main():
	images = []

	# load image data
	reader = sitk.ImageFileReader()
	reader.SetFileName("./data/dataset/training/A01/image_brain_mni.nii.gz")
	images.append(reader.Execute())
	reader.SetFileName("./data/dataset/training/A03/image_brain_mni.nii.gz")
	images.append(reader.Execute())

	labels = []
	labels.append([1,0,0,0])
	labels.append([1,1,0,0])

	# normalize images
	sigma = 2.5
	for i in range(2):
		statisticsFilter = sitk.StatisticsImageFilter()
		statisticsFilter.Execute(images[i])

		intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
		intensityWindowingFilter.SetOutputMaximum(255)
		intensityWindowingFilter.SetOutputMinimum(0)
		intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+sigma*statisticsFilter.GetSigma());
		intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-sigma*statisticsFilter.GetSigma());

		images[i] = intensityWindowingFilter.Execute(images[i])

	images_np = []
	for i in range(2):
		images_np.append(sitk.GetArrayFromImage(images[i]))

	# placeholders
	input_placeholder = tf.placeholder(tf.float32, (None,images_np[0].shape[0],images_np[0].shape[1],images_np[0].shape[2],1))
	output_placeholder = tf.placeholder(tf.float32, (None, 4))

	# graph
	network = Lenet3D(
		num_classes=4,
		is_training=True,
		activation_fn="relu",
		keep_prob=1.0
		)
	logits_op = network.GetNetwork(input_placeholder)
	logits_op = tf.reshape(logits_op, [-1,4])

	loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_op,labels=output_placeholder))

	sigmoid_op = tf.sigmoid(logits_op)

	global_step = tf.train.get_or_create_global_step()
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
	train_op = optimizer.minimize(
		loss=loss_op,
		global_step=global_step
		)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(9999):
			for j in range(2):
				image = images_np[j][np.newaxis,:,:,:,np.newaxis]
				label = np.asarray(labels[j])
				print(label)
				label = label[np.newaxis,:]

				sigmoid, loss, train = sess.run([sigmoid_op,loss_op, train_op], feed_dict={input_placeholder: image, output_placeholder: label})
				print("step {}: loss: {}".format(i,loss))
				print("step {}: ground truth: {}".format(i,label))
				print("step {}: output: {}".format(i,sigmoid))

if __name__=="__main__":
	main()