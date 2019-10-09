import tensorflow as tf
import NiftiDataset
import numpy as np
import datetime

class MedicalImageClassifier(object):
	def __init__(self,sess,config):
		"""
		Args:
			sess: Tensorflow session
			config: Model configuration
		"""
		self.sess = sess
		self.config = config
		self.model = None
		self.graph = tf.Graph()
		self.graph.as_default()
		self.batch_size = 1
		self.input_channel_num = 1
		self.output_channel_num = 1
		self.input_placeholder = None
		self.output_placeholder = None
		self.image_filenames = []
		self.class_names = []
		self.train_data_dir = "./data/dataset/training"
		self.test_data_dir = "./data/dataset/testing"
		self.log_dir = "./tmp/log"
		self.ckpt_dir = "./tmp/ckpt"
		self.train_iterator = None
		self.test_iterator = None
		self.testing = True
		self.train_transforms = []
		self.test_transforms = []
		self.next_element_train = None
		self.next_element_test = None
		self.epoches = 999999999999999
		self.dimension = 3
		self.network = None
		self.restore_training = True

	def read_config(self):
		self.input_channel_num = len(self.config['TrainingSetting']['Data']['ImageFilenames'])
		self.output_class_num = len(self.config['TrainingSetting']['Data']['ClassNames'])

		self.batch_size = self.config['TrainingSetting']['BatchSize']
		self.patch_shape = self.config['TrainingSetting']['PatchShape']
		self.dimension = len(self.config['TrainingSetting']['PatchShape'])
		self.image_log = self.config['TrainingSetting']['ImageLog']

		self.image_filenames = self.config['TrainingSetting']['Data']['ImageFilenames']
		self.label_filename = self.config['TrainingSetting']['Data']['LabelFilename']
		self.class_names = self.config['TrainingSetting']['Data']['ClassNames']

		self.train_data_dir = self.config['TrainingSetting']['Data']['TrainingDataDirectory']
		self.test_data_dir = self.config['TrainingSetting']['Data']['TestingDataDirectory']
		self.testing = self.config['TrainingSetting']['Testing']

		self.restore_training = self.config['TrainingSetting']['Restore']
		self.log_dir = self.config['TrainingSetting']['LogDir']
		self.ckpt_dir = self.config['TrainingSetting']['CheckpointDir']

		self.epoches = self.config['TrainingSetting']['Epoches']

		self.network_name = self.config['TrainingSetting']['Network']['Name']
		self.network_dropout_rate = self.config['TrainingSetting']['Network']['Dropout']

		self.optimizer_name = self.config['TrainingSetting']['Optimizer']['Name']
		self.initial_learning_rate = self.config['TrainingSetting']['Optimizer']['InitialLearningRate']
		self.decay_factor = self.config['TrainingSetting']['Optimizer']['Decay']['Factor']
		self.decay_step = self.config['TrainingSetting']['Optimizer']['Decay']['Step']
		# self.spacing = self.config['TrainingSetting']['Spacing']

	def dataset_iterator(self,data_dir,transforms,train=True):
		# Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
		with tf.device('/cpu:0'):
			if self.dimension==2:
				Dataset = NiftiDataset.NiftiDataset2D()
				sys.exit('2D image pipeline under development')
			else:
				Dataset = NiftiDataset.NiftiDataset3D(
					data_dir=data_dir,
					image_filenames=self.image_filenames,
					label_filename=self.label_filename,
					class_names=self.class_names,
					transforms=transforms,
					train=train,
					)
			dataset = Dataset.get_dataset()
			dataset = dataset.shuffle(buffer_size=1)
			dataset = dataset.batch(self.batch_size,drop_remainder=False)

		return dataset.make_initializable_iterator()

	def build_model_graph(self):
		self.global_step = tf.train.get_or_create_global_step()

		if self.dimension==2:
			input_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.input_channel_num)
		elif self.dimension == 3:
			input_batch_shape = (None, self.patch_shape[0], self.patch_shape[1], self.patch_shape[2], self.input_channel_num)
		else:
			sys.exit('Invalid Patch Shape (length should be 2 or 3)')

		output_batch_shape = (None, self.output_channel_num)

		# create placeholder for data input 
		self.input_placeholder = tf.placeholder(tf.float32, input_batch_shape)
		self.output_placeholder = tf.placeholder(tf.float32, output_batch_shape)

		# plot input and output images to tensorboard
		if self.image_log:
			for batch in range(self.batch_size):
				for input_channel in range(self.input_channel_num):
					if self.patch_shape==2:
						image_log = tf.cast(self.input_placeholder[batch:batch+1,:,:,input_channel], dtype=tf.uint8)
						tf.summary.image(self.image_filenames[input_channel],image_log, max_outputs=self.batch_size)
					else:
						image_log = tf.cast(self.input_placeholder[batch:batch+1,:,:,:,input_channel], dtype=tf.uint8)
						tf.summary.image(self.image_filenames[input_channel],tf.transpose(image_log,[3,1,2,0]), max_outputs=self.patch_shape[-1])

		# training and testing augmentation pipeline
		self.train_transforms = [
			NiftiDataset.Normalization(),
			NiftiDataset.RandomNoise()
		]

		self.test_transforms = [
			NiftiDataset.Normalization()
		]

		#  get input and output datasets
		self.train_iterator = self.dataset_iterator(self.train_data_dir,self.train)
		self.next_element_train = self.train_iterator.get_next()

	def train(self):
		# read config to class variables
		self.read_config()

		"""Train the classifier"""
		self.build_model_graph()

		start_epoch = tf.get_variable("start_epoch", shape=[1], initializer=tf.zeros_initializer, dtype=tf.int32)
		start_epoch_inc = start_epoch.assign(start_epoch+1)

		# actual training cycle
		# Initialize all variables
		self.sess.run(tf.initializers.global_variables())
		print("{}: Start training...".format(datetime.datetime.now()))

		# loop over epochs
		for epoch in np.arange(start_epoch.eval(session=self.sess),self.epoches):
			print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch))

			# initialize iterator in each new epoch
			self.sess.run(self.train_iterator.initializer)

			# training phase
			while True:
				try:
					self.sess.run(tf.initializers.local_variables())
					images, label = self.sess.run(self.next_element_train)
					print(images.shape,label.shape)

				except tf.errors.OutOfRangeError:
					start_epoch_inc.op.run()
					break

# images = []

	# # load image data
	# reader = sitk.ImageFileReader()
	# reader.SetFileName("./data/dataset/training/A01/image_brain_mni.nii.gz")
	# images.append(reader.Execute())
	# reader.SetFileName("./data/dataset/training/A03/image_brain_mni.nii.gz")
	# images.append(reader.Execute())

	# labels = []
	# labels.append([1,0,0,0])
	# labels.append([1,1,0,0])

	# # normalize images
	# sigma = 2.5
	# for i in range(2):
	# 	statisticsFilter = sitk.StatisticsImageFilter()
	# 	statisticsFilter.Execute(images[i])

	# 	intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
	# 	intensityWindowingFilter.SetOutputMaximum(255)
	# 	intensityWindowingFilter.SetOutputMinimum(0)
	# 	intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+sigma*statisticsFilter.GetSigma());
	# 	intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-sigma*statisticsFilter.GetSigma());

	# 	images[i] = intensityWindowingFilter.Execute(images[i])

	# images_np = []
	# for i in range(2):
	# 	images_np.append(sitk.GetArrayFromImage(images[i]))

	# # graph
	# network = Lenet3D(
	# 	num_classes=4,
	# 	is_training=True,
	# 	activation_fn="relu",
	# 	keep_prob=1.0
	# 	)
	# logits_op = network.GetNetwork(input_placeholder)
	# logits_op = tf.reshape(logits_op, [-1,4])

	# loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_op,labels=output_placeholder))

	# sigmoid_op = tf.sigmoid(logits_op)


	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
	# train_op = optimizer.minimize(
	# 	loss=loss_op,
	# 	global_step=global_step
	# 	)

	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True

	# with tf.Session(config=config) as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	for i in range(9999):
	# 		for j in range(2):
	# 			image = images_np[j][np.newaxis,:,:,:,np.newaxis]
	# 			label = np.asarray(labels[j])
	# 			print(label)
	# 			label = label[np.newaxis,:]

	# 			sigmoid, loss, train = sess.run([sigmoid_op,loss_op, train_op], feed_dict={input_placeholder: image, output_placeholder: label})
	# 			print("step {}: loss: {}".format(i,loss))
	# 			print("step {}: ground truth: {}".format(i,label))
	# 			print("step {}: output: {}".format(i,sigmoid))