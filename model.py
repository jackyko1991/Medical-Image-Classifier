import tensorflow as tf
import NiftiDataset
import numpy as np
import datetime
import networks
import sys
import os
import shutil

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
		self.output_channel_num = len(self.config['TrainingSetting']['Data']['ClassNames'])

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
		self.spacing = self.config['TrainingSetting']['Spacing']

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
		self.train_iterator = self.dataset_iterator(self.train_data_dir,self.train_transforms)
		self.next_element_train = self.train_iterator.get_next()

		if self.testing:
			self.test_iterator = self.dataset_iterator(self.test_data_dir,self.test_transforms)
			self.next_element_test = self.test_iterator.get_next()

		# network models
		if self.network_name == "LeNet":
			self.network = networks.Lenet3D(
			num_classes=self.output_channel_num,
			is_training=True,
			activation_fn="relu",
			keep_prob=1.0
			)
		else:
			sys.exit('Invalid Network')

		self.logits_op = self.network.GetNetwork(self.input_placeholder)
		self.logits_op = tf.reshape(self.logits_op, [-1,self.output_channel_num])

		self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_op,labels=self.output_placeholder))

		self.sigmoid_op = tf.sigmoid(self.logits_op)
		self.result_op = tf.math.round(self.sigmoid_op)

		acc, self.acc_op = tf.metrics.accuracy(labels=self.output_placeholder, predictions=self.result_op)

		tf.summary.scalar('loss', self.loss_op)
		tf.summary.scalar('accuracy',self.acc_op)

	def train(self):
		# read config to class variables
		self.read_config()

		"""Train the classifier"""
		self.build_model_graph()

		# learning rate
		with tf.name_scope("learning_rate"):
			self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step,
				self.decay_step, self.decay_factor, staircase=False, name="learning_rate")
		tf.summary.scalar('learning_rate', self.learning_rate)

		# optimizer
		with tf.name_scope("optimizer"):
			if self.optimizer_name == "GradientDescent":
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
			else:
				sys.exit('Invalid Optimizer')

			train_op = optimizer.minimize(
				loss=self.loss_op,
				global_step=self.global_step
				)

		start_epoch = tf.get_variable("start_epoch", shape=[1], initializer=tf.zeros_initializer, dtype=tf.int32)
		start_epoch_inc = start_epoch.assign(start_epoch+1)

		# actual training cycle
		# Initialize all variables
		self.sess.run(tf.initializers.global_variables())
		print("{}: Start training...".format(datetime.datetime.now()))

		#  saver
		print("{}: Setting up Saver...".format(datetime.datetime.now()))

		if not self.restore_training:
			# clear log directory
			if os.path.exists(self.log_dir):
				shutil.rmtree(self.log_dir)
			os.makedirs(self.log_dir)

			# clear checkpoint directory
			if os.path.exists(self.ckpt_dir):
				shutil.rmtree(self.ckpt_dir)
			os.makedirs(self.ckpt_dir)
			saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
			checkpoint_prefix = os.path.join(self.ckpt_dir,"checkpoint")
		else:
			saver = tf.train.Saver(keep_checkpoint_every_n_hours=5)
			checkpoint_prefix = os.path.join(self.ckpt_dir,"checkpoint")

			# check if checkpoint exists
			if os.path.exists(checkpoint_prefix+"-latest"):
				print("{}: Last checkpoint found at {}, loading...".format(datetime.datetime.now(),self.ckpt_dir))
				latest_checkpoint_path = tf.train.latest_checkpoint(self.ckpt_dir,latest_filename="checkpoint-latest")
				saver.restore(self.sess, latest_checkpoint_path)
			
			print("{}: Last checkpoint epoch: {}".format(datetime.datetime.now(),start_epoch.eval(session=self.sess)[0]))
			print("{}: Last checkpoint global step: {}".format(datetime.datetime.now(),tf.train.global_step(self.sess, self.global_step)))


		
		summary_op = tf.summary.merge_all()
		train_summary_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
		if self.testing:
			test_summary_writer = tf.summary.FileWriter(self.log_dir + '/test', self.sess.graph)

		# loop over epochs
		for epoch in np.arange(start_epoch.eval(session=self.sess),self.epoches):
			print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch+1))

			# initialize iterator in each new epoch
			self.sess.run(self.train_iterator.initializer)

			# training phase
			while True:
				try:
					self.sess.run(tf.initializers.local_variables())
					images, label = self.sess.run(self.next_element_train)
					if images.shape[0] < self.batch_size:
						if self.dimension == 2:
							images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3]))
							label_zero_pads = np.zeros((self.batch_size-label.shape[0],images.shape[1]))
						else:
							images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
						
						label_zero_pads = np.zeros((self.batch_size-label.shape[0],label.shape[1]))
						images = np.concatenate((images,images_zero_pads))
						label = np.concatenate((label,label_zero_pads))

					sigmoid, loss, result, accuracy, train = self.sess.run(
						[self.sigmoid_op,self.loss_op, self.result_op, self.acc_op, train_op], 
						feed_dict={self.input_placeholder: images, self.output_placeholder: label})
					print("{}: loss: {}".format(datetime.datetime.now(),loss))
					print("{}: accuracy: {}".format(datetime.datetime.now(),accuracy))
					print("{}: ground truth: {}".format(datetime.datetime.now(),label))
					print("{}: result: {}".format(datetime.datetime.now(),result))
					print("{}: sigmoid: {}".format(datetime.datetime.now(),sigmoid))

					# perform summary log after training op
					summary = self.sess.run(summary_op,feed_dict={
						self.input_placeholder: images,
						self.output_placeholder: label
						})

					train_summary_writer.add_summary(summary,global_step=tf.train.global_step(self.sess,self.global_step))
					train_summary_writer.flush()

				except tf.errors.OutOfRangeError:
					start_epoch_inc.op.run()
					self.network.is_training = False

					print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,self.ckpt_dir))
					if not (os.path.exists(self.ckpt_dir)):
						os.makedirs(self.ckpt_dir,exist_ok=True)
					saver.save(self.sess, checkpoint_prefix, 
						global_step=tf.train.global_step(self.sess, self.global_step),
						latest_filename="checkpoint-latest")
					print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))

					break

			# testing phase
			if self.testing:
				print("{}: Training of epoch {} finishes, testing start".format(datetime.datetime.now(),epoch+1))
				self.sess.run(self.test_iterator.initializer)
				while True:
					try:
						self.sess.run(tf.local_variables_initializer())
						images, label = self.sess.run(self.next_element_test)
						if images.shape[0] < self.batch_size:
							if self.dimension == 2:
								images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3]))
								label_zero_pads = np.zeros((self.batch_size-label.shape[0],images.shape[1]))
							else:
								images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
							
							label_zero_pads = np.zeros((self.batch_size-label.shape[0],label.shape[1]))
							images = np.concatenate((images,images_zero_pads))
							label = np.concatenate((label,label_zero_pads))
						
						self.network.is_training = False;
						sigmoid, loss, result, accuracy, train = self.sess.run(
							[self.sigmoid_op,self.loss_op, self.result_op, self.acc_op, train_op], 
							feed_dict={self.input_placeholder: images, self.output_placeholder: label})
						print("{}: loss: {}".format(datetime.datetime.now(),loss))
						print("{}: accuracy: {}".format(datetime.datetime.now(),accuracy))
						print("{}: ground truth: {}".format(datetime.datetime.now(),label))
						print("{}: result: {}".format(datetime.datetime.now(),result))
						print("{}: sigmoid: {}".format(datetime.datetime.now(),sigmoid))

						# perform summary log after testing op
						summary = self.sess.run(summary_op,feed_dict={
							self.input_placeholder: images,
							self.output_placeholder: label
							})

						test_summary_writer.add_summary(summary, global_step=tf.train.global_step(self.sess, self.global_step))
						test_summary_writer.flush()

					except tf.errors.OutOfRangeError:
						break

		# close tensorboard summary writer
		train_summary_writer.close()
		if FLAGS.testing:
			test_summary_writer.close()