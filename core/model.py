import tensorflow as tf
from core import NiftiDataset
from core import networks
from core import transforms
import numpy as np
import datetime
import sys
import os
import shutil
import math
import multiprocessing
from tqdm import tqdm
import pandas as pd
from utils.report import report
import SimpleITK as sitk
np.set_printoptions(precision=4,suppress=True)

def dice_loss(y_true, y_pred, epsilon=1e-7,reduce_axis=-1, loss_type="sorensen"):
	numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=reduce_axis)
	if loss_type == "sorensen":
		denominator = tf.reduce_sum(y_true + y_pred, axis=reduce_axis)
	elif loss_type == "jaccard":
		denominator = tf.reduce_sum(y_true*y_true + y_pred*y_pred, axis=reduce_axis)
	else:
		raise Exception("Unknown loss type")

	return 1 - (numerator) / (denominator + epsilon)

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
		self.image_log = self.config['TrainingSetting']['ImageLog']

		self.image_filenames = self.config['TrainingSetting']['Data']['ImageFilenames']
		self.label_filename = self.config['TrainingSetting']['Data']['LabelFilename']
		self.class_names = self.config['TrainingSetting']['Data']['ClassNames']

		# additional features
		if ('AdditionalFeaturesFilename' in self.config["TrainingSetting"]['Data']) and ('AdditionalFeatures' in self.config["TrainingSetting"]['Data']):
			self.additional_features_filename = self.config["TrainingSetting"]['Data']
			self.additional_features = self.config["TrainingSetting"]['Data']['AdditionalFeatures']
			self.additional_features_num = len(self.config["TrainingSetting"]['Data']['AdditionalFeatures'])
		else:
			self.additional_features = []
			self.additional_features_num = 0

		self.train_data_dir = self.config['TrainingSetting']['Data']['TrainingDataDirectory']
		self.test_data_dir = self.config['TrainingSetting']['Data']['TestingDataDirectory']
		self.testing = self.config['TrainingSetting']['Testing']

		self.restore_training = self.config['TrainingSetting']['Restore']
		self.log_dir = self.config['TrainingSetting']['LogDir']
		self.ckpt_dir = self.config['TrainingSetting']['CheckpointDir']

		self.epoches = self.config['TrainingSetting']['Epoches']
		self.max_steps = self.config['TrainingSetting']['MaxSteps']
		self.log_interval = self.config['TrainingSetting']['LogInterval']
		self.testing_step_interval = self.config['TrainingSetting']['TestingStepInterval']

		self.network_name = self.config['Network']['Name']
		self.network_dropout_rate = self.config['Network']['Dropout']
		self.spacing = self.config['Network']['Spacing']
		self.patch_shape = self.config['Network']['PatchShape']
		self.dimension = len(self.config['Network']['PatchShape'])

		self.optimizer_name = self.config['TrainingSetting']['Optimizer']['Name']
		self.initial_learning_rate = self.config['TrainingSetting']['Optimizer']['InitialLearningRate']
		self.decay_factor = self.config['TrainingSetting']['Optimizer']['Decay']['Factor']
		self.decay_step = self.config['TrainingSetting']['Optimizer']['Decay']['Step']
		self.loss_fn = self.config['TrainingSetting']['LossFunction']['Name']
		self.classificatin_type = self.config['TrainingSetting']['LossFunction']['Multiclass/Multilabel']

		self.model_path = self.config['PredictionSetting']['ModelPath']
		self.checkpoint_path = self.config['PredictionSetting']['CheckPointPath']
		self.evaluation_data_dir = self.config['PredictionSetting']['Data']['EvaluationDataDirectory']
		self.report_output = self.config['PredictionSetting']['ReportOutput']
		self.evaluation_output_filename = self.config['PredictionSetting']['Data']['OutputFilename']

	def dataset_iterator(self,data_dir,transforms,train=True):
		# Force input pipepline to CPU:0 to avoid operations sometimes ended up at GPU and resulting a slow down
		with tf.device('/cpu:0'):
			if self.additional_features_num == 0:
				Dataset = NiftiDataset.NiftiDataset(
					data_dir=data_dir,
					image_filenames=self.image_filenames,
					label_filename=self.label_filename,
					case_column_name = "case",
					class_names=self.class_names,
					transforms=transforms,
					train=train
					)
			else:
				Dataset = NiftiDataset.NiftiDataset(
					data_dir=data_dir,
					image_filenames=self.image_filenames,
					label_filename=self.label_filename,
					case_column_name = "case",
					class_names=self.class_names,
					additional_features_filename = self.additional_features_filename,
					additional_features = self.additional_features,
					transforms=transforms,
					train=train
					)
			dataset = Dataset.get_dataset()
			if self.dimension == 2:
				dataset = dataset.shuffle(buffer_size=multiprocessing.cpu_count())
			else:
				dataset = dataset.shuffle(buffer_size=multiprocessing.cpu_count())
			dataset = dataset.batch(self.batch_size,drop_remainder=False)
			dataset = dataset.prefetch(1)

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
		self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout')

		# placeholder for additional features
		if self.additional_features_num > 0:
			additional_batch_shape = (None, self.additional_features_num)
			self.additional_features_placeholder = tf.placeholder(tf.float32,additional_batch_shape)

		# plot input and output images to tensorboard
		if self.image_log:
			if len(self.patch_shape)==2:
				for input_channel in range(self.input_channel_num):
					
					image_log = tf.cast(self.input_placeholder[:,:,:,input_channel:input_channel+1], dtype=tf.uint8)
					tf.summary.image(self.image_filenames[input_channel],image_log, max_outputs=self.batch_size)
			else:
				for batch in range(self.batch_size):
					for input_channel in range(self.input_channel_num):
						image_log = tf.cast(self.input_placeholder[batch:batch+1,:,:,:,input_channel], dtype=tf.uint8)
						tf.summary.image(self.image_filenames[input_channel],tf.transpose(image_log,[3,1,2,0]), max_outputs=self.patch_shape[-1])

		# training and testing augmentation pipeline
		self.train_transforms = transforms.train_transforms(self.spacing, self.patch_shape)
		self.test_transforms = transforms.test_transforms(self.spacing, self.patch_shape)

		#  get input and output datasets
		self.train_iterator = self.dataset_iterator(self.train_data_dir,self.train_transforms)
		self.next_element_train = self.train_iterator.get_next()

		if self.testing:
			self.test_iterator = self.dataset_iterator(self.test_data_dir,self.test_transforms)
			self.next_element_test = self.test_iterator.get_next()

		# network models
		print("{}: Network: {}".format(datetime.datetime.now(),self.network_name))
		if self.network_name == "LeNet":
			if self.dimension == 2:
				self.network = networks.Lenet2D(
					num_classes=self.output_channel_num,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder
					)
			else:	
				self.network = networks.Lenet3D(
					num_classes=self.output_channel_num,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder
					)
		elif self.network_name == "AlexNet":
			if self.dimension == 2:
				self.network = networks.Alexnet2D(
					num_classes=self.output_channel_num,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder
					)
			else:
				self.network = networks.Alexnet3D(
					num_classes=self.output_channel_num,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder
					)
		elif "Inception" in self.network_name and "ResNet" not in self.network_name:
			if self.dimension == 2:
				self.network = networks.InceptionNet2D(
					num_classes=self.output_channel_num,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder,
					version=int(self.network_name[-1])
					)
			else:
				exit()
		elif "Inception" in self.network_name and "ResNet" in self.network_name:
			if self.dimension == 2:
				self.network = networks.InceptionNet2D(
					num_classes=self.output_channel_num,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder,
					version=int(self.network_name[-1]),
					residual=True
					)
			else:
				exit()
		elif self.network_name == "Vgg":
			if self.dimension == 2:
				self.network = networks.Vgg2D(
					num_classes=self.output_channel_num,
					num_channels=64,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder,
					module_config=[2,2,3,3,3],
					batch_norm_momentum=0.99,
					fc_channels=[4096,4096])
			else:
				self.network = networks.Vgg3D(
					num_classes=self.output_channel_num,
					num_channels=64,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder,
					module_config=[2,2,3,3,3],
					batch_norm_momentum=0.99,
					fc_channels=[4096,4096])
		elif self.network_name == "ResNet":
			if self.dimension == 2:
				self.network = networks.Resnet2D(
					num_classes=self.output_channel_num,
					num_channels=64,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder,
					init_conv_shape=5,
					init_pool=True,
					module_config=[2,2,2])
			else:
				self.network = networks.Resnet3D(
					num_classes=self.output_channel_num,
					num_channels=64,
					is_training=True,
					activation_fn="relu",
					dropout=self.dropout_placeholder,
					init_conv_shape=5,
					init_pool=True,
					module_config=[2])
		else:
			sys.exit('Invalid Network')

		self.logits_op = self.network.GetNetwork(self.input_placeholder)
		self.logits_op = tf.reshape(self.logits_op, [-1,self.output_channel_num])

		if self.additional_features_num > 0:
			dense0 = tf.concat([tf.nn.relu(self.logits_op),self.additional_features_placeholder],axis=1)
			dense1 = tf.layers.dense(inputs=dense0,units=500, activation=self.activation_fn)
			#self.logits_additional_features_op = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)
			self.logits_op = tf.layers.dense(inputs=dense1,units=self.num_classes, activation=None)

		if self.classificatin_type == "Multilabel":
			self.prob_op = tf.sigmoid(self.logits_op)
			self.result_op = tf.cast(tf.math.round(self.prob_op),dtype=tf.uint8)
			if self.loss_fn == "xent":
				self.sigmoid_xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_op,labels=self.output_placeholder)
				self.loss_op = tf.reduce_mean(self.sigmoid_xent,0)
			elif self.loss_fn == "sorensen":
				self.loss_op = dice_loss(y_true=self.output_placeholder,y_pred=self.prob_op)
			elif self.loss_fn == "jaccard":
				self.loss_op = dice_loss(y_true=self.output_placeholder,y_pred=self.prob_op,loss_type="jaccard")
		elif self.classificatin_type == "Multiclass":
			self.prob_op = tf.nn.softmax(self.logits_op)
			self.result_op = tf.cast(tf.one_hot(tf.argmax(self.prob_op,1),self.output_channel_num),dtype=tf.uint8)
			if self.loss_fn == "xent":
				self.softmax_xent = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_op,labels=self.output_placeholder)
				self.loss_op = tf.reduce_mean(self.softmax_xent,0)
			elif self.loss_fn == "sorensen":
				self.loss_op = dice_loss(y_true=self.output_placeholder,y_pred=self.prob_op)
			elif self.loss_fn == "jaccard":
				self.loss_op = dice_loss(y_true=self.output_placeholder,y_pred=self.prob_op,loss_type="jaccard")
		else:
			exit("Classification type can only be \"Multiclass\" or \"Multilabel\", training abort")

		self.avg_loss_op = tf.reduce_mean(self.loss_op)

		tf.summary.scalar('loss/average', self.avg_loss_op)
		# tf.summary.scalar('accuracy/overall',self.acc_op)

		self.class_tp = []
		self.class_tn = []
		self.class_fp = []
		self.class_fn = []
		self.class_auc = []
		class_accuracy = []
		class_precision = []
		class_sensitivity = []
		class_specificity = []

		for i in range(len(self.class_names)):
			epsilon = 1e-6
			class_acc, class_acc_op = tf.metrics.accuracy(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
			class_tp, class_tp_op = tf.metrics.true_positives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
			class_tn, class_tn_op = tf.metrics.true_negatives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
			class_fp, class_fp_op = tf.metrics.false_positives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
			class_fn, class_fn_op = tf.metrics.false_negatives(labels=tf.cast(self.output_placeholder[:,i],dtype=tf.uint8), predictions=tf.cast(self.result_op[:,i],dtype=tf.uint8))
			precision = class_tp_op/(class_tp_op+class_fp_op+epsilon)
			sensitivity = class_tp_op/(class_tp_op+class_fn_op+epsilon)
			specificity = class_tn_op/(class_tn_op+class_fp_op+epsilon)

			# # remove nan values
			# nan_replace_value = 1.0
			# accuracy = tf.where(tf.is_nan(class_acc_op), tf.ones_like(class_acc_op) * nan_replace_value, class_acc_op)
			# precision = tf.where(tf.is_nan(precision), tf.ones_like(precision) * nan_replace_value, precision)
			# sensitivity = tf.where(tf.is_nan(sensitivity), tf.ones_like(sensitivity) * nan_replace_value, sensitivity)
			# specificity = tf.where(tf.is_nan(specificity), tf.ones_like(specificity) * nan_replace_value, specificity)

			accuracy = class_acc_op

			class_auc, class_auc_op = tf.metrics.auc(labels=self.output_placeholder[:,i], predictions=self.prob_op[:,i])

			self.class_tp.append(class_tp_op)
			self.class_tn.append(class_tn_op)
			self.class_fp.append(class_fp_op)
			self.class_fn.append(class_fn_op)
			self.class_auc.append(class_auc_op)
			class_accuracy.append(accuracy)
			class_precision.append(precision)
			class_sensitivity.append(sensitivity)
			class_specificity.append(specificity)

			if self.classificatin_type == "Multilabel":
				tf.summary.scalar('loss/' + self.class_names[i], self.loss_op[i])
			tf.summary.scalar('accuracy/' + self.class_names[i], accuracy)
			tf.summary.scalar('precision/' + self.class_names[i],precision)
			tf.summary.scalar('sensitivity/' + self.class_names[i],sensitivity)
			tf.summary.scalar('specificity/' + self.class_names[i],specificity)
			tf.summary.scalar('auc/' + self.class_names[i],class_auc_op)

		avg_accuracy = tf.reduce_mean(class_accuracy)
		avg_precision = tf.reduce_mean(class_precision)
		avg_sensitivity = tf.reduce_mean(class_sensitivity)
		avg_specificity = tf.reduce_mean(class_specificity)
		avg_auc = tf.reduce_mean(self.class_auc)
		tf.summary.scalar('accuracy/average',avg_accuracy)
		tf.summary.scalar('precision/average', avg_precision)
		tf.summary.scalar('sensitivity/average', avg_sensitivity)
		tf.summary.scalar('specificity/average', avg_specificity)
		tf.summary.scalar('auc/average', avg_auc)

		self.acc_op = avg_accuracy
		self.precision_op = avg_precision
		self.sensitivity_op = avg_sensitivity
		self.specificity_op = avg_specificity
		self.auc_op = avg_auc

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
			elif self.optimizer_name == "Momentum":
				optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,momentum=0.9)
			elif self.optimizer_name == "Adam":
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)
			else:
				sys.exit('Invalid Optimizer')

			# for batch norm to function normally
			update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)

			train_op = optimizer.minimize(
				loss=self.avg_loss_op,
				global_step=self.global_step
				)

			train_op = tf.group([train_op, update_ops])

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

		# testing initializer need to execute outside training loop
		if self.testing:
			self.sess.run(self.test_iterator.initializer)

		# loop over epochs
		for epoch in np.arange(start_epoch.eval(session=self.sess),self.epoches):
			print("{}: Epoch {} starts...".format(datetime.datetime.now(),epoch+1))

			# initialize iterator in each new epoch
			self.sess.run(self.train_iterator.initializer)

			# training phase
			while True:
				if self.global_step.eval() > self.max_steps:
					sys.exit("Reach maximum training steps")
				try:
					self.sess.run(tf.initializers.local_variables())
					images, label = self.sess.run(self.next_element_train)
					# print("{}: step {}: get next element ok".format(datetime.datetime.now(), self.global_step.eval()))
					print("step {}: get next element ok".format(self.global_step.eval()))
					if images.shape[0] < self.batch_size:
						# if self.dimension == 2:
						# 	images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3]))
						# 	label_zero_pads = np.zeros((self.batch_size-label.shape[0],images.shape[1]))
						# else:
						# 	images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
						
						# label_zero_pads = np.zeros((self.batch_size-label.shape[0],label.shape[1]))
						# images = np.concatenate((images,images_zero_pads))
						# label = np.concatenate((label,label_zero_pads))
						if self.dimension == 2:
							images = np.tile(images,(math.ceil(self.batch_size/images.shape[0]),1,1,1))
						else:
							images = np.tile(images,(math.ceil(self.batch_size/images.shape[0]),1,1,1,1))
						label = np.tile(label,(math.ceil(self.batch_size/label.shape[0]),1))

						images = images[:self.batch_size,]
						label = label[:self.batch_size,]

					prob, loss, result, accuracy, precision, sensitivity, specificity, auc, train = self.sess.run(
						[self.prob_op,self.avg_loss_op, self.result_op, self.acc_op, self.precision_op, self.sensitivity_op, self.specificity_op, self.auc_op, train_op], 
						feed_dict={
							self.input_placeholder: images, 
							self.output_placeholder: label,
							self.dropout_placeholder: self.network_dropout_rate,
							self.network.is_training: True})
					print("{}: Training loss: {:.4f}".format(datetime.datetime.now(),loss))
					print("{}: Training accuracy: {:.4f}".format(datetime.datetime.now(),accuracy))
					print("{}: Training precision: {:.4f}".format(datetime.datetime.now(),precision))
					print("{}: Training sensitivity: {:.4f}".format(datetime.datetime.now(),sensitivity))
					print("{}: Training specificity: {:.4f}".format(datetime.datetime.now(),specificity))
					print("{}: Training auc: {:.4f}".format(datetime.datetime.now(),auc))
					print("{}: Training ground truth: \n{}".format(datetime.datetime.now(),label[:]))
					print("{}: Training result: \n{}".format(datetime.datetime.now(),result[:]))
					print("{}: Training probability: \n{}".format(datetime.datetime.now(),prob[:]))

					# perform summary log after training op
					summary = self.sess.run(summary_op,feed_dict={
						self.input_placeholder: images,
						self.output_placeholder: label,
						self.dropout_placeholder: 0.0,
						self.network.is_training: False
						})

					train_summary_writer.add_summary(summary,global_step=tf.train.global_step(self.sess,self.global_step))
					train_summary_writer.flush()

					# save checkpoint
					if self.global_step.eval()%self.log_interval == 0:
						print("{}: Saving checkpoint of step {} at {}...".format(datetime.datetime.now(),self.global_step.eval(),self.ckpt_dir))
						if not (os.path.exists(self.ckpt_dir)):
							os.makedirs(self.ckpt_dir,exist_ok=True)
						saver.save(self.sess, checkpoint_prefix, 
							global_step=tf.train.global_step(self.sess, self.global_step),
							latest_filename="checkpoint-latest")

					# testing phase
					if self.testing and (self.global_step.eval()%self.testing_step_interval == 0):
						# self.network.is_training = False

						try:
							images, label = self.sess.run(self.next_element_test)
						except tf.errors.OutOfRangeError:
							self.sess.run(self.test_iterator.initializer)
							images, label = self.sess.run(self.next_element_test)
							
						if images.shape[0] < self.batch_size:
							# 	if self.dimension == 2:
							# 		images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3]))
							# 		label_zero_pads = np.zeros((self.batch_size-label.shape[0],images.shape[1]))
							# 	else:
							# 		images_zero_pads = np.zeros((self.batch_size-images.shape[0],images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
								
							# label_zero_pads = np.zeros((self.batch_size-label.shape[0],label.shape[1]))
			 				# images = np.concatenate((images,images_zero_pads))
							# label = np.concatenate((label,label_zero_pads))

							if self.dimension == 2:
								images = np.tile(images,(math.ceil(self.batch_size/images.shape[0]),1,1,1))
							else:
								images = np.tile(images,(math.ceil(self.batch_size/images.shape[0]),1,1,1,1))
							label = np.tile(label,(math.ceil(self.batch_size/label.shape[0]),1))

							images = images[:self.batch_size,]
							label = label[:self.batch_size,]

						prob, loss, result, accuracy, precision, sensitivity, specificity, auc, summary = self.sess.run(
							[self.prob_op,self.avg_loss_op, self.result_op, self.acc_op, self.precision_op, self.sensitivity_op, self.specificity_op, self.auc_op, summary_op], 
							feed_dict={
								self.input_placeholder: images, 
								self.output_placeholder: label,
								self.dropout_placeholder: 0.0,
								self.network.is_training: False})
						print("{}: Testing loss: {:.4f}".format(datetime.datetime.now(),loss))
						print("{}: Testing accuracy: {:.4f}".format(datetime.datetime.now(),accuracy))
						print("{}: Testing precision: {:.4f}".format(datetime.datetime.now(),precision))
						print("{}: Testing sensitivity: {:.4f}".format(datetime.datetime.now(),sensitivity))
						print("{}: Testing specificity: {:.4f}".format(datetime.datetime.now(),specificity))
						print("{}: Testing auc: {:.4f}".format(datetime.datetime.now(),auc))
						print("{}: Testing ground truth: \n{}".format(datetime.datetime.now(),label[:5]))
						print("{}: Testing result: \n{}".format(datetime.datetime.now(),result[:5]))
						print("{}: Testing prob: \n{}".format(datetime.datetime.now(),prob[:5]))

						test_summary_writer.add_summary(summary, global_step=tf.train.global_step(self.sess, self.global_step))
						test_summary_writer.flush()

				except tf.errors.OutOfRangeError:
					start_epoch_inc.op.run()
					# self.network.is_training = False

					print("{}: Saving checkpoint of epoch {} at {}...".format(datetime.datetime.now(),epoch+1,self.ckpt_dir))
					if not (os.path.exists(self.ckpt_dir)):
						os.makedirs(self.ckpt_dir,exist_ok=True)
					saver.save(self.sess, checkpoint_prefix, 
						global_step=tf.train.global_step(self.sess, self.global_step),
						latest_filename="checkpoint-latest")
					print("{}: Saving checkpoint succeed".format(datetime.datetime.now()))

					break

		# close tensorboard summary writer
		train_summary_writer.close()
		if self.testing:
			test_summary_writer.close()

	def predict(self):
		# read config to class variables
		self.read_config()

		# restore model grpah
		# tf.reset_default_graph()
		imported_meta = tf.train.import_meta_graph(self.model_path)

		# create transformation to image
		predict_transforms = transforms.predict_transforms(self.spacing, self.patch_shape)

		print("{}: Start evaluation...".format(datetime.datetime.now()))

		imported_meta.restore(self.sess, self.checkpoint_path)
		print("{}: Restore checkpoint success".format(datetime.datetime.now()))

		# create output csv file
		columns = ['case']
		for class_name in self.class_names:
			columns.append(class_name)

		output_df = pd.DataFrame(columns=columns)

		pbar = tqdm(os.listdir(self.evaluation_data_dir))
		for case in pbar:
			pbar.set_description(case)

			# check image data exists
			image_paths = []
			image_file_exists = True
			for image_channel in range(self.input_channel_num):
				image_paths.append(os.path.join(self.evaluation_data_dir,case,self.config['PredictionSetting']['Data']['ImageFilenames'][image_channel]))

				if not os.path.exists(image_paths[image_channel]):
					image_file_exists = False
					break

			if not image_file_exists:
				print("{}: Image file not found at {}".format(datetime.datetime.now(),os.path.dirname(image_paths[0])))
				break

			print("{}: Evaluating image at {}".format(datetime.datetime.now(),os.path.dirname(image_paths[0])))

			# read image file
			images = []
			images_tfm = []

			for image_channel in range(self.input_channel_num):
				reader = sitk.ImageFileReader()
				reader.SetFileName(image_paths[image_channel])
				image = reader.Execute()
				images.append(image)
				# preprocess the image and label before inference
				images_tfm.append(image)

			sample = {'images':images_tfm}

			for predict_tfm in predict_transforms:
				try:
					sample = predict_tfm(sample)
				except:
					tqdm.write("Dataset preprocessing error: {}: {}".format(case,predict_tfm.name))
					exit()

			images_tfm = sample['images']

			# convert image to numpy array
			for image_channel in range(self.input_channel_num):
				image_ = sitk.GetArrayFromImage(images_tfm[image_channel])
				image_ = np.asarray(image_,np.float32)

				if self.dimension == 2:
					if image_channel == 0:
						images_np = image_[:,:,np.newaxis]
					else:
						images_np = np.append(images_np, image_[:,:,np.newaxis], axis=-1)
				elif self.dimension == 3:
					# to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
					image_ = np.transpose(image_,(2,1,0))
					if image_channel == 0:
						images_np = image_[:,:,:,np.newaxis]
					else:
						images_np = np.append(images_np, image_[:,:,:,np.newaxis], axis=-1)

			images_np = images_np[np.newaxis,]

			if self.classificatin_type == "Multilabel":
				prob = self.sess.run(['Sigmoid:0'], feed_dict={
							'Placeholder:0': images_np,
							'dropout:0':0.0,
							'train_phase_placeholder:0':False})
			elif self.classificatin_type == "Multiclass":
				prob = self.sess.run(['Softmax:0'], feed_dict={
							'Placeholder:0': images_np,
							'dropout:0':0.0,
							'train_phase_placeholder:0':False})

			output = {'case': case}

			for channel in range(self.output_channel_num):
				if self.input_channel_num == 1:
					output[self.class_names[channel]] = prob[channel]
				else:
					output[self.class_names[channel]] = prob[0][0,channel]
			output_df = output_df.append(output, ignore_index=True)

			if self.report_output:
				doc = report.Report(images=images,result=prob[0],class_names=self.class_names)
				doc.WritePdf(os.path.join(self.evaluation_data_dir,case,"report"))

			tqdm.write("{}: Evaluation of {} complete:".format(datetime.datetime.now(), case))
			for i in range(self.output_channel_num):
				if self.input_channel_num == 1:
					tqdm.write("{}: {}%".format(self.class_names[i],prob[0][i]*100))
				else:
					tqdm.write("{}: {}%".format(self.class_names[i],prob[0][0,i]*100))

		# write csv
		output_df.to_csv(self.evaluation_output_filename,index=False)
