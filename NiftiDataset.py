import SimpleITK as sitk
import tensorflow as tf
import os
import numpy as np
import math
import random
import multiprocessing
import pandas as pd

class NiftiDataset2D(object):
	"""
	"""

	def __init__(self,
		data_dir = '',
		source_filenames = [],
		target_filenames = [],
		transforms=None,
		train=False):
		self.data_dir = data_dir
		self.source_filenames = source_filenames
		self.target_filenames = target_filenames
		self.transforms = transforms
		self.train = train

	def read_image(self,path):
		reader = sitk.ImageFileReader()
		reader.SetFileName(path)
		return reader.Execute()

	def get_dataset(self):
		slices_list = []
		# read all images to generate the candidate slice list
		for case in os.listdir(self.data_dir):
			image = self.read_image(os.path.join(self.data_dir,case,self.source_filenames[0]))
			for i in range(image.GetSize()[2]):
				slices_list.append({'case':case,'slice':i})

		dataset = tf.data.Dataset.from_tensor_slices(slices_list)
		dataset = dataset.map(lambda slice_: tuple(tf.py_func(
			self.input_parser, slice_, [tf.float32,tf.float32])),
			num_parallel_calls=multiprocessing.cpu_count())

		self.dataset = dataset
		self.data_size = len(source_paths)
		return self.dataset

	def input_parser(self,slice_):
		# read source and target images
		source_images = []
		target_images = []

class NiftiDataset3D(object):
	"""
	load image-label pair for training, testing and inference.
	Currently only support linear interpolation method
	Args:
		data_dir (string): Path to data directory.
		source_filename (string): Filename of source image data.
		target_filename (string): Filename of target image data.
		transforms (list): List of SimpleITK image transformations.
		train (bool): Determine whether the dataset class run in training/inference mode. When set to false, an empty label with same metadata as image is generated.
	"""

	def __init__(self,
		data_dir = '',
		image_filenames = [],
		label_filename = 'label.csv',
		class_names = [],
		transforms=None,
		train=False,):

		# Init membership variables
		self.data_dir = data_dir
		self.image_filenames = image_filenames
		self.label_filename = label_filename
		self.class_names = class_names
		self.transforms = transforms
		self.train = train
		self.label_df = None

	def get_dataset(self):
		image_dirs = []

		# read labels from csv file
		self.label_df = pd.read_csv(self.label_filename)

		dataset = tf.data.Dataset.from_tensor_slices(os.listdir(self.data_dir))
		dataset = dataset.map(lambda case: tuple(tf.py_func(
			self.input_parser, [case], [tf.float32,tf.int64])),
			num_parallel_calls=multiprocessing.cpu_count())
		self.dataset = dataset
		self.data_size = len(os.listdir(self.data_dir))
		return self.dataset

	def read_image(self,path):
		reader = sitk.ImageFileReader()
		reader.SetFileName(path)
		return reader.Execute()

	def input_parser(self, case):
		case = case.decode("utf-8")

		# read images
		images = []

		for channel in range(len(self.image_filenames)):
			image_ = self.read_image(os.path.join(self.data_dir,case,self.image_filenames[channel]))
			images.append(image_)

		# cast images
		castImageFilter = sitk.CastImageFilter()
		castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
		for channel in range(len(images)):
			images[channel] = castImageFilter.Execute(images[channel])
			# check header consistency
			sameSize = images[channel].GetSize() == images[0].GetSize()
			sameSpacing = images[channel].GetSpacing() == images[0].GetSpacing()
			sameDirection = images[channel].GetDirection() == images[0].GetDirection()

			if sameSize and sameSpacing and sameDirection:
				continue
			else:
				raise Exception('Header info inconsistent: {}'.format(image_paths[channel]))
				exit()

		# get the associate label
		label = self.label_df.loc[self.label_df['Case no.']==case].iloc[0].values.tolist()[1:]
		
		sample = {'images':images, 'label':label}

		# if self.transforms:
		# 	for transform in self.transforms:
		# 		try:
		# 			sample = transform(sample)
		# 		except:
		# 			print("Dataset preprocessing error: {}".format(os.path.dirname(source_paths[0])))
		# 			exit()

		# convert sample to tf tensors
		for channel in range(len(sample['images'])):
			images_np_ = sitk.GetArrayFromImage(sample['images'][channel])
			images_np_ = np.asarray(images_np_,np.float32)
			# to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
			images_np_ = np.transpose(images_np_,(2,1,0))
			if channel == 0:
				images_np = images_np_[:,:,:,np.newaxis]
			else:
				images_np = np.append(images_np,images_np_[:,:,:,np.newaxis],axis=-1)

		return images_np, label

class Normalization(object):
	"""
	Normalize an image to 0 - 255
	"""

	def __init__(self):
		self.name = 'Normalization'

	def __call__(self, sample):
		sources, targets = sample['sources'], sample['targets']
		# normalizeFilter = sitk.NormalizeImageFilter()
		# image, label = sample['image'], sample['label']
		# image = normalizeFilter.Execute(image)
		resacleFilter = sitk.RescaleIntensityImageFilter()
		resacleFilter.SetOutputMaximum(255)
		resacleFilter.SetOutputMinimum(0)

		for channel in range(len(sources)):
			sources[channel] = resacleFilter.Execute(sources[channel])

		for channel in range(len(targets)):
			targets[channel] = resacleFilter.Execute(targets[channel])

		return {'sources': sources, 'targets': targets}

# class RandomFlip(object):
# 	"""
# 	Randomly Flip image by user specified axes
# 	"""

# 	def __init__(self, axes):
# 		self.name = 'Flip'
# 		assert len(axes)>0 and len(axes)<=3
# 		self.axes = axes

# 	def __call__(self, sample):
# 		image, label = sample['image'], sample['label']

# 		flip = np.random.randint(2, size=1)[0]
# 		if flip:
# 			flipFilter = sitk.FlipImageFilter()
# 			for image_channel in range(len(image)):
# 				flipFilter.SetFlipAxes(self.axes)
# 				image[image_channel] = flipFilter.Execute(image[image_channel])
# 			label = flipFilter.Execute(label)

# 		return {'image': image, 'label': label}

class StatisticalNormalization(object):
	"""
	Normalize an image by mapping intensity with intensity distribution
	"""

	def __init__(self, sigma, pre_norm=False):
		self.name = 'StatisticalNormalization'
		assert isinstance(sigma, float)
		self.sigma = sigma
		self.pre_norm=pre_norm

	def __call__(self, sample):
		sources, targets = sample['sources'], sample['targets']

		for channel in range(len(sources)):
			if self.pre_norm:
				normalFilter= sitk.NormalizeImageFilter()
				sources[channel] = normalFilter.Execute(sources[channel])

			statisticsFilter = sitk.StatisticsImageFilter()
			statisticsFilter.Execute(sources[channel])

			intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
			intensityWindowingFilter.SetOutputMaximum(255)
			intensityWindowingFilter.SetOutputMinimum(0)
			intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+self.sigma*statisticsFilter.GetSigma());
			intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-self.sigma*statisticsFilter.GetSigma());

			sources[channel] = intensityWindowingFilter.Execute(sources[channel])

		for channel in range(len(targets)):
			if self.pre_norm:
				normalFilter= sitk.NormalizeImageFilter()
				targets[channel] = normalFilter.Execute(targets[channel])

			statisticsFilter = sitk.StatisticsImageFilter()
			statisticsFilter.Execute(targets[channel])

			intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
			intensityWindowingFilter.SetOutputMaximum(255)
			intensityWindowingFilter.SetOutputMinimum(0)
			intensityWindowingFilter.SetWindowMaximum(statisticsFilter.GetMean()+self.sigma*statisticsFilter.GetSigma());
			intensityWindowingFilter.SetWindowMinimum(statisticsFilter.GetMean()-self.sigma*statisticsFilter.GetSigma());

			targets[channel] = intensityWindowingFilter.Execute(targets[channel])

		return {'sources': sources, 'targets': targets}

# class ExtremumNormalization(object):
# 	"""
# 	Normalize an image by mapping intensity with intensity maximum and minimum
# 	"""

# 	def __init__(self, percent=0.05):
# 		self.name = 'ExtremumNormalization'
# 		assert isinstance(percent, float)
# 		self.percent = percent

# 	def __call__(self, sample):
# 		image, label = sample['image'], sample['label']

# 		for image_channel in range(len(image)):
# 			statisticsFilter = sitk.StatisticsImageFilter()
# 			statisticsFilter.Execute(image[image_channel])

# 			intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
# 			intensityWindowingFilter.SetOutputMaximum(255)
# 			intensityWindowingFilter.SetOutputMinimum(0)
# 			windowMax = (statisticsFilter.GetMaximum() - statisticsFilter.GetMinimum())*(1-self.percent) + statisticsFilter.GetMinimum()
# 			windowMin = (statisticsFilter.GetMaximum() - statisticsFilter.GetMinimum())*self.percent + statisticsFilter.GetMinimum()
# 			intensityWindowingFilter.SetWindowMaximum(windowMax);
# 			intensityWindowingFilter.SetWindowMinimum(windowMin);

# 			image[image_channel] = intensityWindowingFilter.Execute(image[image_channel])

# 		return {'image': image, 'label': label}

# class ManualNormalization(object):
# 	"""
# 	Normalize an image by mapping intensity with given max and min window level
# 	"""

# 	def __init__(self,windowMin, windowMax):
# 		self.name = 'ManualNormalization'
# 		assert isinstance(windowMax, (int,float))
# 		assert isinstance(windowMin, (int,float))
# 		self.windowMax = windowMax
# 		self.windowMin = windowMin

# 	def __call__(self, sample):
# 		image, label = sample['image'], sample['label']
# 		intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
# 		intensityWindowingFilter.SetOutputMaximum(255)
# 		intensityWindowingFilter.SetOutputMinimum(0)
# 		intensityWindowingFilter.SetWindowMaximum(self.windowMax);
# 		intensityWindowingFilter.SetWindowMinimum(self.windowMin);

# 		image = intensityWindowingFilter.Execute(image)

# 		return {'image': image, 'label': label}

# class Reorient(object):
# 	"""
# 	(Beta) Function to orient image in specific axes order
# 	The elements of the order array must be an permutation of the numbers from 0 to 2.
# 	"""

# 	def __init__(self, order):
# 		self.name = 'Reoreient'
# 		assert isinstance(order, (int, tuple))
# 		assert len(order) == 3
# 		self.order = order

# 	def __call__(self, sample):
# 		reorientFilter = sitk.PermuteAxesImageFilter()
# 		reorientFilter.SetOrder(self.order)
# 		image = reorientFilter.Execute(sample['image'])
# 		label = reorientFilter.Execute(sample['label'])

# 		return {'image': image, 'label': label}

# class Invert(object):
# 	"""
# 	Invert the image intensity from 0-255 
# 	"""

# 	def __init__(self):
# 		self.name = 'Invert'

# 	def __call__(self, sample):
# 		invertFilter = sitk.InvertIntensityImageFilter()
# 		image = invertFilter.Execute(sample['image'],255)
# 		label = sample['label']

# 		return {'image': image, 'label': label}

class Resample3D(object):
	"""
	Resample the volume in a sample to a given voxel size

	Args:
		voxel_size (float or tuple): Desired output size.
		If float, output volume is isotropic.
		If tuple, output voxel size is matched with voxel size
		Currently only support linear interpolation method
	"""

	def __init__(self, voxel_size):
		self.name = 'Resample 3D'

		assert isinstance(voxel_size, (int, float, tuple, list))
		if isinstance(voxel_size, (int, float)):
			self.voxel_size = (voxel_size, voxel_size, voxel_size)
		else:
			assert len(voxel_size) == 3
			self.voxel_size = voxel_size

	def __call__(self, sample):
		sources, targets = sample['sources'], sample['targets']

		resampler = sitk.ResampleImageFilter()
		for image_channel in range(len(sources)):
			old_spacing = sources[image_channel].GetSpacing()
			old_size = sources[image_channel].GetSize()

			new_spacing = self.voxel_size

			new_size = []
			for i in range(3):
				new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
			new_size = tuple(new_size)
			resampler.SetInterpolator(2)
			resampler.SetOutputSpacing(new_spacing)
			resampler.SetSize(new_size)

			# resample on image
			resampler.SetOutputOrigin(sources[image_channel].GetOrigin())
			resampler.SetOutputDirection(sources[image_channel].GetDirection())
			# print("Resampling image...")
			sources[image_channel] = resampler.Execute(sources[image_channel])

		for image_channel in range(len(targets)):
			old_spacing = targets[image_channel].GetSpacing()
			old_size = targets[image_channel].GetSize()

			new_spacing = self.voxel_size

			new_size = []
			for i in range(3):
				new_size.append(int(math.ceil(old_spacing[i]*old_size[i]/new_spacing[i])))
			new_size = tuple(new_size)
			resampler.SetInterpolator(2)
			resampler.SetOutputSpacing(new_spacing)
			resampler.SetSize(new_size)

			# resample on image
			resampler.SetOutputOrigin(targets[image_channel].GetOrigin())
			resampler.SetOutputDirection(targets[image_channel].GetDirection())
			# print("Resampling image...")
			targets[image_channel] = resampler.Execute(targets[image_channel])

		return {'sources': sources, 'targets': targets}

class Padding3D(object):
	"""
	Add padding to the image if size is smaller than patch size

	Args:
		output_size (tuple or int): Desired output size. If int, a cubic volume is formed
	"""

	def __init__(self, output_size):
		self.name = 'Padding 3D'

		assert isinstance(output_size, (int, tuple, list))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

		assert all(i > 0 for i in list(self.output_size))

	def __call__(self,sample):
		sources, targets = sample['sources'], sample['targets']

		size_old = sources[0].GetSize()

		if (size_old[0] >= self.output_size[0]) and (size_old[1] >= self.output_size[1]) and (size_old[2] >= self.output_size[2]):
			return sample
		else:
			output_size = list(self.output_size)
			if size_old[0] > self.output_size[0]:
				output_size[0] = size_old[0]
			if size_old[1] > self.output_size[1]:
				output_size[1] = size_old[1]
			if size_old[2] > self.output_size[2]:
				output_size[2] = size_old[2]

			output_size = tuple(output_size)

			for image_channel in range(len(sources)):
				resampler = sitk.ResampleImageFilter()
				resampler.SetOutputSpacing(sources[image_channel].GetSpacing())
				resampler.SetSize(output_size)

				# resample on image
				resampler.SetInterpolator(2)
				resampler.SetOutputOrigin(sources[image_channel].GetOrigin())
				resampler.SetOutputDirection(sources[image_channel].GetDirection())
				sources[image_channel] = resampler.Execute(sources[image_channel])

			for image_channel in range(len(targets)):
				resampler = sitk.ResampleImageFilter()
				resampler.SetOutputSpacing(targets[image_channel].GetSpacing())
				resampler.SetSize(output_size)

				# resample on image
				resampler.SetInterpolator(2)
				resampler.SetOutputOrigin(targets[image_channel].GetOrigin())
				resampler.SetOutputDirection(targets[image_channel].GetDirection())
				targets[image_channel] = resampler.Execute(targets[image_channel])

			return {'sources': sources, 'targets': targets}

class RandomCrop3D(object):
	"""
	Crop the 3D image randomly in a sample. This is usually used for data augmentation.

	Args:
	output_size (tuple or int): Desired output size. If int, cubic crop is made.
	"""

	def __init__(self, output_size):
		self.name = 'Random Crop 3D'

		assert isinstance(output_size, (int, tuple, list))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size, output_size)
		else:
			assert len(output_size) == 3
			self.output_size = output_size

	def __call__(self,sample):
		sources, targets = sample['sources'], sample['targets']
		size_old = sources[0].GetSize()
		size_new = self.output_size

		contain_label = False

		roiFilter = sitk.RegionOfInterestImageFilter()
		roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

		if size_old[0] <= size_new[0]:
			start_i = 0
		else:
			start_i = np.random.randint(0, size_old[0]-size_new[0])

		if size_old[1] <= size_new[1]:
			start_j = 0
		else:
			start_j = np.random.randint(0, size_old[1]-size_new[1])

		if size_old[2] <= size_new[2]:
			start_k = 0
		else:
			start_k = np.random.randint(0, size_old[2]-size_new[2])

		roiFilter.SetIndex([start_i,start_j,start_k])

		for channel in range(len(sources)):
			sources[channel] = roiFilter.Execute(sources[channel])
		for channel in range(len(targets)):
			targets[channel] = roiFilter.Execute(targets[channel])

		return {'sources': sources, 'targets': targets}

class RandomNoise(object):
	"""
	Randomly add noise to the source image in a sample. This is usually used for data augmentation.
	"""
	def __init__(self):
		self.name = 'Random Noise'

	def __call__(self, sample):
		self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
		self.noiseFilter.SetMean(0)
		self.noiseFilter.SetStandardDeviation(0.1)

		# print("Normalizing image...")
		sources, targets = sample['sources'], sample['targets']

		for image_channel in range(len(sources)):
			sources[image_channel] = self.noiseFilter.Execute(sources[image_channel])		

		return {'sources': sources, 'targets': targets}

# class ConfidenceCrop(object):
# 	"""
# 	Crop the image in a sample that is certain distance from individual labels center. 
# 	This is usually used for data augmentation with very small label volumes.
# 	The distance offset from connected label centroid is model by Gaussian distribution with mean zero and user input sigma (default to be 2.5)
# 	i.e. If n isolated labels are found, one of the label's centroid will be randomly selected, and the cropping zone will be offset by following scheme:
# 	s_i = np.random.normal(mu, sigma*crop_size/2), 1000)
# 	offset_i = random.choice(s_i)
# 	where i represents axis direction
# 	A higher sigma value will provide a higher offset

# 	Args:
# 	output_size (tuple or int): Desired output size. If int, cubic crop is made.
# 	sigma (float): Normalized standard deviation value.
# 	"""

# 	def __init__(self, output_size, sigma=2.5):
# 		self.name = 'Confidence Crop'

# 		assert isinstance(output_size, (int, tuple))
# 		if isinstance(output_size, int):
# 			self.output_size = (output_size, output_size, output_size)
# 		else:
# 			assert len(output_size) == 3
# 			self.output_size = output_size

# 		assert isinstance(sigma, (float, tuple))
# 		if isinstance(sigma, float) and sigma >= 0:
# 			self.sigma = (sigma,sigma,sigma)
# 		else:
# 			assert len(sigma) == 3
# 			self.sigma = sigma

# 	def __call__(self,sample):
# 		image, label = sample['image'], sample['label']
# 		size_new = self.output_size

# 		# guarantee label type to be integer
# 		castFilter = sitk.CastImageFilter()
# 		castFilter.SetOutputPixelType(sitk.sitkInt8)
# 		label = castFilter.Execute(label)

# 		ccFilter = sitk.ConnectedComponentImageFilter()
# 		labelCC = ccFilter.Execute(label)

# 		labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
# 		labelShapeFilter.Execute(labelCC)

# 		if labelShapeFilter.GetNumberOfLabels() == 0:
# 			# handle image without label
# 			selectedLabel = 0
# 			centroid = (int(self.output_size[0]/2), int(self.output_size[1]/2), int(self.output_size[2]/2))
# 		else:
# 			# randomly select of the label's centroid
# 			selectedLabel = random.randint(1,labelShapeFilter.GetNumberOfLabels())
# 			centroid = label.TransformPhysicalPointToIndex(labelShapeFilter.GetCentroid(selectedLabel))

# 		centroid = list(centroid)

# 		start = [-1,-1,-1] #placeholder for start point array
# 		end = [self.output_size[0]-1, self.output_size[1]-1,self.output_size[2]-1] #placeholder for start point array
# 		offset = [-1,-1,-1] #placeholder for start point array
# 		for i in range(3):
# 			# edge case
# 			if centroid[i] < (self.output_size[i]/2):
# 				centroid[i] = int(self.output_size[i]/2)
# 			elif (image.GetSize()[i]-centroid[i]) < (self.output_size[i]/2):
# 				centroid[i] = image.GetSize()[i] - int(self.output_size[i]/2) -1

# 			# get start point
# 			while ((start[i]<0) or (end[i]>(image.GetSize()[i]-1))):
# 				offset[i] = self.NormalOffset(self.output_size[i],self.sigma[i])
# 				start[i] = centroid[i] + offset[i] - int(self.output_size[i]/2)
# 				end[i] = start[i] + self.output_size[i] - 1

# 		roiFilter = sitk.RegionOfInterestImageFilter()
# 		roiFilter.SetSize(self.output_size)
# 		roiFilter.SetIndex(start)
# 		croppedImage = roiFilter.Execute(image)
# 		croppedLabel = roiFilter.Execute(label)

# 		return {'image': croppedImage, 'label': croppedLabel}

# 	def NormalOffset(self,size, sigma):
# 		s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
# 		return int(round(random.choice(s)))

# class ConfidenceCrop2(object):
# 	"""
# 	Crop the image in a sample that is certain distance from individual labels center. 
# 	This is usually used for data augmentation with very small label volumes.
# 	The distance offset from connected label bounding box center is a uniform random distribution within user specified range
# 	Regions containing label is considered to be positive while regions without label is negative. This distribution is governed by the user defined probability

# 	Args:
# 	output_size (tuple or int): Desired output size. If int, cubic crop is made.
# 	range (int): Bounding box random offset max value
# 	probability (float): Probability to get positive labels
# 	"""

# 	def __init__(self, output_size, rand_range=3,probability=0.5, random_empty_region=False):
# 		self.name = 'Confidence Crop 2'

# 		assert isinstance(output_size, (int, tuple))
# 		if isinstance(output_size, int):
# 			self.output_size = (output_size, output_size, output_size)
# 		else:
# 			assert len(output_size) == 3
# 			self.output_size = output_size

# 		assert isinstance(rand_range, (int,tuple))
# 		if isinstance(rand_range, int) and rand_range >= 0:
# 			self.rand_range = (rand_range,rand_range,rand_range)
# 		else:
# 			assert len(rand_range) == 3
# 			self.rand_range = rand_range

# 		assert isinstance(probability, float)
# 		if probability >= 0 and probability <= 1:
# 			self.probability = probability

# 		assert isinstance(random_empty_region, bool)
# 		self.random_empty_region=random_empty_region

# 	def __call__(self,sample):
# 		image, label = sample['image'], sample['label']
# 		size_new = self.output_size

# 		# guarantee label type to be integer
# 		castFilter = sitk.CastImageFilter()
# 		castFilter.SetOutputPixelType(sitk.sitkInt16)
# 		label = castFilter.Execute(label)

# 		# choose whether positive or negative label to crop
# 		zerosList = [0]*int(10*(1-self.probability))
# 		onesList = [1]*int(10*self.probability)
# 		choiceList = []
# 		choiceList.extend(zerosList)
# 		choiceList.extend(onesList)
# 		labelType = random.choice(choiceList)

# 		if labelType == 0:
# 			# randomly pick a region
# 			if self.random_empty_region:
# 				image, label = self.RandomEmptyRegion(image,label)
# 			else:
# 				image, label = self.RandomRegion(image,label)
# 		else:
# 			# get the number of labels
# 			ccFilter = sitk.ConnectedComponentImageFilter()
# 			labelCC = ccFilter.Execute(label)
# 			labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
# 			labelShapeFilter.Execute(labelCC)

# 			if labelShapeFilter.GetNumberOfLabels() == 0:
# 				if self.random_empty_region:
# 					image, label = self.RandomEmptyRegion(image,label)
# 				else:
# 					image, label = self.RandomRegion(image,label)
# 			else:
# 				selectedLabel = random.choice(range(0,labelShapeFilter.GetNumberOfLabels())) + 1 
# 				selectedBbox = labelShapeFilter.GetBoundingBox(selectedLabel)
# 				index = [0,0,0]
# 				for i in range(3):
# 					index[i] = selectedBbox[i] + int(selectedBbox[i+3]/2) - int(self.output_size[i]/2) + random.choice(range(-1*self.rand_range[i],self.rand_range[i]+1))
# 					if image[0].GetSize()[i] - index[i] - 1 < self.output_size[i]:
# 						index[i] = image[0].GetSize()[i] - self.output_size[i] - 1
# 					if index[i]<0:
# 						index[i] = 0

# 				roiFilter = sitk.RegionOfInterestImageFilter()
# 				roiFilter.SetSize(self.output_size)
# 				roiFilter.SetIndex(index)

# 				for image_channel in range(len(image)):
# 					image[image_channel] = roiFilter.Execute(image[image_channel])
# 				label = roiFilter.Execute(label)

# 		return {'image': image, 'label': label}

# 	def RandomEmptyRegion(self,image, label):
# 		index = [0,0,0]
# 		contain_label = False
# 		while not contain_label:
# 			for i in range(3):
# 				index[i] = random.choice(range(0,label.GetSize()[i]-self.output_size[i]-1))
# 			roiFilter = sitk.RegionOfInterestImageFilter()
# 			roiFilter.SetSize(self.output_size)
# 			roiFilter.SetIndex(index)
# 			label_ = roiFilter.Execute(label)
# 			statFilter = sitk.StatisticsImageFilter()
# 			statFilter.Execute(label_)

# 			if statFilter.GetSum() < 1:
# 				for image_channel in range(len(image)):
# 					label = label_
# 					image[image_channel] = roiFilter.Execute(image[image_channel])
# 				contain_label = True
# 				break
# 		return image,label

# 	def RandomRegion(self,image, label):
# 		index = [0,0,0]
# 		for i in range(3):
# 			index[i] = random.choice(range(0,label.GetSize()[i]-self.output_size[i]-1))
# 		roiFilter = sitk.RegionOfInterestImageFilter()
# 		roiFilter.SetSize(self.output_size)
# 		roiFilter.SetIndex(index)
# 		label = roiFilter.Execute(label)

# 		for image_channel in range(len(image)):
# 			image[image_channel] = roiFilter.Execute(image[image_channel])
			
# 		return image,label


# class BSplineDeformation(object):
# 	"""
# 	Image deformation with a sparse set of control points to control a free form deformation.
# 	Details can be found here: 
# 	https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf
# 	https://itk.org/Doxygen/html/classitk_1_1BSplineTransform.html

# 	Args:
# 		randomness (int,float): BSpline deformation scaling factor, default is 10.
# 	"""

# 	def __init__(self, randomness=10):
# 		self.name = 'BSpline Deformation'

# 		assert isinstance(randomness, (int,float))
# 		if randomness > 0:
# 			self.randomness = randomness
# 		else:
# 			raise RuntimeError('Randomness should be non zero values')

# 	def __call__(self,sample):
# 		image, label = sample['image'], sample['label']
# 		spline_order = 3
# 		domain_physical_dimensions = [image.GetSize()[0]*image.GetSpacing()[0],image.GetSize()[1]*image.GetSpacing()[1],image.GetSize()[2]*image.GetSpacing()[2]]

# 		bspline = sitk.BSplineTransform(3, spline_order)
# 		bspline.SetTransformDomainOrigin(image.GetOrigin())
# 		bspline.SetTransformDomainDirection(image.GetDirection())
# 		bspline.SetTransformDomainPhysicalDimensions(domain_physical_dimensions)
# 		bspline.SetTransformDomainMeshSize((10,10,10))

# 		# Random displacement of the control points.
# 		originalControlPointDisplacements = np.random.random(len(bspline.GetParameters()))*self.randomness
# 		bspline.SetParameters(originalControlPointDisplacements)

# 		image = sitk.Resample(image, bspline)
# 		label = sitk.Resample(label, bspline)
# 		return {'image': image, 'label': label}

# 	def NormalOffset(self,size, sigma):
# 		s = np.random.normal(0, size*sigma/2, 100) # 100 sample is good enough
# 		return int(round(random.choice(s)))