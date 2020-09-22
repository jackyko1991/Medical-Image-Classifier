from core import NiftiDataset

def train_transforms2D(spacing,patch_shape):
	transforms2D = [
		NiftiDataset.StatisticalNormalization(2.5),
		NiftiDataset.RandomRotate2D(),
		NiftiDataset.Resample2D(spacing),
		NiftiDataset.Padding2D(patch_shape),
		NiftiDataset.RandomCrop2D(patch_shape),
		NiftiDataset.RandomNoise()
	]

	return transforms2D

def test_transforms2D(spacing,patch_shape):

	return

def train_transforms3D(spacing, patch_shape):
	transforms = [
		# NiftiDataset.Normalization(),
		# NiftiDataset.ManualNormalization(0,75),
		NiftiDataset.StatisticalNormalization(2.5),
		NiftiDataset.Resample3D(spacing),
		NiftiDataset.Padding3D(patch_shape),
		NiftiDataset.RandomCrop3D(patch_shape),
		NiftiDataset.RandomNoise()
		]

	return transforms
	
def test_transforms3D(spacing, patch_shape):
	transforms = [
		# NiftiDataset.Normalization(),
		# NiftiDataset.ManualNormalization(0,75),
		NiftiDataset.StatisticalNormalization(2.5),
		NiftiDataset.Resample3D(spacing),
		NiftiDataset.Padding3D(patch_shape),
		NiftiDataset.RandomCrop3D(patch_shape),
		]

	return transforms

def train_transforms(spacing,patch_shape):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(train_transforms2D(spacing,patch_shape))
	elif  len(spacing) == 3 and len(patch_shape) == 3:
		return(train_transforms3D(spacing,patch_shape))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")

def test_transforms(spacing,patch_shape):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(test_transforms2D(spacing,patch_shape))
	elif  len(spacing) == 3 and len(patch_shape) == 3:
		return(test_transforms3D(spacing,patch_shape))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")