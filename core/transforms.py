from core import NiftiDataset

def train_transforms(spacing, patch_shape):
	transforms = [
		NiftiDataset.Normalization(),
		# NiftiDataset.ManualNormalization(0,75),
		NiftiDataset.Resample3D(spacing),
		NiftiDataset.Padding3D(patch_shape),
		NiftiDataset.RandomCrop3D(patch_shape),
		NiftiDataset.RandomNoise()
		]

	return transforms
	
def test_transforms(spacing, patch_shape):
	transforms = [
		NiftiDataset.Normalization(),
		# NiftiDataset.ManualNormalization(0,75),
		NiftiDataset.Resample3D(self.spacing),
		NiftiDataset.Padding3D(self.patch_shape),
		NiftiDataset.RandomCrop3D(self.patch_shape),
		]

	return transforms