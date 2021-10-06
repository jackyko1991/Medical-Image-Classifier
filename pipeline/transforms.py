from pipeline import NiftiDataset
import yaml

# def train_transforms2D(spacing,patch_shape,pipeline_yaml):
# 	transforms2D = [
# 		NiftiDataset.RandomNoise(2.5),
# 		NiftiDataset.StatisticalNormalization(2.5),
# 		NiftiDataset.Resample2D(spacing),
# 		NiftiDataset.Padding2D(patch_shape),
# 		NiftiDataset.RandomCrop2D(patch_shape),
# 		NiftiDataset.RandomRotate2D(),
		
# 	]

# 	return transforms2D

# def test_transforms2D(spacing,patch_shape,pipeline_yaml):
# 	transforms2D = [
# 		NiftiDataset.RandomNoise(2.5),
# 		NiftiDataset.StatisticalNormalization(2.5),
# 		NiftiDataset.Resample2D(spacing),
# 		NiftiDataset.Padding2D(patch_shape),
# 		NiftiDataset.RandomCrop2D(patch_shape),
# 		NiftiDataset.RandomRotate2D(),
# 	]

# 	return transforms2D

# def train_transforms3D(spacing, patch_shape,pipeline_yaml):
# 	transforms = [
# 		NiftiDataset.RandomNoise(0.05),
# 		# NiftiDataset.Normalization(),
# 		# NiftiDataset.ManualNormalization([[0,800],[0,50]]),
# 		NiftiDataset.ManualNormalization([[0,800],[0,50],[0,5],[0,100000],[0,0.5]]),
# 		# NiftiDataset.ManualNormalization([[0,50],[0,5],[0,100000],[0,0.5]]),
# 		# NiftiDataset.ManualNormalization([[0,800]]),
# 		# NiftiDataset.StatisticalNormalization(2.5),
# 		NiftiDataset.Resample3D(spacing),
# 		NiftiDataset.Padding3D(patch_shape),
# 		NiftiDataset.RandomRotate3D(max_angle_X=5,max_angle_Y=5,max_angle_Z=5),
# 		NiftiDataset.RandomCrop3D(patch_shape),
# 		]

# 	return transforms
	
# def test_transforms3D(spacing, patch_shape,pipeline_yaml):
# 	transforms = [
# 		NiftiDataset.RandomNoise(0.05),
# 		# NiftiDataset.Normalization(),
# 		# NiftiDataset.ManualNormalization([[0,800],[0,50]]),
# 		NiftiDataset.ManualNormalization([[0,800],[0,50],[0,5],[0,100000],[0,0.5]]),
# 		# NiftiDataset.ManualNormalization([[0,50],[0,5],[0,100000],[0,0.5]]),
# 		# NiftiDataset.ManualNormalization([[0,800]]),
# 		#NiftiDataset.StatisticalNormalization(2.5),
# 		NiftiDataset.Resample3D(spacing),
# 		NiftiDataset.Padding3D(patch_shape),
# 		NiftiDataset.RandomCrop3D(patch_shape),
# 		]

# 	return transforms

# def predict_transforms2D(spacing,patch_shape,pipeline_yaml):
# 	transforms = [
# 		NiftiDataset.StatisticalNormalization(2.5),
# 		NiftiDataset.Resample2D(spacing),
# 		NiftiDataset.Padding2D(patch_shape),
# 		NiftiDataset.CropCenter2D(patch_shape),
# 	]

# 	return transforms

# def predict_transforms3D(spacing,patch_shape,pipeline_yaml):
# 	transforms = [
# 		NiftiDataset.StatisticalNormalization(2.5),
# 		NiftiDataset.Resample3D(spacing),
# 		NiftiDataset.Padding3D(patch_shape),
# 		NiftiDataset.CropCenter3D(patch_shape),
# 	]

# 	return transforms

def transforms(spacing, patch_shape, pipeline_yaml, phase="train", dim="2D"):
	assert phase in ["train","test","predict"], "Preprocessing transfrom phase can only be train, test or predict"
	assert dim in ["2D", "3D"], "Preprocessing transfrom dimension can only be 2D or 3D"

	with open(pipeline_yaml) as f:
		pipeline_ = yaml.load(f)

	transforms = []

	if pipeline_["preprocess"][phase][dim] is not None:
		for transform in pipeline_["preprocess"][phase][dim]:
			try:
				tfm = getattr(NiftiDataset,transform["name"])(*[],**transform["variables"])
			except:
				tfm = getattr(NiftiDataset,transform["name"])()
			transforms.append(tfm)

	return transforms

def train_transforms(spacing,patch_shape,pipeline_yaml):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="train",dim="2D"))
	elif  len(spacing) == 3 and len(patch_shape) == 3:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="train",dim="3D"))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")

def test_transforms(spacing,patch_shape,pipeline_yaml):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="test",dim="2D"))
	elif  len(spacing) == 3 and len(patch_shape) == 3:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="test",dim="3D"))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")

def predict_transforms(spacing,patch_shape,pipeline_yaml):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="predict",dim="2D"))
	elif  len(spacing) == 3 and len(patch_shape) == 3:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="predict",dim="3D"))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")