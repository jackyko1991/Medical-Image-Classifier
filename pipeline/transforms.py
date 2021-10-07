from pipeline import NiftiDataset
import yaml

class SpacingConstructor:
    def __init__(self, spacing=[]):
        self.spacing = spacing

    def __call__(self, loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode):
        return self.spacing

class PatchShapeConstructor:
    def __init__(self, patch_shape=[]):
        self.patch_shape = patch_shape

    def __call__(self, loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode):
        return self.patch_shape

def get_loader(spacing, patch_shape):
    """Add constructors to PyYAML loader."""
    loader = yaml.SafeLoader
    spacing_constructor = SpacingConstructor(spacing)
    patch_shape_constructor = PatchShapeConstructor(patch_shape)

    loader.add_constructor("!spacing", spacing_constructor)
    loader.add_constructor("!patch_shape", patch_shape_constructor)
    return loader

def transforms(spacing, patch_shape, pipeline_yaml, phase="train", dim="2D"):
	assert phase in ["train","test","predict"], "Preprocessing transfrom phase can only be train, test or predict"
	assert dim in ["2D", "3D"], "Preprocessing transfrom dimension can only be 2D or 3D"

	with open(pipeline_yaml) as f:
		pipeline_ = yaml.load(f,Loader=get_loader(spacing,patch_shape))

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
	elif len(spacing) == 3 and len(patch_shape) == 3:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="train",dim="3D"))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")

def test_transforms(spacing,patch_shape,pipeline_yaml):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="test",dim="2D"))
	elif len(spacing) == 3 and len(patch_shape) == 3:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="test",dim="3D"))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")

def predict_transforms(spacing,patch_shape,pipeline_yaml):
	if len(spacing) == 2 and len(patch_shape) == 2:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="predict",dim="2D"))
	elif len(spacing) == 3 and len(patch_shape) == 3:
		return(transforms(spacing,patch_shape,pipeline_yaml,phase="predict",dim="3D"))
	else:
		raise ValueError("spacing and patch_shape should be same size of 2 or 3")