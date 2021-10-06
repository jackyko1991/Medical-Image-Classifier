import NiftiDataset
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

def main():
    # load the yaml
    spacing = [0.75, 0.75]
    patch_shape = [256, 256]

    with open("./pipeline/pipeline.yaml") as f:
        pipeline_ = yaml.load(f, Loader=get_loader(spacing,patch_shape))

    # start preprocessing
    print(pipeline_["preprocess"])

    train_transform_3d = []
    train_transform_2d = []
    test_transform_3d = []
    test_transform_2d = []


    if pipeline_["preprocess"]["train"]["3D"] is not None:
        for transform in pipeline_["preprocess"]["train"]["3D"]:
            tfm_cls = getattr(NiftiDataset,transform["name"])(*[],**transform["variables"])
            train_transform_3d.append(tfm_cls)

    if pipeline_["preprocess"]["train"]["2D"] is not None:
        for transform in pipeline_["preprocess"]["train"]["2D"]:
            tfm_cls = getattr(NiftiDataset,transform["name"])(*[],**transform["variables"])
            train_transform_2d.append(tfm_cls)

    if pipeline_["preprocess"]["test"]["3D"] is not None:
    	for transform in pipeline_["preprocess"]["test"]["3D"]:
    		tfm_cls = getattr(NiftiDataset,transform["name"])(*[],**transform["variables"])
    		test_transform_3d.append(tfm_cls)

    if pipeline_["preprocess"]["test"]["2D"] is not None:
    	for transform in pipeline_["preprocess"]["test"]["2D"]:
    		tfm_cls = getattr(NiftiDataset,transform["name"])(*[],**transform["variables"])
    		test_transform_2d.append(tfm_cls)

    print("training transform 3d:")
    [print(tfm.__dict__) for tfm in train_transform_3d]

    print("training transform 2d:")
    [print(tfm.__dict__) for tfm in train_transform_2d]

    print("testing transform 3d:")
    [print(tfm.__dict__) for tfm in test_transform_3d]

    print("testing transform 2d:")
    [print(tfm.__dict__) for tfm in test_transform_2d]

if __name__ == "__main__":
	main()