import NiftiDataset
import yaml
import SimpleITK as sitk
import os

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
    loader = yaml.Loader
    spacing_constructor = SpacingConstructor(spacing)
    patch_shape_constructor = PatchShapeConstructor(patch_shape)

    loader.add_constructor("!spacing", spacing_constructor)
    loader.add_constructor("!patch_shape", patch_shape_constructor)
    # loader.add_constructor('!composite_transform', composite_transform_constructor)
    return loader

def load_pipeline(spacing, patch_shape, pipeline_yaml, verbose=True):
    with open(pipeline_yaml) as f:
        pipeline_ = yaml.load(f, Loader=get_loader(spacing,patch_shape))

    # start preprocessing
    # print(pipeline_["preprocess"])

    train_transform_3d = []
    train_transform_2d = []
    test_transform_3d = []
    test_transform_2d = []

    if pipeline_["preprocess"]["train"]["3D"] is not None:
        for transform in pipeline_["preprocess"]["train"]["3D"]:

            if transform["name"] == "CompositeTransform":
                method_channel_list = []
                for method in transform["variables"]["method_channel_list"]:
                    tfm_cls_ch = getattr(NiftiDataset, method["name"])(*[],**method["variables"])
                    method_channel_list.append((tfm_cls_ch, method["channel"]))
                tfm_cls = NiftiDataset.CompositeTransform(method_channel_list)
            else:
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

    if verbose:
        print("training transform 3d:")
        [print(tfm.__dict__) for tfm in train_transform_3d]

        print("training transform 2d:")
        [print(tfm.__dict__) for tfm in train_transform_2d]

        print("testing transform 3d:")
        [print(tfm.__dict__) for tfm in test_transform_3d]

        print("testing transform 2d:")
        [print(tfm.__dict__) for tfm in test_transform_2d]

    return {"train_transform_3d": train_transform_3d, "train_transform_2d": train_transform_2d, "test_transform_3d": test_transform_3d, "test_transform_2d": test_transform_2d}

def main():
    # load the yaml
    spacing = [1, 1, 1]
    patch_shape = [32,32,200]

    transforms = load_pipeline(spacing, patch_shape,"./pipeline/pipeline_carotid_cfd_mag.yaml")

    # test the trasform
    image_dir = "/mnt/DIIR-JK-NAS/data/carotid/data_kfold/fold_0/test/002_left"
    image_output_dir = "/mnt/DIIR-JK-NAS/data/carotid/data_kfold/fold_0/test/002_left/transformed_output"
    image_files = ["p.nii.gz","U_mag.nii.gz"]
    sample = {"images": []}

    for image_file in image_files:
        sample["images"].append(sitk.ReadImage(os.path.join(image_dir,image_file)))

    for tfm in transforms["train_transform_3d"]:
        sample = tfm(sample)

    # export the output
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for image_file, image in zip(image_files, sample["images"]):
        sitk.WriteImage(image,os.path.join(image_output_dir,image_file))


if __name__ == "__main__":
	main()