import os
import SimpleITK as sitk
from tqdm import tqdm

def readDicom(folder):
	print( "Reading Dicom directory:", folder )
	reader = sitk.ImageSeriesReader()

	dicom_names = reader.GetGDCMSeriesFileNames(folder)
	reader.SetFileNames(dicom_names)

	image = reader.Execute()
	return image

def dcm2nii(srcFolder, tgtFilename):
	# check if srcFolder is empty
	if len(os.listdir(srcFolder)) == 0:
		print("Dicom folder is empty:",srcFolder)
		return

	image = readDicom(srcFolder)
	sitk.WriteImage( image, tgtFilename )

def main():
	dcm_dir = "../data/dicom_sorted_neg"
	nii_dir = "../data/nii"
	nii_name = "image.nii.gz"

	if not os.path.isdir(nii_dir):
		os.makedirs(nii_dir,exist_ok=True)

	for case in tqdm(os.listdir(dcm_dir)):
		os.makedirs(os.path.join(nii_dir,case))
		dcm2nii(os.path.join(dcm_dir,case),os.path.join(nii_dir,case,nii_name))

if __name__=="__main__":
	main()