import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, NewPage, Center
from pylatex.utils import italic, NoEscape
import SimpleITK as sitk
import numpy as np

class Report(object):
	def __init__(self,
		images,
		result=[0],
		class_names=[""]
		):
		print("create report object")

		self.images = images
		self.class_names = class_names
		self.result = result

		document_options = ['a4paper']
		geometry_options = {"tmargin":"1in","lmargin":"1in","bmargin":"1in","rmargin":"1in"}

		self.doc = Document(
			document_options=document_options,
			geometry_options=geometry_options)
		
		with self.doc.create(Section('CNN ICAD Stroke Classification Report',numbering=False)):
			with self.doc.create(Subsection('Maximum Intensity Projections',numbering=False)):
				fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True,  sharey=True, figsize=(12,4))

				with self.doc.create(Figure(position='h!')) as plot:
					for image_channel in range(len(images)):
						# convert image to numpy array
						image = sitk.GetArrayFromImage(images[image_channel])
						image_np = np.asarray(image,np.float32)

						ax[image_channel].set_xticklabels([])
						ax[image_channel].set_yticklabels([])
						ax[image_channel].set_xticks([])
						ax[image_channel].set_yticks([])
						ax[image_channel].imshow(np.flipud(image_np),cmap='gray')

					plt.subplots_adjust(wspace=0, hspace=0,top = 1, bottom = 0, right =1, left = 0)
					plot.add_plot(dpi=300,facecolor='b')
					plt.clf()

				# # self.doc.append(NewPage())
				with self.doc.create(Subsection('Results',numbering=False)) as results:
					with results.create(Center()) as centered:
						with centered.create(Tabular('c c')) as table:
							table.add_hline()
							table.add_row(('Disease Type','Probability'))
							table.add_hline()
							for i in range(len(self.class_names)):
								table.add_row((self.class_names[i], "{0:.3g}%".format(self.result[0][i]*100)))
							table.add_hline()
		
	def WritePdf(self,path):
		self.doc.generate_pdf(filepath=path)

	def WriteTex(self,path):
		self.doc.generate_tex(filepath=path)