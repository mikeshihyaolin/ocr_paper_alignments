'''
Shih-Yao (Mike) Lin
2020-05-15
'''
from utils import *
import glob

class DataSynthesis:

	def __init__(self, threshold=127):
		self.threshold = threshold

	def run(self, foreground_dir, background_dir):

		foreground = cv2.imread(foreground_dir)
		background = cv2.imread(background_dir)
		gray = cv2.imread(foreground_dir,0)
		height, width = gray.shape

		''' resize the background image'''
		background = cv2.resize(background,(width,height), interpolation = cv2.INTER_AREA)

		'''get the mask of foreground and background'''
		ret,background_mask = cv2.threshold(gray,self.threshold,255,cv2.THRESH_BINARY)
		ret,foreground_mask = cv2.threshold(gray,self.threshold,255,cv2.THRESH_BINARY_INV)

		'''get the fusion image'''
		background = cv2.bitwise_and(background,background, mask = background_mask)
		foreground = cv2.bitwise_and(foreground,foreground, mask = foreground_mask)
		fusion = cv2.bitwise_or(background,foreground)

		return fusion, foreground_mask

def main():

	obj = DataSynthesis(200)

	'''load background imgs dir'''
	background_imgs = "./background_imgs/*"
	bg_img_list = sorted(glob.glob(background_imgs))

	'''load foreground imgs dir'''
	foreground_imgs = "./ocr_samples/1/*"
	fg_img_list = sorted(glob.glob(foreground_imgs))

	''' set output dir'''
	# mask_dir = "./mike_dataset/mike_mask"
	# reset(mask_dir)
	fusion_dir = "./mike_dataset/synthetic_data"
	reset(fusion_dir)

	''' prcoess '''
	for bg in bg_img_list:
		print(bg)
		bg_name = bg.split("/")[-1]

		for fg in fg_img_list:

			fg_name = fg.split("/")[-1]
			print(fg_name)

			fusion, foreground_mask = obj.run(fg, bg)

			cv2.imwrite(fusion_dir+"/"+bg_name+"_"+fg_name, fusion)
			# cv2.imwrite(mask_dir+"/"+fg_name, foreground_mask)

if __name__=="__main__":
	main()


