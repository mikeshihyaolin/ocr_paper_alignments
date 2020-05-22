'''
Shih-Yao (Mike) Lin
2020-05-21
'''
from utils import *
import glob
import json

class DataLabeling:

	def __init__(self):
		''' label the four corners '''
		self.prev_x, self.prev_y = -1,-1
		self.first_x,self.first_y = -1,-1
		self.click_time = 0
		self.img = None
		self.label_map = None
		self.corners = {}

	def load_img(self, img_dir):
		self.img = cv2.imread(img_dir)
		self.height, self.width, _ = self.img.shape
		print(self.height, self.width)
		self.corners["height"] = self.height
		self.corners["width"] = self.width
		# create a label map
		self.label_map = np.zeros((self.height, self.width,3),np.uint8)

	# mouse callback function
	def draw_circle(self,event,x,y,flags,param):

		if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(self.img,(x,y),10,(0,255,0),-1)

			if self.click_time == 0:
				self.first_x,self.first_y = x,y
			elif self.click_time >0 and self.click_time <3:
				cv2.line(self.img, (self.prev_x,self.prev_y), (x,y), (0,255,0), 5)
				cv2.line(self.label_map, (self.prev_x,self.prev_y), (x,y), (255,255,255), 5)
			else:
				cv2.line(self.img, (self.prev_x,self.prev_y), (x,y), (0,255,0), 5)
				cv2.line(self.img, (self.first_x,self.first_y), (x,y), (0,255,0), 5)
				cv2.line(self.label_map, (self.prev_x,self.prev_y), (x,y), (255,255,255), 5)
				cv2.line(self.label_map, (self.first_x,self.first_y), (x,y), (255,255,255), 5)
				
			self.prev_x,self.prev_y = x,y
			self.click_time += 1

			self.corners[self.click_time]=(x,y)
			print(self.click_time)

	def run_label(self, output_labels):
		cv2.namedWindow('image')
		cv2.setMouseCallback('image',self.draw_circle)

		while(1):

			# cv2.imshow('labels',self.label_map)
			cv2.imshow('image',self.img)

			if self.click_time ==4:
				cv2.waitKey(0)
				break
			k = cv2.waitKey(1) & 0xFF

			if k == 27:
				break

		cv2.destroyAllWindows()

		print(self.corners)

		with open(output_labels,"w") as output:
			json.dump(self.corners, output)

		self.click_time = 0
		return self.img, self.label_map,

def main():

	obj = DataLabeling()

	# '''example for labeling single data'''
	# file_dir = '../ocr_samples/1/tan19a-page-001.jpg'
	# file_name = file_dir.split("/")[-1]
	# output = "./"+file_name+".json"
	# obj.load_img(file_dir)
	# obj.run_label(output)

	''' example for labeling entire images'''
	foreground_imgs = "./ocr_samples/1/*"
	fg_img_list = sorted(glob.glob(foreground_imgs))

	output_folder = "./mike_dataset/labels"
	reset(output_folder)

	''' [removable] this is just for visualization'''
	tmp_path = "./mike_dataset/label_visualization"
	# reset(tmp_path)

	for f_path in fg_img_list:

		fg_name = f_path.split("/")[-1]
		output = output_folder +"/"+fg_name+".json"

		obj.load_img(f_path)
		img, label_img = obj.run_label(output)

		''' [removable]'''
		cv2.imwrite(tmp_path+"/img_"+fg_name+".jpg", img)
		cv2.imwrite(tmp_path+"/labelimg_"+fg_name+".jpg", label_img)


if __name__=="__main__":
	main()


