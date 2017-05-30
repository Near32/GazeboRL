import cv2, os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

imgh = 240
imgw = 1280
imgch = 3


imgh_r = 240
imgw_r = 1280

#imgw_r = 640


INPUT_SHAPE = (imgh,imgw,imgch)

INPUT_SHAPE_R = [imgh_r,imgw_r,imgch]

def resize(image, imghr, imgwr) :
	return cv2.resize(image, (imgwr,imghr), cv2.INTER_AREA)
	
def rgb2yuv(image) :
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	
def preprocess(image, imghr, imgwr) :
	image = resize(image, imghr, imgwr)
	image = rgb2yuv(image)
	image = np.array(image)*1.0/127.5
	image -= 1.0
	#plt.imshow(image)
	#plt.show()
	return image
	
def random_shadow(image) :
	x1, y1 = imgw * np.random.rand(), 0
	x2, y2 = imgw * np.random.rand(), imgh
	xm, ym = np.mgrid[0:imgh,0:imgw]
	
	mask = np.zeros_like(image[:,:,1])
	mask[(ym-y1)*(x2-x1)-(y2-y1)*(xm-x1)>0] = 1
	
	cond = mask==np.random.randint(2)
	s_ratio = np.random.uniform(low=0.2, high=0.5)
	
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	hls[:,:,1][cond] = hls[:,:,1][cond]*s_ratio
	
	return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
	
def random_brightness(image):
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(image) :
	image = random_shadow(image)
	image = random_brightness(image)
	return image
	
def batch_generator( input_path, output_path, batch_size, is_training) :
	images = np.empty( [batch_size,  imgh, imgw, imgch])
	outputs = np.empty( [batch_size,2] )
	while True :
		i = 0
		for index in np.random.permutation(input_path.shape[0]) :
			output = output_path[index]
			image = input_path[index]
			if is_training and np.random.rand() < 0.6 :
				image = augment( image )
			images[i] = np.array( preprocess(image, imgh, imgw) ).reshape( (imgh,imgw,imgch) )
			outputs[i] = output
			i+=1
			if i==batch_size :
				break
		yield images,outputs


class Dataset :
	def __init__(self, path, test_size = 0.2, imgh=imgh_r, imgw=imgw_r, imgd=3) :
		self.imgh = imgh
		self.imgw = imgw
		self.imgd = imgd
		self.path = path
		self.codata = np.load(path+'/dataset1Robot.CmdsOdoms.npz')
		self.cmds = self.codata['cmds']
		self.outputSize = self.cmds.shape[1]
		self.odoms = self.codata['odoms']
		self.nbrSamples = len(self.cmds)
		self.indexes = range(self.nbrSamples)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( self.indexes, self.cmds, test_size=test_size, random_state=42)
		self.trainsize = len(self.x_train)
		self.testsize = len(self.x_test)
		
	def batch_generator(self, batch_size, is_training) :
		images = np.empty( [batch_size, self.imgh*self.imgw*self.imgd] )
		outputs = np.empty( [batch_size, self.outputSize] )
		while True :
			i = 0
			nbrSample = self.trainsize
			if is_training == False :
				nbrSample = self.testsize
			for index in np.random.permutation(nbrSample) :
				output = None
				image = None
				if is_training :
					output = self.y_train[index]
					image = cv2.imread( self.path+'/'+str(self.x_train[index])+'.png')
					if np.random.rand() < 0.75 :
						image = augment(image)
				else :
					output = self.y_test[index]
					image = cv2.imread( self.path+'/'+str(self.x_test[index])+'.png')
				
				images[i] = np.array( preprocess(image, self.imgh, self.imgw) ).reshape( (-1,self.imgh*self.imgw*self.imgd) )
				outputs[i] = output
				
				i+=1
				if i==batch_size :
					images = np.array(images)
					outputs = np.array(outputs)
					break
			#yield (images,outputs)
			return images,outputs
			
	def getTrainSize(self) :
		return self.trainsize
		
	def getTestSize(self) :
		return self.testsize
					
