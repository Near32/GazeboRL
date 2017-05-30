import numpy as np
import cv2

path = './datasets/dataset1Robot.ImagesCmdsOdoms.npz'
#path = './datasets/dataset1Robot.Images.npz'
data = np.load(path)

images = data['images']
cmds = data['cmds']
odoms = data['odoms']

pathout = './images'
np.savez(pathout+'/dataset1Robot.CmdsOdoms', cmds=cmds, odoms = odoms)

for i in range(images.shape[0]) :
	cv2.imwrite( pathout+'/'+str(i)+'.png', images[i])
	

