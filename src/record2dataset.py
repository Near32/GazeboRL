import numpy as np
import cv2
from cv_bridge import CvBridge

data_out = "./dataset1Robot.ImagesCmds.npz"
data_dir = "./dataset1Robot.npz"
data = np.load(data_dir)

images = data['images']
cmds = data['cmds']

bim = list()
bcm = list()

bridge = CvBridge()

for im in images :
	bim.append( bridge.imgmsg_to_cv2(im, "bgr8") )

for cm in cmds :
	bcm.append( (cm.linear.x, cm.angular.z) )

del images
del cmds
del data
	
bcm = np.array(bcm)
bim = np.array(bim)

print(bcm.shape, bim.shape)

np.savez(data_out, images=bim, cmds=bcm)


