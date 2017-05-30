import numpy as np
import rospy
import time

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge


pathout = './dataset1Robot.ImagesCmdsOdoms.npz'
continuer = True
bridge = CvBridge()


buffimages = list()
images = list()

buffodom = list()
odoms = list()

buffcmd = list()
cmds = list()

def callbackODOMETRY(odom) :
	buffodom.append(odom)
		
def callbackOMNIVIEW(image) :
	buffimages.append( bridge.imgmsg_to_cv2(image, "bgr8") )

def callbackCMD(cmd) :
	buffcmd.append( (cmd.linear.x, cmd.angular.z) )



def shutdown() :
	continuer = False
	time.sleep(1)
	global cmds
	cmds = np.array(cmds)
	
	np.savez(pathout, images=images, odoms=odoms, cmds=cmds)
	
	
rospy.init_node('recorder_node', anonymous=False)
rospy.on_shutdown(shutdown)

subODOM = rospy.Subscriber( '/robot_model_teleop_0/odom_diffdrive', Odometry, callbackODOMETRY )
subOMNI = rospy.Subscriber( '/robot_model_teleop_0/OMNIVIEW', Image, callbackOMNIVIEW )
subCMD = rospy.Subscriber( '/robot_model_teleop_0/cmd_vel', Twist, callbackCMD )


rec_rate = rospy.Rate(50)

tcmd = None
todom = None
timg = None
push = False
while continuer :
	
	if len(buffimages) :
		#images.append(buffimages[-1])
		timg = buffimages[-1]
		del buffimages[:]
		push = True
		
	if len(buffodom) :
		#odoms.append(buffodom[-1])
		todom = buffodom[-1]
		del buffodom[:]
		push = True
		
	if len(buffcmd) :
		#cmds.append(buffcmd[-1])
		tcmd = buffcmd[-1]
		del buffcmd[:]
		push = True
		
	if push and tcmd is not None and timg is not None and tcmd is not None :
		push = False
		images.append(timg)
		odoms.append(todom)
		cmds.append(tcmd)
		
		
	rec_rate.sleep()

