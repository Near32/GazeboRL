from GazeboRL import GazeboRL, Swarm1GazeboRL#, init_roscore
import time
import rospy
from Agent1 import NN, INPUT_SHAPE_R
import cv2
import numpy as np
from cv_bridge import CvBridge
bridge = CvBridge()

def ros2np(img) :
	return bridge.imgmsg_to_cv2(img, "bgr8")


nbrinput = INPUT_SHAPE_R
nbroutput = 2
filepath_base = './logs/'
dropoutK = 0.5
batch_size = 1024 
lr = 5.0e-4
def initAgent():		
	model = NN( filepath_base,nbrinput=nbrinput,nbroutput=nbroutput,lr=lr,filepathin=None)
	model.init()
	return model


port = 11315
env = Swarm1GazeboRL(port=port)
port += 1
env.make()


env1 = Swarm1GazeboRL(port=port)
port += 1
env1.make()


agent = initAgent()
print('\n\nwait for 5 sec...\n\n')
time.sleep(5)

env.setPause(False)
env1.setPause(False)

env.reset()
env1.reset()

env.setPause(False)
env1.setPause(False)


action = [0.0,0.0]
action1 = [0.0,0.0]

meantime = 0.0
i=10
while i :
	output = env.step(action)
	output1 = env1.step(action1)
	rospy.loginfo(output[1:])
	if output[0] is not None :
		for topic in output[0].keys() :
			if 'OMNIVIEW' in topic :
				img = np.array(ros2np(output[0][topic]))
				#cv2.imshow('image',img)
				#cv2.waitKey(1)
				start = time.time()
				try :
					action = agent.inference(x=img)[0][0]
				except Exception as e :
					rospy.loginfo('error occurred..'+str(e))
					action = [0.0,0.0]
				elapsed = time.time()-start
				meantime+=elapsed
				rospy.loginfo('elt:'+str(elapsed)+'::action : ')
				rospy.loginfo(action)
	
	if output1[0] is not None :
		for topic in output1[0].keys() :
			if 'OMNIVIEW' in topic :
				img = np.array(ros2np(output1[0][topic]))
				#cv2.imshow('image',img)
				#cv2.waitKey(1)
				start = time.time()
				try :
					action1 = agent.inference(x=img)[0][0]
				except Exception as e :
					rospy.loginfo('error occurred..'+str(e))
					action1 = [0.0,0.0]
				elapsed = time.time()-start
				meantime+=elapsed
				rospy.loginfo('elt1:'+str(elapsed)+'::action1 : ')
				rospy.loginfo(action1)
	
	i-=1
	time.sleep(0.1)

rospy.loginfo('MEAN COMPUTATION TIME :'+str(meantime/1000.0))
#0.011085 seconds == 90Hz

#cv2.destroyAllWindows()

env.close()

