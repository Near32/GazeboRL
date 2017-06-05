from GazeboRL import GazeboRL, Swarm1GazeboRL, init_roscore
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
	
#commands = {'launch': None}
#launchCom = []

#launchCom.append('rosrun gazebo_ros spawn_model -file /home/kevin/rosbuild_ws/sandbox/GazeboRL/object.urdf -urdf -z 1 -model my_object')
#launchCom.append('roslaunch gazebo_ros empty_world.launch paused:=true use_sim_time:=false gui:=true throttled:=false headless:=false debug:=false')
#launchCom.append('rosrun gazebo_ros spawn_model -file /home/kevin/rosbuild_ws/sandbox/OPUSim/models/target_model/model.sdf -sdf -z 1 -x 0 -y 0 -model my_target')

#launchCom.append('roslaunch OPUSim robot1swarm.launch')
#launchCom.append('roslaunch OPUSim robot2swarm.launch')

#commands['launch'] = launchCom

#env = GazeboRL(commands)

#init_roscore()

env = Swarm1GazeboRL()
env.make()
agent = initAgent()
print('\n\nwait for 5 sec...\n\n')
time.sleep(5)
env.setPause(False)

env.reset()

action = [0.0,0.0]

meantime = 0.0
i=1000
while i :
	output = env.step(action)
	rospy.loginfo(output[1:])
	if output[0] is not None :
		for topic in output[0].keys() :
			if 'OMNIVIEW' in topic :
				img = np.array(ros2np(output[0][topic]))
				cv2.imshow('image',img)
				cv2.waitKey(1)
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
	i-=1
	time.sleep(0.1)

rospy.loginfo('MEAN COMPUTATION TIME :'+str(meantime/1000.0))
#0.011085 seconds == 90Hz

cv2.destroyAllWindows()

env.close()

