#!/usr/bin/python

import numpy as np
import rospy
import tf
import time

from geometry_msgs.msg import Pose, Twist
import argparse

parser = argparse.ArgumentParser(description="Zero-order-holder node for RL framework.")
parser.add_argument('-topic', help='topic to hold.', dest='topic', type=str, default='/robot_model_teleop_0/cmd_vel_controlLaw' )
args, unknown = parser.parse_known_args()

print(args)
print(unknown)

buffstate = list()
def callbackState(state) :
	buffstate.append(state)
	


continuer = True
def shuttingdown() :
	continuer = False

rospy.init_node('ZeroOrderHolder_node', anonymous=False)
rospy.on_shutdown(shuttingdown)

subState = rospy.Subscriber( args.topic, Twist, callbackState )
pubR = rospy.Publisher('/RL/zoh'+args.topic, Twist,queue_size=10)

freq = 100
rate = rospy.Rate(freq)

tstate = None
tr = Twist()

			
while continuer :
	
	if len(buffstate) :
		tstate = buffstate[-1]
		del buffstate[:]
		tr = tstate			
				
	if tr is not None :
		pubR.publish(tr)
	
	if continuer :	
		rate.sleep()
	

