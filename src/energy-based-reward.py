#!/usr/bin/python

import numpy as np
import rospy
import time

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64
import argparse

parser = argparse.ArgumentParser(description="Energy-based Reward node for RL framework.")
parser.add_argument('-r', help='radius distance from the target.', dest='radius', type=float, default=2.0)
parser.add_argument('-Omega', help='natural frequency of the oscillators.', dest='Omega', type=float,	default=2.0)
parser.add_argument('-tDA', help='threshold distance to account for obstacles.', dest='thresholdDistAccount', type=float, default=0.6)
parser.add_argument('-a', help='proportional gain to the radius controller.', dest='a', type=float, default=1.5)
parser.add_argument('-kv', help='proportional gain for the linear velocity controller.', dest='kv', type=float, default=0.1)
parser.add_argument('-kw', help='proportional gain for the angular velocity controller.', dest='kw', type=float, default=0.2)
parser.add_argument('-kR', help='proportional gain for the radius deviation controller.', dest='kR', type=float, default=10.0)
parser.add_argument('-epsilon', help='coupling strength between oscillators.', dest='epsilon', default=1.0)

args, unknown = parser.parse_known_args()

print(args)
print(unknown)

buffstate = list()
def callbackState(state) :
	buffstate.append(state)
	


continuer = True
def shutdown() :
	continuer = False

rospy.init_node('EnergyBasedReward_node', anonymous=False)
rospy.on_shutdown(shutdown)

subState = rospy.Subscriber( '/gazebo/model_states', ModelStates, callbackState )
pubR = rospy.Publisher('/RL/reward',Float64,queue_size=10)


rate = rospy.Rate(100)

tstate = None
tr = Float64()
tr.data = 0.0
maxvalue = 10.0

while continuer :
	
	if len(buffstate) :
		tstate = buffstate[-1]
		del buffstate[:]
		
		#gather information :
		nbrRobot = 0
		#cp = todom.pose.pose.position
		#ct = todom.twist.twist
		#rospy.loginfo(tstate)
		robots = list()
		target = None
		obstacles = list()
		
		for (name,pose,twist) in zip(tstate.name,tstate.pose,tstate.twist) :
			if 'robot' in name :
				#then we can count one more robot :
				nbrRobot +=1
				#save its pose and twist :
				p = np.array([ pose.position.x , pose.position.y, pose.position.z ])
				q = np.array([ pose.orientation.x , pose.orientation.y, pose.orientation.z, pose.orientation.w ])
				tl = np.array([ twist.linear.x , twist.linear.y, twist.linear.z ])
				ta = np.array([ twist.angular.x , twist.angular.y, twist.angular.z ])
				
				robots.append( (p , q, tl, ta) )
			
			if 'target' in name :
				p = np.array([ pose.position.x , pose.position.y, pose.position.z ])
				q = np.array([ pose.orientation.x , pose.orientation.y, pose.orientation.z, pose.orientation.w ])
				tl = np.array([ twist.linear.x , twist.linear.y, twist.linear.z ])
				ta = np.array([ twist.angular.x , twist.angular.y, twist.angular.z ])
				
				target = (p, q, tl, ta)
				
			if 'obstacle' in name :
				p = np.array([ pose.position.x , pose.position.y, pose.position.z ])
				q = np.array([ pose.orientation.x , pose.orientation.y, pose.orientation.z, pose.orientation.w ])
				tl = np.array([ twist.linear.x , twist.linear.y, twist.linear.z ])
				ta = np.array([ twist.angular.x , twist.angular.y, twist.angular.z ])
				
				obstacles.append( (p, q, tl, ta) )
			
		if target is not None :	
			#let us compute the rewards to publish :
			#rospy.loginfo(robots)
			tr.data = -1.0 #* ( lambdap*rp+(1-lambdap)*rt +penality)
			
	if tr is not None :
		pubR.publish(tr)
	
	if continuer :	
		rate.sleep()


