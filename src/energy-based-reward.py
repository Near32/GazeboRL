#!/usr/bin/python

import numpy as np
import rospy
import time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float64
import argparse

parser = argparse.ArgumentParser(description="Energy-based Reward node for RL framework.")
parser.add_argument('-r', help='radius distance from the target.', dest='radius', type=float, default=2.0)
parser.add_argument('-v', help='rotational velocity around the target.', dest='velocity', type=float, default=1.0)
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

buffodom = list()
def callbackODOMETRY(odom) :
	buffodom.append(odom)
	


continuer = True
def shutdown() :
	continuer = False

rospy.init_node('reward_node', anonymous=False)
rospy.on_shutdown(shutdown)

subODOM = rospy.Subscriber( '/robot_model_teleop_0/odom_diffdrive', Odometry, callbackODOMETRY )
pubR = rospy.Publisher('/RL/reward',Float64,queue_size=10)


rate = rospy.Rate(100)

todom = None
tr = Float64()
tr.data = 0.0
maxvalue = 10.0

while continuer :
	
	if len(buffodom) :
		todom = buffodom[-1]
		del buffodom[:]
		
		#gather information :
		cp = todom.pose.pose.position
		ct = todom.twist.twist
		
		#let us compute the rewards to publish :
		radius = np.sqrt( cp.x**2+cp.y**2 )
		rp = (radius-args.radius)**2
		#rt = ( ct.linear.x - args.velocity )**2
		rt = (( ct.linear.x - args.velocity )**2)/10.0
		penality = ( ct.angular.z )**2 + (( ct.linear.x )**2)/20.0
		#high favours positional constraint...
		lambdap = 0.99
		#lp = 2.0
		#tr.data = -1.0 * ( lambdap*rp+(1-lambdap)*rt+lp*penality )
		tr.data = -1.0 * ( lambdap*rp+(1-lambdap)*rt +penality)
		
	if tr is not None :
		pubR.publish(tr)
	
	if continuer :	
		rate.sleep()


