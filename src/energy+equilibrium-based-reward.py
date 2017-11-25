#!/usr/bin/python

import numpy as np
import rospy
import tf
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
parser.add_argument('-mass', help='fictive mass to use in the kinetic energy computation.', dest='mass', default=1e1 )
args, unknown = parser.parse_known_args()

print(args)
print(unknown)

buffstate = list()
def callbackState(state) :
	buffstate.append(state)
	


continuer = True
def shuttingdown() :
	continuer = False

rospy.init_node('EnergyBasedReward_node', anonymous=False)
rospy.on_shutdown(shuttingdown)

subState = rospy.Subscriber( '/gazebo/model_states', ModelStates, callbackState )
pubR = rospy.Publisher('/RL/reward',Float64,queue_size=10)

freq = 100
dt = 1.0/freq
rate = rospy.Rate(freq)

tstate = None
tr = Float64()
tr.data = 0.0
maxvalue = 10.0
beta = 1.0


def f(a,r,rd) :
	return a*r*(1-(r**2)/(rd**2))
	
def g(om,eps,psi) :
	return om+eps*np.sin(psi)
	
def v(kv,a,r,rd,theta,om,eps,psi) :
	return kv*( f(a,r,rd)*np.cos(theta)+r*g(om,eps,psi)*np.sin(theta) )

def w(kw,a,r,rd,theta,om,eps,psi) :
	return kw*( r*g(om,eps,psi)*np.cos(theta)-f(a,r,rd)*np.sin(theta) )
	
def h1(kr, rdd, rd) :
	return kr*(rdd-rd)

def h2(kr, rt, robs, thetaobs) :
	tobs_f = thetaobs
	#denum = np.log( 1.0+kr*np.abs(rt-robs)+1e-3)
	denum = np.log( 1.0+kr*np.maximum( 0.0,robs-0.75*rt )+1e-3)
	
	if tobs_f >= 0.0 :
		return 1.0/denum
	else :
		return -1.0/denum

def rddot( kr, rdd, rd, rt, robs, thetaobs) :
	tobs_f = thetaobs
	if np.abs(thetaobs) > np.pi/2.0 :
		tobs_f = np.pi/2.0
		
	ltheta = np.cos(tobs_f)

	h_1 = h1(kr, rdd, rd)
	h_2 = h2(kr, rt, robs, tobs_f)

	return (1.0-ltheta)*h_1-ltheta*h_2


def controlLaw(kv,kw,a,r,rd,theta,om,eps,psi) :
	return np.reshape( [ v(kv,a,r,rd,theta,om,eps,psi), w(kv,a,r,rd,theta,om,eps,psi) ], newshape=(2,1) )
	
	
rd = dict()
rewards = dict()
			
while not rospy.is_shutdown() :
	
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
		
		#take care of the target first :
		for (name,pose,twist) in zip(tstate.name,tstate.pose,tstate.twist) :
			if 'target' in name :
				p = np.array([ pose.position.x , pose.position.y, pose.position.z ])
				q = np.array([ pose.orientation.x , pose.orientation.y, pose.orientation.z, pose.orientation.w ])
				tl = np.array([ twist.linear.x , twist.linear.y, twist.linear.z ])
				ta = np.array([ twist.angular.x , twist.angular.y, twist.angular.z ])
				
				target = (p, q, tl, ta)
		
		# now we have everything to compute for the robots :
		for (name,pose,twist) in zip(tstate.name,tstate.pose,tstate.twist) :
			if 'robot' in name :
				#then we can count one more robot :
				nbrRobot +=1
				#let us initialize rd if necessary :
				if not (name in rd) :
					rd[name] = args.radius
				#save its pose and twist :
				p = np.array([ pose.position.x , pose.position.y, pose.position.z ])
				tl = np.array([ twist.linear.x , twist.linear.y, twist.linear.z ])
				ta = np.array([ twist.angular.x , twist.angular.y, twist.angular.z ])
				
				quaternion = (
					pose.orientation.x,
					pose.orientation.y,
					pose.orientation.z,
					pose.orientation.w)
				euler = tf.transformations.euler_from_quaternion(quaternion)
				
				#backup computation... :
				phi = np.arctan2( p[1], p[0] )
				r = np.sqrt( p[0]**2+p[1]**2)
				theta = euler[2] - phi
				
				# real computation ...
				if target is not None :
					ptarget = p-target[0]
					#phi = np.arctan2( ptarget[1], ptarget[0] )
					#r = np.sqrt( ptarget[0]**2+ptarget[1]**2)
					#theta = euler[2] - phi
					
					phi = np.arctan2( ptarget[1], ptarget[0] )
				
					while phi > np.pi :
						phi -= 2*np.pi
					while phi < -np.pi :
						phi += 2*np.pi
				
					r = np.sqrt( ptarget[0]**2+ptarget[1]**2)
					theta = euler[2] - phi
				
					while theta > np.pi :
						theta -= 2*np.pi
					while theta < -np.pi :
						theta += 2*np.pi
					
				
				robots.append( {'name' : name, 'rd' : rd[name], 'phi' : phi, 'r': r, 'theta' : theta, 'position' : p , 'euler' : euler, 'linear_vel' : tl, 'angular_vel' : ta} )
			
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
				
				obstacles.append( (p, q, tl, ta, name) )
			
		if target is not None and nbrRobot >=1:	
			eps = args.epsilon
			if nbrRobot == 1 :
				eps = 0.0
			
			if nbrRobot >= 1 :
				swarm_kinetic_energy = 0.0
				swarm_equilibrium = 0.0
				#let us ordonate them by values of theta, from min to max :
				robots = sorted(robots, key=lambda el : el['phi'])
				for i in range(nbrRobot) :
					#coupling Index :
					cIdx = (i+1)%nbrRobot
					# compute the angular differences :
					psi = robots[cIdx]['phi'] - robots[i]['phi'] 
					#make sure that those values are positively taken :
					while psi < 0.0 :
						psi += 2*np.pi
						
					robots[i]['psi'] = psi
					
					#compute the control law 
					robots[i]['controlLaw'] = controlLaw( args.kv, args.kw, args.a, robots[i]['r'], args.radius, robots[i]['theta'], args.Omega, eps, robots[i]['psi'])
					
					#compute distance and angular offset to obstacles :
					dists = list()
					for obs in obstacles :
						dx = robots[i]['position'][0]-obs[0][0]
						dy = robots[i]['position'][1]-obs[0][1]
						dist = np.sqrt( (dx)**2 + (dy)**2)
						angular = np.arctan2(dy,dx) - robots[i]['theta']
						dists.append( (obs[4], dist, angular) )
					mindistobs = list( sorted(dists, key=lambda el : el[1] )[0] )
					#refactoring the thetaobs var between -pi and +pi :
					#rospy.loginfo(mindistobs)
					while mindistobs[2] >= np.pi :
						mindistobs[2] -= 2*np.pi
					while mindistobs[2] <= -np.pi :
						mindistobs[2] += 2*np.pi
					#rospy.loginfo('robot {} : mindist obs : robs={} :: thetaobs={}'.format(i,mindistobs[1], mindistobs[2]*180.0/np.pi) )
					robots[i]['robs'] = mindistobs[1]
					robots[i]['thetaobs'] = mindistobs[2]
					
					# compute rddot :
					robots[i]['rdd'] = rddot( args.kR, args.radius, rd[robots[i]['name']], args.thresholdDistAccount, robots[i]['robs'], robots[i]['thetaobs'] )
					
					# update rdd :
					#rospy.loginfo('robot {} :  rd={} :: rddot={}'.format(i, rd[robots[i]['name']], robots[i]['rdd']) )
					rd[robots[i]['name']] +=  dt*robots[i]['rdd']
					
					if rd[robots[i]['name']] >= 8.0 :
						rd[robots[i]['name']] = 8.0
					if rd[robots[i]['name']] <= 1.0 :
						rd[robots[i]['name']] = 1.0
					#rospy.loginfo('robot {} :  rd={} :: rddot={}'.format(i, rd[robots[i]['name']], robots[i]['rdd']) )
						
				tr.data = 0.0	
				for i in range(nbrRobot) :
					cIdx = (i+1)%nbrRobot
					psi_dot = robots[cIdx]['controlLaw'][0,0]*np.sin( robots[cIdx]['theta'] )/(robots[cIdx]['r']+1e-4) - robots[i]['controlLaw'][1,0]*np.sin( robots[i]['theta'] )/(robots[i]['r']+1e-4)
					#robots[i]['state_dot'] =  np.reshape( np.array( [ robots[i]['controlLaw'][0,0]*np.cos( robots[i]['theta']) , robots[i]['controlLaw'][1,0] , psi_dot ] ), newshape=(3,1) )
					robots[i]['state_dot'] =  np.reshape( [ robots[i]['controlLaw'][0,0]*np.cos( robots[i]['theta']) , robots[i]['controlLaw'][1,0] , psi_dot, robots[i]['rdd'] ], newshape=(4,1) )
					#robots[i]['kinetic_energy'] = 0.5*args.mass* (  robots[i]['state_dot'][0,0]**2 + robots[i]['state_dot'][1,0]**2 + robots[i]['state_dot'][2,0]**2 )
					
					#robots[i]['kinetic_energy'] = 0.5 * args.mass * (  robots[i]['state_dot'][0,0]**2 + robots[i]['state_dot'][1,0]**2 + robots[i]['state_dot'][2,0]**2 + robots[i]['state_dot'][3,0]**2  )
					robots[i]['kinetic_energy'] = 0.5 * args.mass * (  robots[i]['state_dot'][0,0]**2 + robots[i]['state_dot'][1,0]**2 + robots[i]['state_dot'][3,0]**2  )
					
					robots[i]['equilibrium'] = ((robots[i]['r']-args.radius)/args.radius)**2 + ((robots[i]['theta']+np.pi/2.0)/(np.pi/2.0))**2
					
					
					swarm_kinetic_energy += robots[i]['kinetic_energy']
					swarm_equilibrium += robots[i]['equilibrium']
					
					rewards[robots[i]['name']] = robots[i]['kinetic_energy']+beta*robots[i]['equilibrium']
					
					tr.data += 1.0/(1e-3 + rewards[robots[i]['name']])
					
					#rospy.loginfo('robot: {} :: phi={} :: psi={} :: theta={}'.format(robots[i]['name'],robots[i]['phi']*180.0/np.pi,robots[i]['psi']*180.0/np.pi, robots[i]['theta']*180.0/np.pi ) )
					#rospy.loginfo('robot: {} :: v={} :: w={}'.format(robots[i]['name'],robots[i]['controlLaw'][0],robots[i]['controlLaw'][1] ) )
					#rospy.loginfo('robot: {} :: {} {} {}'.format(robots[i]['name'],robots[i]['state_dot'][0],robots[i]['state_dot'][1], robots[i]['state_dot'][2] ) )
					#rospy.loginfo(robots[i]['state_dot'])
					#rospy.loginfo('robot: {} :: kinetic energy = {}'.format(robots[i]['name'],robots[i]['kinetic_energy'] ) )
			
			else :
				tr.data = 0.0
				
				
	if tr is not None :
		pubR.publish(tr)
	
	if continuer :	
		rate.sleep()
	


