import subprocess
import threading
#import thread
from threading import Lock
import time
import rospy

from math import sin, cos
import numpy as np

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray
from gazebo_msgs.msg import ModelStates
import os

def init_roscore(env,port):
	subprocess.Popen(['roscore -p '+str(port)+' &'],shell=True,env=env)

class GazeboRL :
	def __init__(self,commands,observationsList=None,publishersList=None, rewardsList=None, env=None):
		'''
		commands : dictionnary of list of string/commands to apply to subprocess :
		- launch : 
		- subs : list of strings of type of subscribe
		- pubs : list of strings of type of publishers
		
		observationsList : list of string of the name of the topic to subscribe : same order as commands['subs']
		publishersList : list of string of the name of the topic to publish to : same order as commands['pubs']
		'''
		
		
		#if commands.has_key('subs') :
		if 'subs' in commands :
			subc = commands['subs']
			if (len(subc) and observationsList is None) or len(subc) != len(observationsList) :
				raise ValueError('List of observations and subscribtions are not accorded.')
		
		#if commands.has_key('pubs') :
		if 'pubs' in commands :
			pubc = commands['pubs']
			if (len(pubc) and publishersList is None) or len(pubc) != len(publishersList) :
				raise ValueError('List of publishers and list of type of data to publish are not accorded.')
		
		self.env = os.environ
		if env is not None :
			self.env = env
			
		self.atShutdownKill = list()
		
		self.commands = commands
		self.observationsList = observationsList
		self.rewardsList = rewardsList
		self.publishersList = publishersList
		
		#ARE TO BE COMPUTED FROM THE OBSERVATIONS THREADS :
		self.observations = []
		self.observation = None
		self.rewards = []
		self.reward = 0.0
		self.dones = []
		self.done = False
		self.infos = []
		self.info = None
		self.actions = []
		self.action = None
		
		self.rMutex = Lock()
	
	def init_node(self) :
		rospy.init_node('GazeboRL_node', anonymous=False)#, xmlrpc_port=self.port)#,tcpros_port=self.port)
		rospy.on_shutdown(self.close)
			
	def make(self) :
		#ROSLAUNCHING with commands dictionary commands['launch'] = list of strings...
		for command in self.commands['launch'] :
			#subprocess.call( command, shell=True )
			#p = subprocess.Popen( command.split())
			#p = subprocess.Popen( command.split(), shell=True, env=self.env)
			p = subprocess.Popen( command, shell=True, env=self.env)
			self.atShutdownKill.append(p)
			
		#p.wait()
		rospy.loginfo("\nGAZEBO RL : ENVIRONMENT "+str(self.port)+" : LAUNCHING...\n")
	
		self.init_node()
		self.subscribtions()
		self.publishersCreations()
		
		return
		
	def subscribtions(self) :
		#TODO in inheritances...
		#ROSSUBSCRIBING with commands dictionary commands['subs'] = list of strings...
		
		#LAUNCH THREADS : subscribe
		
		return
	
	def publishersCreations(self) :
		#TODO in inheritances...
		#LAUNCH THREADS : publish
		
		return
		
	def actionHandling(self) :
		#TODO in inheritances..
		
		return
		
	def synchroniseObservationsRewards(self) :
		#TODO in inheritances...
		
		return
			
	def step(self,actions) :
		#TODO :
		
		self.rMutex.acquire()
		
		self.synchroniseObservationsRewards()
		
		#ROSPUB : action
		self.action = actions
		self.actions.append(self.action)	
			
		if len(self.observations) :
			self.observation = self.observations[-1]
			self.observations = []
		
		if len(self.rewards) :
			self.reward = self.rewards[-1]
			self.rewards = []
			
		if len(self.dones) :
			self.done = self.dones[-1]
			self.dones = []
		
		if len(self.infos) :
			self.info = self.infos[-1]
			self.infos = []
		
		self.rMutex.release()
		
		self.actionHandling()
		
		return self.observation,self.reward,self.done,self.info
		
	
	def setPause(self,pause=True) :
		if pause==True :
			#subprocess.Popen(['rosservice', 'call', '/gazebo/pause_physics'], shell=True, env=self.env)
			subprocess.Popen(['rosservice call /gazebo/pause_physics'], shell=True, env=self.env)
			rospy.loginfo("\nGAZEBO RL : ENVIRONMENT "+str(self.port)+" PHYSICS : PAUSED.\n")
		else :
			#subprocess.Popen(['rosservice', 'call', '/gazebo/unpause_physics'], shell=True, env=self.env)
			subprocess.Popen(['rosservice call /gazebo/unpause_physics'], shell=True, env=self.env)
			rospy.loginfo("\nGAZEBO RL : ENVIRONMENT "+str(self.port)+" PHYSICS : UNPAUSED.\n")
	
	
	def reset(self) :
		self.rMutex.acquire()
		
		self.setPause(True)
		time.sleep(1)
		
		#subprocess.Popen(['rosservice', 'call', '/gazebo/reset_world'], shell=True, env=self.env)
		subprocess.Popen(['rosservice call /gazebo/reset_world'], shell=True, env=self.env)
		#subprocess.Popen(['rosservice', 'call', '/gazebo/reset_simulation'])
		rospy.loginfo("\nGAZEBO RL : ENVIRONMENT "+str(self.port)+" : RESET.\n")
		time.sleep(1)
		
		self.setPause(False)
		
		self.rMutex.release()
	
	def close(self) :
		#TODO :
		# end services...
		#for proc in self.atShutdownKill :
		#	proc.kill()
			
		'''
		command = 'pkill roslaunch'
		subprocess.Popen( command.split())
		command = 'pkill gzclient'
		subprocess.Popen( command.split())
		command = 'pkill gzserver'
		subprocess.Popen( command.split())
		'''
		''''
		command = "kill -9 $(ps | grep \"roslaunch\" | awk \"{ print $1 }\")"
		subprocess.Popen( command, shell=True, env=self.env)
		command = "kill -9 $(ps | grep \"gzserver\" | awk \"{ print $1 }\")"
		subprocess.Popen( command, shell=True, env=self.env)
		
		command = "pkill gzclient"
		subprocess.Popen( command, shell=True, env=self.env)
		'''
		rospy.loginfo("\nGAZEBO RL : ENVIRONMENT "+str(self.port)+" : CLOSED.\n")
		return 
		











class Swarm1GazeboRL(GazeboRL) :
	def __init__(self,port=11311,energy_based=False, fromState=False, coupledSystem=False):
		self.continuousActions = False
		self.energy_based = energy_based
		self.fromState = fromState
		self.coupledSystem = coupledSystem
		
		self.port = port
		self.envdict = os.environ
		self.envdict["ROS_MASTER_URI"] = 'http://localhost:'+str(self.port)
		self.envdict["GAZEBO_MASTER_URI"]='http://localhost:'+str(self.port+40)
		
		init_roscore(self.envdict,self.port)
		time.sleep(1)
		
		commands = {'launch': None}
		launchCom = []
		
		if energy_based == False :
			launchCom.append('roslaunch -p '+str(self.port)+' GazeboRL robot1swarm.launch')
		else :
			if self.fromState == False :
				launchCom.append('roslaunch -p '+str(self.port)+' GazeboRL robot1swarm.EnergyBased.launch')
			else :
				launchCom.append('roslaunch -p '+str(self.port)+' GazeboRL robot1swarm.EnergyBased.fromState.launch')
		
		commands['launch'] = launchCom
		
		# SUBSCRIBERS :
		subsCom = []
		subsCom.append(Image)
		subsCom.append(Odometry)
		if self.fromState :
			subsCom.append(Float64MultiArray)
		if self.coupledSystem :
			subsCom.append(Twist)
		commands['subs'] = subsCom
		
		observationsList = []
		observationsList.append('/robot_model_teleop_0/OMNIVIEW')
		observationsList.append('/robot_model_teleop_0/odom_diffdrive')
		if self.fromState:
			observationsList.append('/RL/state')
		if self.coupledSystem :
			observationsList.append('/RL/zoh/robot_model_teleop_0/cmd_vel_controlLaw')
		
		# PUBLISHERS :
		pubsCom = []
		pubsCom.append(Twist)
		commands['pubs'] = pubsCom
		
		publishersList = []
		publishersList.append('/robot_model_teleop_0/cmd_vel')
		
		rewardsList = []
		rewardsList.append('/RL/reward')
		
		
		self.observationsQueues = dict()
		for topic in observationsList :
			self.observationsQueues[topic] = []
		
		self.rewardsQueues = dict()
		for topic in rewardsList :
			self.rewardsQueues[topic] = []
		
		GazeboRL.__init__(self,commands,observationsList,publishersList,rewardsList, env=self.envdict)
		
		
	
	
	def subscribtions(self) :
		#ROSSUBSCRIBING with commands dictionary commands['subs'] = list of strings...
		#LAUNCH THREADS : subscribe
		self.subscribers = dict()
		# OMNIVIEW :
		self.subscribers[self.observationsList[0] ] = rospy.Subscriber( self.observationsList[0], self.commands['subs'][0], self.callbackOMNIVIEW )
		#rospy.loginfo('{} :: {}'.format(self.observationsList[0], self.commands['subs'][0]) )
		
		# ODOMETRY :
		self.subscribers[self.observationsList[1] ] = rospy.Subscriber( self.observationsList[1], self.commands['subs'][1], self.callbackODOMETRY )
		#rospy.loginfo('{} :: {}'.format(self.observationsList[1], self.commands['subs'][1]) )
		
		# MODEL_STATE :
		if self.fromState :
			self.subscribers[self.observationsList[2] ] = rospy.Subscriber( self.observationsList[2], self.commands['subs'][2], self.callbackMODELSTATE )
		#rospy.loginfo('{} :: {}'.format(self.observationsList[1], self.commands['subs'][1]) )
		
		# CONTROLLAW :
		if self.coupledSystem :
			if self.fromState :
				self.subscribers[self.observationsList[3] ] = rospy.Subscriber( self.observationsList[3], self.commands['subs'][3], self.callbackCONTROLLAW )
			else :
				self.subscribers[self.observationsList[2] ] = rospy.Subscriber( self.observationsList[2], self.commands['subs'][3], self.callbackCONTROLLAW )
		#rospy.loginfo('{} :: {}'.format(self.observationsList[1], self.commands['subs'][1]) )
		
		#reward :
		self.subscribers[self.rewardsList[0] ] = rospy.Subscriber( self.rewardsList[0], Float64, self.callbackREWARD )
		#rospy.loginfo('{} :: {}'.format(self.observationsList[1], self.commands['subs'][1]) )
		
		
		return
	
	def publishersCreations(self) :
		#LAUNCH THREADS : publish
		self.publishers = dict()
		self.pub_bools = dict()
		self.pub_rates = dict()
		self.pub_threads = dict()
		# VELOCITY :
		self.publishers[self.publishersList[0]] = rospy.Publisher( self.publishersList[0], self.commands['pubs'][0], queue_size=10) 
		self.pub_bools[self.publishersList[0]] = True
		self.pub_rates[self.publishersList[0]] = rospy.Rate(1)
		#self.pub_threads[self.publishersList[0]] = thread.start_new_thread ( Swarm1GazeboRL.publishVELOCITY, (self,self.continuousActions) )
		self.pub_threads[self.publishersList[0]] = threading.Thread( target=Swarm1GazeboRL.publishVELOCITY, args=(self,self.continuousActions) )
		self.pub_threads[self.publishersList[0]].start()
		
		return
	
	def publishVELOCITY(self,continuous=True) :
		twistmsg = Twist()
		
		if continuous :
			while self.continuer :
				while self.pub_bools[self.publishersList[0]] :
					action = [0,0]
			
					if len(self.actions) :
						action = self.actions[-1]
			
					twistmsg.linear.x = action[0]
					twistmsg.angular.z = action[1]
			
					self.publishers[self.publishersList[0]].publish( twistmsg )
					self.pub_rates[self.publishersList[0]].sleep()
					#rospy.loginfo('DEBUG :: GazeboRL :: Published a twist message...')
		else :
			action = [0,0]
	
			if len(self.actions) :
				action = self.actions[-1]
	
			twistmsg.linear.x = action[0]
			twistmsg.angular.z = action[1]
	
			self.publishers[self.publishersList[0]].publish( twistmsg )
			#rospy.loginfo('DEBUG :: GazeboRL :: Published a twist message:')
			#rospy.loginfo(action)
				
				
		return
		
	def actionHandling(self) :
		if self.continuousActions == False :
			self.publishVELOCITY(self.continuousActions)
			
	def synchroniseObservationsRewards(self) :
		observationItems = dict()
		enoughItem = True
		for topic in self.observationsQueues.keys() :
			if len(self.observationsQueues[topic]):
				#rospy.loginfo('DEBUG:'+topic)
				#rospy.loginfo(self.observationsQueues[topic])
				obsItem = self.observationsQueues[topic][-1]
				observationItems[topic]=obsItem
			else :
				enoughItem = False
				
		if enoughItem :
			#erase :
			for topic in self.observationsQueues :
				self.observationsQueues[topic] = []
			#forward for the distribution :
			self.observations.append(observationItems)
		
		#deal with the reward(s) :	
		rewardItems = dict()
		for topic in self.rewardsQueues.keys() :
			if len(self.rewardsQueues[topic]):
				#rospy.loginfo('DEBUG:reward:'+topic)
				#rospy.loginfo(self.rewardsQueues[topic][-1])
				rewardItem = self.rewardsQueues[topic][-1]
				rewardItems[topic] = rewardItem
				self.rewardsQueues[topic] = []
				#TODO handle the case when there is multiple rewards...
			
		if len(rewardItems):
			self.rewards.append(rewardItems)
			
	
	def reset(self) :
		self.rMutex.acquire()
		
		self.setPause(True)
		time.sleep(1)
		
		subprocess.Popen(['rosservice call /gazebo/reset_world'], shell=True, env=self.env)
		#subprocess.Popen(['rosservice call /gazebo/reset_world'], env=self.env)
		#subprocess.Popen(['rosservice', 'call', '/gazebo/reset_simulation'])
		rospy.loginfo("\nGAZEBO RL : ENVIRONMENT "+str(self.port)+" : RESET.\n")
		time.sleep(1)
		
		self.setPause(False)
		
		self.randomInitialization()
		
		self.rMutex.release()
		
	
	def randomInitialization(self) :
		randx = 0.0
		randy = 0.0
		robot_name = "robot_0"
		
		notokay = True
		sizemin = -2.5
		sizemax = 2.5
		sizeobs = 0.75
		posobs = [ [2.0, 0.0], [-2.0, 0.0], [0.0, 0.0] ]
		#posobs = [ [0.0, 0.0] ]
		
		while notokay	:
			notokay = False
			randx = np.random.uniform(low=sizemin,high=sizemax)
			randy = np.random.uniform(low=sizemin,high=sizemax)
			
			#verification :
			for obs in posobs :
				distance = np.sqrt( (randx-obs[0])**2+(randy-obs[1])**2 )
				if distance < sizeobs :
					notokay = True
		
		randtheta = np.random.uniform(low=-3.1415,high=3.1415)
		
		#subprocess.Popen(['rosservice', 'call', '/gazebo/get_model_state', '\'model_name: robot_0\' ', '\'relative_entity_name: world\' '])
		#subprocess.Popen(['rosservice', 'call', '/gazebo/get_world_properties' ])
		#subprocess.Popen(['rosservice', 'call', '/gazebo/set_model_state', "{model_state: { model_name: "+robot_name+", pose: { position: { x: "+str(randx)+", y: "+str(randy)+", z: 0 }, orientation: {x: 0, y: 0, z: "+str(sin(randtheta/2.0))+", w: "+str(cos(randtheta/2.0))+" } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world }} "], shell=True, env=self.env)		
		#subprocess.Popen(['rosservice call /gazebo/set_model_state {model_state: { model_name: '+robot_name+', pose: { position: { x: '+str(randx)+', y: '+str(randy)+', z: 0 }, orientation: {x: 0, y: 0, z: '+str(sin(randtheta/2.0))+', w: '+str(cos(randtheta/2.0))+' } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world }} '], shell=True, env=self.env)		
		command = 'rosservice call /gazebo/set_model_state \"{ model_state: { model_name: '+robot_name+', pose: { position: { x: '+str(randx)+', y: '+str(randy)+', z: 0 }, orientation: {x: 0, y: 0, z: '+str(sin(randtheta/2.0))+', w: '+str(cos(randtheta/2.0))+' } }, twist: { linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 } } , reference_frame: world }}\" '
		subprocess.Popen(command, shell=True, env=self.env)		
		#rospy.loginfo("\n"+"--- "+command+" ---"+"\n")
		
			
	def callbackODOMETRY(self, odom ) :
		self.rMutex.acquire()
		
		self.observationsQueues[self.observationsList[1]].append( odom)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: ODOMETRY')
	
	def callbackREWARD(self, r ) :
		self.rMutex.acquire()
		
		self.rewardsQueues[self.rewardsList[0]].append( r)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: Reward')
	
	
	def callbackOMNIVIEW(self, omniview ) :
		self.rMutex.acquire()
		#rospy.loginfo(omniview)
		self.observationsQueues[self.observationsList[0]].append( omniview)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: OMNIVIEW')
	
	def callbackMODELSTATE(self, model_states ) :
		self.rMutex.acquire()
		self.observationsQueues[self.observationsList[2]].append( model_states)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: gazebo/model_states')
	
	def callbackCONTROLLAW(self, cmd_vel ) :
		self.rMutex.acquire()
		#rospy.loginfo(cmd_vel)
		if self.fromState :
			self.observationsQueues[self.observationsList[3]].append( cmd_vel)
		else :
			self.observationsQueues[self.observationsList[2]].append( cmd_vel)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: cmd_vel_controlLaw')
	
