import subprocess
import thread
from threading import Lock
import time
import rospy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class GazeboRL :
	def __init__(self,commands,observationsList=None,publishersList=None):
		'''
		commands : dictionnary of list of string/commands to apply to subprocess :
		- launch : 
		- subs : list of strings of type of subscribe
		- pubs : list of strings of type of publishers
		
		observationsList : list of string of the name of the topic to subscribe : same order as commands['subs']
		publishersList : list of string of the name of the topic to publish to : same order as commands['pubs']
		'''
		
		rospy.init_node('GazeboRL_node', anonymous=False)
		rospy.on_shutdown(self.close)
		
		if commands.has_key('subs') :
			subc = commands['subs']
			if (len(subc) and observationsList is None) or len(subc) != len(observationsList) :
				raise ValueError('List of observations and subscribtions are not accorded.')
		
		if commands.has_key('pubs') :
			pubc = commands['pubs']
			if (len(pubc) and publishersList is None) or len(pubc) != len(publishersList) :
				raise ValueError('List of publishers and list of type of data to publish are not accorded.')
		
		
		self.commands = commands
		self.observationsList = observationsList
		self.publishersList = publishersList
		
		#ARE TO BE COMPUTED FROM THE OBSERVATIONS THREADS :
		self.observations = []
		self.rewards = []
		self.dones = []
		self.infos = []
		self.actions = []
		
		self.rMutex = Lock()
		
	def make(self) :
		#ROSLAUNCHING with commands dictionary commands['launch'] = list of strings...
		for command in self.commands['launch'] :
			#subprocess.call( command, shell=True )
			p = subprocess.Popen( command.split())
			
		#p.wait()
		print("\n\nGAZEBO RL : ENVIRONMENT : LAUNCHING...\n\n")
	
		
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
		
			
	def step(self,actions) :
		#TODO :
		
		self.rMutex.acquire()
		
		#ROSPUB : action
		self.actions.append(actions)	
			
		observation = None
		if len(self.observations) :
			observation = self.observations[-1]
			self.observation = []
		
		reward = None
		if len(self.rewards) :
			reward = self.rewards[-1]
			self.rewards = []
			
		done = False
		if len(self.dones) :
			done = self.dones[-1]
			self.dones = []
		
		info = dict()
		if len(self.infos) :
			info = self.infos[-1]
			self.infos = []
		
		self.rMutex.release()
		
		self.actionHandling()
		
		return observation,reward,done,info
		
	
	def setPause(self,pause=True) :
		if pause==True :
			subprocess.Popen(['rosservice', 'call', '/gazebo/pause_physics'])
			print("\n\nGAZEBO RL : ENVIRONMENT PHYSICS : PAUSED.\n\n")
		else :
			subprocess.Popen(['rosservice', 'call', '/gazebo/unpause_physics'])
			print("\n\nGAZEBO RL : ENVIRONMENT PHYSICS : UNPAUSED.\n\n")
	
	
	def reset(self) :
		self.rMutex.acquire()
		
		self.setPause(True)
		time.sleep(1)
		
		subprocess.Popen(['rosservice', 'call', '/gazebo/reset_world'])
		print("\n\nGAZEBO RL : ENVIRONMENT : RESET.\n\n")
		time.sleep(1)
		
		self.setPause(False)
		
		self.rMutex.release()
	
	def close(self) :
		#TODO :
		# end services...
		command = 'pkill roslaunch'
		subprocess.Popen( command.split())
		command = 'pkill gzclient'
		subprocess.Popen( command.split())
		command = 'pkill gzserver'
		subprocess.Popen( command.split())
		print("\n\nGAZEBO RL : ENVIRONMENT : CLOSED.\n\n")
		return 
		


class Swarm1GazeboRL(GazeboRL) :
	def __init__(self):
		self.continuousActions = False
		
		commands = {'launch': None}
		launchCom = []
		launchCom.append('roslaunch OPUSim robot1swarm.launch')
		#launchCom.append('roslaunch OPUSim robot2swarm.launch')
		commands['launch'] = launchCom
		
		# SUBSCRIBERS :
		subsCom = []
		subsCom.append(Image)
		subsCom.append(Odometry)
		commands['subs'] = subsCom
		
		observationsList = []
		observationsList.append('/robot_model_teleop_0/OMNIVIEW')
		observationsList.append('/robot_model_teleop_0/odom_diffdrive')
		
		# PUBLISHERS :
		pubsCom = []
		pubsCom.append(Twist)
		commands['pubs'] = pubsCom
		
		publishersList = []
		publishersList.append('/robot_model_teleop_0/cmd_vel')
		
		
		self.observationsQueues = dict()
		for topic in observationsList :
			self.observationsQueues[topic] = []
		
		GazeboRL.__init__(self,commands,observationsList,publishersList)
		
		
	
	
	def subscribtions(self) :
		#ROSSUBSCRIBING with commands dictionary commands['subs'] = list of strings...
		#LAUNCH THREADS : subscribe
		self.subscribers = dict()
		# OMNIVIEW :
		self.subscribers[self.observationsList[0] ] = rospy.Subscriber( self.observationsList[0], self.commands['subs'][0], self.callbackOMNIVIEW )
		#print('{} :: {}'.format(self.observationsList[0], self.commands['subs'][0]) )
		
		# ODOMETRY :
		self.subscribers[self.observationsList[1] ] = rospy.Subscriber( self.observationsList[1], self.commands['subs'][1], self.callbackODOMETRY )
		#print('{} :: {}'.format(self.observationsList[1], self.commands['subs'][1]) )
		
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
		self.pub_threads[self.publishersList[0]] = thread.start_new_thread ( Swarm1GazeboRL.publishVELOCITY, (self,self.continuousActions) )
		
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
					rospy.loginfo('DEBUG :: GazeboRL :: Published a twist message...')
		else :
			action = [0,0]
	
			if len(self.actions) :
				action = self.actions[-1]
	
			twistmsg.linear.x = action[0]
			twistmsg.angular.z = action[1]
	
			self.publishers[self.publishersList[0]].publish( twistmsg )
			rospy.loginfo('DEBUG :: GazeboRL :: Published a twist message...')
				
				
		return
		
	def actionHandling(self) :
		if self.continuousActions == False :
			self.publishVELOCITY(self.continuousActions)
			
		
			
	def callbackODOMETRY(self, odom ) :
		self.rMutex.acquire()
		
		self.observationsQueues[self.observationsList[1]].append( odom)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: ODOMETRY')
	
	
	def callbackOMNIVIEW(self, omniview ) :
		self.rMutex.acquire()
		
		self.observationsQueues[self.observationsList[0]].append( omniview)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: OMNIVIEW')
	
