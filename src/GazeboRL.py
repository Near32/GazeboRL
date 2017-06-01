import subprocess
import thread
from threading import Lock
import time
import rospy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

def init_roscore() :
	subprocess.Popen(['roscore'])

class GazeboRL :
	def __init__(self,commands,observationsList=None,publishersList=None, rewardsList=None):
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
		
		self.atShudownKill = list()
		
		self.commands = commands
		self.observationsList = observationsList
		self.rewardsList = rewardsList
		self.publishersList = publishersList
		
		#ARE TO BE COMPUTED FROM THE OBSERVATIONS THREADS :
		self.observations = []
		self.observation = None
		self.rewards = []
		self.rewards = 0.0
		self.dones = []
		self.done = False
		self.infos = []
		self.info = None
		self.actions = []
		self.action = None
		
		self.rMutex = Lock()
		
	def make(self) :
		#ROSLAUNCHING with commands dictionary commands['launch'] = list of strings...
		for command in self.commands['launch'] :
			#subprocess.call( command, shell=True )
			p = subprocess.Popen( command.split())
			self.atShutdownKill.append(p)
			
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
		for proc in self.atShutdownKill :
			proc.kill()
			
		#command = 'pkill roslaunch'
		#subprocess.Popen( command.split())
		#command = 'pkill gzclient'
		#subprocess.Popen( command.split())
		#command = 'pkill gzserver'
		#subprocess.Popen( command.split())
		print("\n\nGAZEBO RL : ENVIRONMENT : CLOSED.\n\n")
		return 
		











class Swarm1GazeboRL(GazeboRL) :
	def __init__(self):
		self.continuousActions = False
		
		commands = {'launch': None}
		launchCom = []
		launchCom.append('roslaunch OPUSim robot1swarm.launch')
		launchCom.append('python ~/rosbuild_ws/sandbox/GazeboRL/src/reward.py -r 2.0 -v 1.0')
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
		
		rewardsList = []
		rewardsList.append('/RL/reward')
		
		
		self.observationsQueues = dict()
		for topic in observationsList :
			self.observationsQueues[topic] = []
		
		self.rewardsQueues = dict()
		for topic in rewardsList :
			self.rewardsQueues[topic] = []
		
		GazeboRL.__init__(self,commands,observationsList,publishersList,rewardsList)
		
		
	
	
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
		
		#reward :
		self.subscribers[self.rewardsList[0] ] = rospy.Subscriber( self.rewardsList[0], Float64, self.callbackREWARD )
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
			
	def synchroniseObservationsRewards(self) :
		observationItems = list()
		enoughItem = True
		for obsBuffer in self.observationsQueues :
			if len(obsBuffer) :
				obsItem = obsBuffer[-1]
				observationItems.append(obsItem)
			else :
				enoughItem = False
				
		if enoughItem :
			#erase :
			for obsBuffer in self.observationsQueues :
				obsBuffer = []
			#forward for the distribution :
			self.observations.append(observationItems)
		
		#deal with the reward(s) :	
		rewardItems = list()
		for rewardsBuffer in self.rewardsQueues :
			if len(rewardsBuffer) :
				rewardItem = rewardsBuffer[-1]
				rewardItems.append(rewardItem)
				rewardsBuffer = []
				#TODO handle the case when there is multiple rewards...
			
		if len(rewardItems):
			self.rewards.append(rewardItems)
			
			
			
	def callbackODOMETRY(self, odom ) :
		self.rMutex.acquire()
		
		self.observationsQueues[self.observationsList[1]].append( odom)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: ODOMETRY')
	
	def callbackREWARD(self, r ) :
		self.rMutex.acquire()
		
		self.rewardsQueues[self.rewardsList[0]].append( r)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: ODOMETRY')
	
	
	def callbackOMNIVIEW(self, omniview ) :
		self.rMutex.acquire()
		#print(omniview)
		self.observationsQueues[self.observationsList[0]].append( omniview)
		
		self.rMutex.release()
		#rospy.loginfo('DEBUG :: GazeboRL : received an observation from Gazebo... :: OMNIVIEW')
	
