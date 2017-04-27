import subprocess
import threading
from threading import Lock
import time

class GazeboRL :
	def __init__(self,commands,observationsList=None):
		'''
		commands : dictionnary of list of string/commands to apply to subprocess :
		- launch : 
		- subs : list of strings of type of subscribe
		
		observationsList : list of string of the name of the topic to subscribe : same order as commands['subs']
		 
		'''
		if commands.has_key('subs') :
			subc = commands['subs']
			if (len(subc) and observationsList is None) or len(subc) != len(observationsList) :
				raise ValueError('List of observations and subscribtions are not accorded.')
		
		
		self.commands = commands
		
		self.obs_threads = []
		#ARE TO BE COMPUTED FROM THE OBSERVATIONS THREADS :
		self.observations = []
		self.rewards = []
		self.dones = []
		self.infos = []
		
		self.rMutex = Lock()
		
	def make(self) :
		#ROSLAUNCHING with commands dictionary commands['launch'] = list of strings...
		for command in self.commands['launch'] :
			#subprocess.call( command, shell=True )
			p = subprocess.Popen( command.split())
			
		#p.wait()
		print("\n\nGAZEBO RL : ENVIRONMENT : LAUNCHING...\n\n")
	
		
		self.subscribtions()
		
		return
		
	def subscribtions(self) :
		#TODO in inheritances...
		#ROSSUBSCRIBING with commands dictionary commands['subs'] = list of strings...
		
		#LAUNCH THREADS : subscribe
		
		#LAUNCH THREADS : publish
		return
	
	
	def step(self,actions) :
		#TODO :
		#ROSPUB : action
		
		#WAIT FOR PUBLISH + RECEIVE...????
		time.sleep(0.001)
		
		
		
		self.rMutex.acquire()
		
		observation = None
		if len(self.observations) :
			self.observations[-1]
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
		
		return observation,reward,done,info
		
	
	def setPause(self,pause=True) :
		if pause==True :
			subprocess.Popen(['rosservice', 'call', '/gazebo/pause_physics'])
			print("\n\nGAZEBO RL : ENVIRONMENT PHYSICS : PAUSED.\n\n")
		else :
			subprocess.Popen(['rosservice', 'call', '/gazebo/unpause_physics'])
			print("\n\nGAZEBO RL : ENVIRONMENT PHYSICS : UNPAUSED.\n\n")
	
	
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
		commands = {'launch': None}
		launchCom = []
		launchCom.append('roslaunch OPUSim robot1swarm.launch')
		#launchCom.append('roslaunch OPUSim robot2swarm.launch')
		
		commands['launch'] = launchCom
		
		observationsList = None
		
		GazeboRL.__init__(self,commands,observationsList)
	
	
	def subscribtions(self) :
		#TODO in inheritances...
		#ROSSUBSCRIBING with commands dictionary commands['subs'] = list of strings...
		
		#LAUNCH THREADS : subscribe
		print("\n\n\nHELLO\n\n\n")
		
		#LAUNCH THREADS : publish
		return
	
	
	
