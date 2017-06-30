# # Reinforcement Learning : DDPG-A2C
## TODO : implement the target network trick ?

useGAZEBO = True

import threading
import multiprocessing
import numpy as np

if useGAZEBO :
	from GazeboRL import GazeboRL, Swarm1GazeboRL, init_roscore
	import time
	import rospy
	from Agent1 import NN, INPUT_SHAPE_R, resize, rgb2yuv
	from cv_bridge import CvBridge
	bridge = CvBridge()

	def ros2np(img) :
		return bridge.imgmsg_to_cv2(img, "bgr8")
else :
	import gym

import cv2
import os
import numpy as np


if useGAZEBO :
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
	


if useGAZEBO :
	env = Swarm1GazeboRL()
	env.make()
	print('\n\nwait for 5 sec...\n\n')
	time.sleep(5)
	env.reset()
	env.setPause(False)

img_size = (84,84,1)
if useGAZEBO :
	img_size = (60,160,3)

rec = False
# In[35]:

maxReplayBufferSize = 20
max_episode_length = 300
updateT = 2
nbrStepsPerReplay = 100
gamma = .99 # discount rate for advantage estimation and reward discounting
imagesize = [img_size[0],img_size[1], img_size[2] ]
s_size = imagesize[0]*imagesize[1]*imagesize[2]
h_size = 256

a_size = 1
model_path = './model-RL-Pendulum'
if useGAZEBO :
	a_size = 2	
	model_path = './model-RL3-GazeboRL-robot1swarm'
num_workers = 4
lr=1e-2

if not os.path.exists(model_path):
    os.makedirs(model_path)    
#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')



def preprocess(image, imghr=60, imgwr=320) :
	img = resize(image, imghr, imgwr)
	image = rgb2yuv(img)
	image = np.array(image)*1.0/127.5
	image -= 1.0
	#plt.imshow(image)
	#plt.show()
	return image.reshape((1,-1))
	
def envstep(env,action) :
	output = env.step(action)
	outimg = None
	outr = None
	outdone = False
	outinfo = None
	
	if output[0] is not None :
		for topic in output[0].keys() :
			if 'OMNIVIEW' in topic :
				img = np.array(ros2np(output[0][topic]))
				outimg = img
	else :
		outimg = np.zeros(size=img_size)
	
	if output[1] is not None :
		outr = output[1]['/RL/reward'].data
	
	if output[2] is not None :
		outdone = output[2]
		
	if output[3] is not None :
		outinfo = output[3]
		
	return outimg, outr, outdone, outinfo
	
	

show = False
load_model = False



import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
#get_ipython().magic('matplotlib inline')

from random import choice
from time import sleep
from time import time

# In[20]:

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
	#s = frame[10:-10,30:-30]
	s = scipy.misc.imresize(s,[84,84])
	s = np.reshape(s,[np.prod(s.shape)]) / 255.0
	return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


# In[21]:

class AC_Network():
	def __init__(self,imagesize,s_size,h_size, a_size,scope,trainer,rec=False):
		with tf.variable_scope(scope):
			#Input and visual encoding layers
			#PLACEHOLDER :
			self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
			#
			self.imageIn = tf.reshape(self.inputs,shape=[-1,imagesize[0],imagesize[1],imagesize[2]])
			self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
				inputs=self.imageIn,num_outputs=32,
				kernel_size=[3,3],stride=[1,1],padding='VALID')
			self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
				inputs=self.conv1,num_outputs=64,
				kernel_size=[3,3],stride=[1,1],padding='VALID')
			hidden = slim.fully_connected(slim.flatten(self.conv2), h_size, activation_fn=tf.nn.elu)
			#self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
			#    inputs=self.conv2,num_outputs=64,
			#    kernel_size=[3,3],stride=[1,1],padding='VALID')
			#hidden = slim.fully_connected(slim.flatten(self.conv3), h_size, activation_fn=tf.nn.elu)

			#Recurrent network for temporal dependencies
			#CAREFUL :
			#	- self.state_init
			#	- self.state_in
			# - self.state_out
			# PLACEHOLDER :
			#	- c_in
			# - h_in
			self.rec = rec
			if self.rec :
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(h_size,state_is_tuple=True)
				c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
				h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
				self.state_init = [c_init, h_init]
				#PLACEHOLDER :
				c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
				#
				#PLACEHOLDER :
				h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
				#
				self.state_in = (c_in, h_in)
				rnn_in = tf.expand_dims(hidden, [0])
				step_size = tf.shape(self.imageIn)[:1]
				state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
				lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
					lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
					time_major=False)
				lstm_c, lstm_h = lstm_state
				self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
				rnn_out = tf.reshape(lstm_outputs, [-1, h_size])
			else :
				rnn_out = hidden

			#Output layers for policy and value estimations
			self.policy = slim.fully_connected(rnn_out, a_size, activation_fn=None, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
			self.Vvalue = slim.fully_connected(rnn_out,1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)						              
			self.actions = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
			actionadvantage = slim.fully_connected(self.actions, 10*a_size,	activation_fn=None, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
			self.Qvalue = slim.fully_connected(actionadvantage+self.Vvalue,1,activation_fn=None,weights_initializer=normalized_columns_initializer(0.01),biases_initializer=None)
			#print(self.value.get_shape().as_list())
			#Only the worker network need ops for loss functions and gradient updating.
			if scope != 'global':
				#PLACEHOLDER :
				self.target_qvalue = tf.placeholder(shape=[None],dtype=tf.float32)
				#
				#Gradients :
				qreshaped = tf.reshape(self.Qvalue,[-1])
				self.Qvalue_loss = 0.5 * tf.reduce_sum(tf.square(self.target_qvalue - qreshaped))
				self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
				#self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
				#self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
				self.policy_loss = -tf.reduce_sum(self.Qvalue)
				self.loss = 0.5 * self.Qvalue_loss + self.policy_loss - self.entropy * 0.01

				#Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
				#local_vars = tf.local_variables()
				self.gradients = tf.gradients(self.loss,local_vars)
				self.var_norms = tf.global_norm(local_vars)
				grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
				
				#Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				#global_vars = tf.trainable_variables()
				self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))



class Worker():
	def __init__(self,game,replayBuffer,name,imagesize,s_size,h_size, a_size,trainer,model_path,global_episodes,rec=False,updateT=100,nbrStepPerReplay=100):
		self.name = "worker_" + str(name)
		self.number = name        
		self.model_path = model_path
		self.trainer = trainer
		self.global_episodes = global_episodes
		self.increment = self.global_episodes.assign_add(1)
		self.episode_rewards = []
		self.episode_lengths = []
		self.episode_mean_values = []
		self.summary_writer = tf.summary.FileWriter(self.model_path+"/training_logs/train_"+str(self.number))

		#Create the local copy of the network and the tensorflow op to copy global paramters to local network
		self.rec = rec
		self.updateT = updateT
		self.local_AC = AC_Network(imagesize,s_size,h_size,a_size,self.name,trainer,self.rec)
		self.update_local_ops = update_target_graph('global',self.name)        

		#self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
		self.actions = np.identity(a_size,dtype=np.float32).tolist()
		self.env = game
		self.rBuffer = replayBuffer
		self.nbrStepPerReplay = nbrStepPerReplay
        
	def train(self,rollout,sess,gamma,bootstrap_value):
		rollout = np.array(rollout)
		observations = rollout[:,0]
		actions = rollout[:,1]
		rewards = rollout[:,2]
		next_observations = rollout[:,3]
		values = rollout[:,5]

		self.target_qvalue = rewards+gamma*bootstrap_value

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		vobs = np.vstack(observations)
		#print(discounted_rewards.shape,vobs.shape)
		if self.rec :
			rnn_state = self.local_AC.state_init
			feed_dict = {self.local_AC.target_qvalue:self.target_qvalue,
				self.local_AC.inputs:vobs,
				self.local_AC.actions:actions,
				self.local_AC.state_in[0]:rnn_state[0],
				self.local_AC.state_in[1]:rnn_state[1]}
		else :
			feed_dict = {self.local_AC.target_qvalue:self.target_qvalue,
				self.local_AC.inputs:vobs,
				self.local_AC.actions:actions}
			v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.Qvalue_loss,
				self.local_AC.policy_loss,
				self.local_AC.entropy,
				self.local_AC.grad_norms,
				self.local_AC.var_norms,
				self.local_AC.apply_grads],
				feed_dict=feed_dict)
		
		return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
		    
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		total_steps = 0
		dummy_action = np.zeros(a_size)
		print ("Starting worker " + str(self.number))
		make_gif_log = False
		with sess.as_default(), sess.graph.as_default():                 
			#Let us first synchronize this worker with the global network :
			if self.number != 0:
				sess.run(self.update_local_ops)
				print('Worker synchronized...')
			while not coord.should_stop():
				episode_buffer = []
				episode_values = []
				episode_frames = []
				episode_reward = 0
				episode_step_count = 0
				d = False

				#Let us start a new episode :
				if self.number == 0 :
					print('MAIN AGENT : initializing episode...')
					if not useGAZEBO :
						s = self.env.reset()
						self.env.render()
						s = process_frame(s)
					else :
						s,dr,ddone,_ = envstep(self.env,dummy_action)
						s = preprocess(s, img_size[0], img_size[1] )
						
					episode_frames.append(s)
					
					if self.rec :
						rnn_state = self.local_AC.state_init
				
					remainingSteps = max_episode_length      
					while d == False :
						remainingSteps -= 1
						#Take an action using probabilities from policy network output.
						if self.rec :
							rnn_state_q = rnn_state
							a,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.Vvalue,self.local_AC.Qvalue,self.local_AC.state_out], 
								feed_dict={self.local_AC.inputs:s,
								self.local_AC.state_in[0]:rnn_state[0],
								self.local_AC.state_in[1]:rnn_state[1]})
							q = sess.run([self.local_AC.Qvalue], 
								feed_dict={self.local_AC.inputs:s,
								self.local_AC.state_in[0]:rnn_state_q[0],
								self.local_AC.state_in[1]:rnn_state_q[1],
								self.local_AC.actions:a})
						else :
							a,v = sess.run([self.local_AC.policy,self.local_AC.Vvalue], 
								feed_dict={self.local_AC.inputs:s})
							q = sess.run([self.local_AC.Qvalue], 
								feed_dict={self.local_AC.inputs:s,
								self.local_AC.actions:a})
						
						a= a[0]		
						rospy.loginfo('ACTION :')
						rospy.loginfo(a)
						logfile = open('./logfile.txt', 'w+')
						logfile.write('episode:{} / step:{} / action:'.format(episode_count,remainingSteps)+str(a)+'\n')
						logfile.close()


						if useGAZEBO :
							s1, r, d, _ = envstep(self.env, a)
						else :
							s1, r, d, _ = self.env.step(a)
							self.env.render()

						if d == False:
							episode_frames.append(s1)
							if useGAZEBO :
								s1 = preprocess(s1, img_size[0], img_size[1] )
							else :
								s1 = process_frame(s1)
						else:
							s1 = s
					
						episode_buffer.append([s,a,r,s1,d,v[0,0]])
						episode_values.append(v[0,0])
					
						episode_reward += r
						s = s1                    
						total_steps += 1
						episode_step_count += 1
					
						'''
						# If the episode hasn't ended, but the experience buffer is full, then we
						# make an update step using that experience rollout.
						if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
							# Since we don't know what the true final return is, we "bootstrap" from our current
							# value estimation.
							if self.rec :
								q1 = sess.run(self.local_AC.Qvalue, 
										feed_dict={self.local_AC.inputs:[s1],
										self.local_AC.state_in[0]:rnn_state[0],
										self.local_AC.state_in[1]:rnn_state[1],
										self.local_AC.actions:a})[0,0]
							else :
								q1 = sess.run(self.local_AC.Qvalue, 
										feed_dict={self.local_AC.inputs:[s1],
										self.local_AC.actions:a})[0,0]
	
							v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,q1)
							episode_buffer = []
						'''    
					
						if remainingSteps < 0 :
							d = True
						if d == True:
							break
					#END OF EPISODE WHILE LOOP...
		      
					self.episode_rewards.append(episode_reward)
					self.episode_lengths.append(episode_step_count)
					self.episode_mean_values.append(np.mean(episode_values))

					#Let us add this episode_buffer to the replayBuffer :
					self.rBuffer.append(episode_buffer)
					if len(self.rBuffer) > maxReplayBufferSize :
						self.rBuffer.pop()

					# Periodically save gifs of episodes, model parameters, and summary statistics.
					if episode_count % 5 == 0 and episode_count != 0:
						if self.name == 'worker_0' and episode_count % 25 == 0:
							time_per_step = 0.05
							images = np.array(episode_frames)
							if make_gif_log :
								make_gif(images,'./frames/image'+str(episode_count)+'.gif',
									duration=len(images)*time_per_step,true_image=True,salience=False)
					if episode_count % 250 == 0 and self.name == 'worker_0':
						saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
						print ("Saved Model")

					mean_reward = np.mean(self.episode_rewards[-5:])
					mean_length = np.mean(self.episode_lengths[-5:])
					mean_value = np.mean(self.episode_mean_values[-5:])
					summary = tf.Summary()
					summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
					summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
					summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
					summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
					self.summary_writer.add_summary(summary, episode_count)

					self.summary_writer.flush()
				
					sess.run(self.increment)

				#END OF IF SELF.NUMBER == 0
				else :
					# Update the network using the experience replay buffer:
					if len(self.rBuffer) != 0:
						idxEpisode = np.random.randint(len(self.rBuffer))
						maxIdxStep = len(self.rBuffer[idxEpisode])-1
						idxSteps = np.random.randint(maxIdxStep,size=min(maxIdxStep,self.nbrStepPerReplay) )
						rollout = self.rBuffer[idxEpisode][idxSteps]
						s1 = rollout[:,3]
						# Since we don't know what the true final return is, we "bootstrap" from our current
						# value estimation.
						# TODO : bootstrap from the target network :
						if self.rec :
							a = sess.run(self.local_AC.policy,
							feed_dict={self.local_AC.inputs:s1,
							self.local_AC.state_in[0]:rnn_state[0],
							self.local_AC.state_in[1]:rnn_state[1]})[0,0]
							q1 = sess.run(self.local_AC.Qvalue, 
							feed_dict={self.local_AC.inputs:s1,
							self.local_AC.state_in[0]:rnn_state[0],
							self.local_AC.state_in[1]:rnn_state[1],
							self.local_AC.actions:a})[0,0]
						else :
							a = sess.run(self.local_AC.policy,
							feed_dict={self.local_AC.inputs:s1})[0,0]
							q1 = sess.run(self.local_AC.Qvalue, 
							feed_dict={self.local_AC.inputs:s1,
							self.local_AC.actions:a})[0,0]
					
						v_l,p_l,e_l,g_n,v_n = self.train( rollout,sess,gamma,q1)

						#Let us update the global network :
						if episode_count % self.updateT == 0 :
							sess.run(self.update_local_ops)

				episode_count += 1
        


tf.reset_default_graph()


#with tf.device("/cpu:0"): 
#with tf.device("/gpu:0"): 
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=lr)
master_network = AC_Network(imagesize,s_size,h_size,a_size,'global',None,rec=rec) # Generate global network
#num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
workers = []
replayBuffer = []
# Create worker classes
for i in range(num_workers):
	game = None
	if useGAZEBO :
		game = env
	else :
		if i == 0 :
			game = gym.make('Pendulum-v0')
	workers.append(Worker(game,replayBuffer,i,imagesize,s_size,h_size,a_size,trainer,model_path,global_episodes,rec,updateT,nbrStepsPerReplay))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	if load_model == True:
		print ('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver.restore(sess,ckpt.model_checkpoint_path)
	else:
		sess.run(tf.global_variables_initializer())
	
	print('MODEL INITIALIZED....')
	
	# This is where the asynchronous magic happens.
	# Start the "work" process for each worker in a separate threat.
	worker_threads = []
	for worker in workers:
		worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
		t = threading.Thread(target=(worker_work))
		t.start()
		sleep(0.5)
		worker_threads.append(t)
	coord.join(worker_threads)

  
env.close()

