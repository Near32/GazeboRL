# # Reinforcement Learning : DDPG-A2C
## TODO : implement the target network trick ?

useGAZEBO = True

show = False
load_model = False

import threading
import multiprocessing
import numpy as np

if useGAZEBO :
	from GazeboRL import GazeboRL, Swarm1GazeboRL, init_roscore
	import time
	import rospy
	from Agent1 import NN, INPUT_SHAPE_R, resize, rgb2yuv, BNlayer
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
	#dropoutK = 0.5
	dropoutK=1.0
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
	LoopRate = rospy.Rate(50)

img_size = (84,84,1)
if useGAZEBO :
	img_size = (180,320,3)

rec = False
# In[35]:

maxReplayBufferSize = 100
max_episode_length = 100
updateT = 1
updateTau = 5e-3
nbrStepsPerReplay = 32
gamma = .99 # discount rate for advantage estimation and reward discounting
imagesize = [img_size[0],img_size[1], img_size[2] ]
s_size = imagesize[0]*imagesize[1]*imagesize[2]
h_size = 256

a_size = 1
model_path = './model-RL-Pendulum'
eps_greedy_prob = 0.3
if useGAZEBO :
	a_size = 2	
	#model_path = './model-RL3-GazeboRL-robot1swarm'
	#model_path = './model-GazeboRL-robot1swarm+ExplorationNoise'
	#model_path = './model-GazeboRL-robot1swarm+ExplorationNoise005'
	
	# ACTUAL RESULT : but the reward is maximized... get read of the minus...
	#model_path = './model-GazeboRL-robot1swarm+ExplorationNoise01'
	#model_path = './model-GazeboRL-robot1swarm+ExplorationNoise02'
	model_path = './DDPG-BA2C-r1s+EN01'
	



num_workers = 2
lr=1e-3

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
		outimg = np.zeros(shape=img_size)
	
	if output[1] is not None :
		outr = output[1]['/RL/reward'].data
	
	if output[2] is not None :
		outdone = output[2]
		
	if output[3] is not None :
		outinfo = output[3]
		
	return outimg, outr, outdone, outinfo
	
	




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
        op_holder.append(to_var.assign(updateTau*from_var+(1-updateTau)*to_var))
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
	def __init__(self,imagesize,s_size,h_size, a_size,scope,trainer,rec=False,dropoutK=1.0):
		self.imagesize = imagesize
		self.s_size = s_size
		self.h_size = h_size
		self.a_size = a_size
		self.scope = scope
		self.trainer = trainer
		self.rec = rec
		self.dropoutK = dropoutK
		self.nbrOutput = 256
		
		self.l2_loss = tf.constant(0.0)
		self.lambda_regL2 = 0.0	
		
		self.summary_ops = []
		
		self.build_model_middle()
		self.build_model_top()
		self.build_loss_functions()
		
		self.summary_ops = tf.summary.merge( self.summary_ops )
		 
	
	
	def weight_variable(self,shape, name=None):
		  initial = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
		  self.l2_loss += tf.nn.l2_loss(initial)
		  return initial

	def bias_variable(self,shape):
		  initial = tf.constant(1e-3, shape=shape)
		  var = tf.Variable(initial)
		  return var

	def variable_summaries(self,var, name):
		  """Attach a lot of summaries to a Tensor."""
		  with tf.name_scope('summaries'):
		      #mean = tf.reduce_mean(var)
		      #self.summary_ops.append( tf.summary.scalar('mean/' + name, mean) )
		      #with tf.name_scope('stddev'):
		      #    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		      #self.summary_ops.append( tf.summary.scalar('stddev/' + name, stddev) )
		      #self.summary_ops.append( tf.summary.scalar('max/' + name, tf.reduce_max(var)) )
		      #self.summary_ops.append( tf.summary.scalar('min/' + name, tf.reduce_min(var)) )
		      self.summary_ops.append( tf.summary.histogram(name, var) )

	def nn_layer(self,input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		"""Reusable code for making a simple neural net layer.
		It does a matrix multiply, bias add, and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = self.weight_variable([input_dim, output_dim])
				self.variable_summaries(weights, layer_name + '/weights')
			with tf.name_scope('biases'):
				biases = self.bias_variable([output_dim])
				self.variable_summaries(biases, layer_name + '/biases')
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			activations = act(preactivate, name='activation')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/activations', activations) )
			return activations
	
	def nn_layerBN(self,input_tensor, input_dim, output_dim, phase, layer_name, act=tf.nn.relu):
		"""Reusable code for making a simple neural net layer.
		It does a matrix multiply, bias add, and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = self.weight_variable([input_dim, output_dim])
				self.variable_summaries(weights, layer_name + '/weights')
			with tf.name_scope('biases'):
				biases = self.bias_variable([output_dim])
				self.variable_summaries(biases, layer_name + '/biases')
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor, weights) + biases
				#preactivate = tf.contrib.layers.batch_norm(preactivate, center=True, scale=True, reuse=True, is_training=phase,scope=layer_name+'/bn')
				preactivate = BNlayer(preactivate, is_training=phase, scope=layer_name+'/bn')
				self.summary_ops.append( tf.summary.histogram(layer_name + '/pre_activations', preactivate) )
			activations = act(preactivate, name='activation')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/activations', activations) )
			return activations
		
	def nn_layer_actMaxpoolConv2dDivide2(self,input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2):
		"""Reusable code for making a conv-pool-act neural net layer that divise the width and height by 2 with default parameters.
		
		It does a conv, a max_pool, a matrix multiply, bias add, and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			
			# Variable handling :
			with tf.name_scope('weights'):
				weights = self.weight_variable([filter_size, filter_size, input_dim[3], output_dim[0]])
				self.variable_summaries(weights, layer_name + '/weights')
			with tf.name_scope('biases'):
				biases = self.bias_variable([output_dim[0]])
				self.variable_summaries(biases, layer_name + '/biases')
			
			# Convolution :	
			conv = tf.nn.conv2d(input_tensor, weights, [1, stride, stride, 1], padding='SAME')
			# Max Pooling :
			maxpool = tf.nn.max_pool(conv, [1,pooldim,pooldim,1],[1,poolstride,poolstride,1], padding='SAME')
			
			#hidden = tf.nn.relu(maxpool + layer1_biases)
			#return hidden
			# This Variable will hold the state of the weights for the layer
			
			with tf.name_scope('maxpool_conv_Wx_plus_b'):
				preactivate = maxpool+biases
				self.summary_ops.append( tf.summary.histogram(layer_name + '/pre_activations', preactivate) )
			
			# Activation :
			activations = act(preactivate, name='activation')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/activations', activations) )
			
			# size handling :
			batch_size = input_dim[0]
			pad = 1 # ' SAME ' 
			H = 1 + (input_dim[1]+2*pad-filter_size)/poolstride
			W = 1 + (input_dim[2]+2*pad-filter_size)/poolstride
			out_dim = [ batch_size, H, W, output_dim[0] ]
			
			
			return activations, out_dim
			
	def layer_conv2dBNAct(self,input_tensor, input_dim, output_dim, phase, layer_name='conv2dBNAct', act=tf.identity, filter_size=3, stride=1, padding='SAME'):
		"""Reusable code for making a conv-BN-act neural net layer that give the same size output with default parameters.
		
		It does a conv, add biases,batch normalize and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			
			# Variable handling :
			with tf.name_scope('weights'):
				weights = self.weight_variable([filter_size, filter_size, input_dim[3], output_dim[0]])
				self.variable_summaries(weights, layer_name + '/weights')
			with tf.name_scope('biases'):
				biases = self.bias_variable([output_dim[0]])
				self.variable_summaries(biases, layer_name + '/biases')
			
			# Convolution :	
			conv = tf.nn.conv2d(input_tensor, weights, [1, stride, stride, 1], padding=padding)
			#preactivate = tf.contrib.layers.batch_norm(conv+biases, center=True, scale=True, reuse=True, is_training=phase,scope=layer_name+'/bn')
			preactivate = BNlayer(conv+biases, is_training=phase, scope=layer_name+'/bn')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/pre_activations', preactivate) )
			
			# Activation :
			activations = act(preactivate, name='activation')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/activations', activations) )
			
			# size handling :
			batch_size = input_dim[0]
			pad = 1 # ' SAME ' 
			if padding == 'VALID' : pad = 0 # ' VALID ' 
			
			H = 1 + (input_dim[1]+2*pad-filter_size)/stride
			W = 1 + (input_dim[2]+2*pad-filter_size)/stride
			out_dim = [ batch_size, H, W, output_dim[0] ]
			
			print("layer : "+layer_name+"/conv : input : batch x {}x{}x{} // batch x {}x{}x{}".format(input_dim[1],input_dim[2],input_dim[3],H,W,output_dim[0]))
			
			return activations, out_dim
			
	def layer_maxpoolBNAct(self,input_tensor, input_dim, phase, layer_name='maxpoolBNAct', act=tf.nn.relu, pool_size=2, poolstride=2,padding='SAME'):
		"""Reusable code for making a pool-BN-act neural net layer that divise the width and height by 2 with default parameters.
		
		It does a max_pool, batch normalize and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			
			# Max Pooling :
			maxpool = tf.nn.max_pool(input_tensor, [1,pool_size,pool_size,1],[1,poolstride,poolstride,1], padding=padding)
			
			with tf.name_scope('maxpool'):
				preactivate = maxpool
				#preactivate = tf.contrib.layers.batch_norm(maxpool, center=True, scale=True, reuse=True, is_training=phase,scope=layer_name+'/bn')
				self.summary_ops.append( tf.summary.histogram(layer_name + '/pre_activations', preactivate) )
			
			# Activation :
			activations = act(preactivate, name='activation')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/activations', activations) )
			
			# size handling :
			batch_size = input_dim[0]
			pad = 1 # ' SAME ' 
			if padding == 'VALID' : pad = 0 # ' VALID ' 
			
			H = 1 + (input_dim[1]+2*pad-pool_size)/poolstride
			W = 1 + (input_dim[2]+2*pad-pool_size)/poolstride
			out_dim = [ batch_size, H, W, input_dim[3] ]
			
			
			print("layer : "+layer_name+"/MaxPool : input : batch x {}x{}x{} // batch x {}x{}x{}".format(input_dim[1],input_dim[2],input_dim[3],H,W,input_dim[3]))
			
			return activations, out_dim
							
	def nn_layer_conv2dMaxpoolBNAct(self,input_tensor, input_dim, output_dim, phase, layer_name='conv2dMaxpoolBNAct', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2, convpadding='VALID', poolpadding='SAME'):
		"""Reusable code for making a conv-pool-BN-act neural net layer that divise the width and height by 2 with default parameters.
		
		It does a conv, a max_pool, bias add,batch normalize and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			
			# Variable handling :
			with tf.name_scope('weights'):
				weights = self.weight_variable([filter_size, filter_size, input_dim[3], output_dim[0]])
				self.variable_summaries(weights, layer_name + '/weights')
			with tf.name_scope('biases'):
				biases = self.bias_variable([output_dim[0]])
				self.variable_summaries(biases, layer_name + '/biases')
			
			# Convolution :	
			conv = tf.nn.conv2d(input_tensor, weights, [1, stride, stride, 1], padding=convpadding)
			# Max Pooling :
			maxpool = tf.nn.max_pool(conv, [1,pooldim,pooldim,1],[1,poolstride,poolstride,1], padding=poolpadding)
			
			#hidden = tf.nn.relu(maxpool + layer1_biases)
			#return hidden
			# This Variable will hold the state of the weights for the layer
			
			with tf.name_scope('maxpool_conv_Wx_plus_b'):
				preactivate = maxpool+biases
				#preactivate = tf.contrib.layers.batch_norm(preactivate, center=True, scale=True, reuse=True, is_training=phase,scope=layer_name+'/bn')
				preactivate = BNlayer(preactivate, is_training=phase, scope=layer_name+'/bn')
				self.summary_ops.append( tf.summary.histogram(layer_name + '/pre_activations', preactivate) )
			
			# Activation :
			activations = act(preactivate, name='activation')
			self.summary_ops.append( tf.summary.histogram(layer_name + '/activations', activations) )
			
			# size handling :
			batch_size = input_dim[0]
			convpad = 1 # ' SAME '
			if convpadding == 'VALID' :	convpad = 0
			convH = 1 + (input_dim[1]+2*convpad-filter_size)/stride
			convW = 1 + (input_dim[2]+2*convpad-filter_size)/stride
			conv_out_dim = [ batch_size, convH, convW, output_dim[0] ]
			
			pad = 1 #' SAME '
			if poolpadding == 'VALID' :	pad = 0 #' VALID '
			
			poolH = 1 + (convH+2*pad-pooldim)/poolstride
			poolW = 1 + (convW+2*pad-pooldim)/poolstride
			out_dim = [ batch_size, poolH, poolW, output_dim[0] ]
			
			
			return activations, out_dim
			
	def layer_conv2dBNMaxpoolBNAct(self,input_tensor, input_dim, output_dim, phase, layer_name='conv2dBNMaxpoolBNAct', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2, convpadding='SAME', poolpadding='VALID'):
		"""Reusable code for making a conv-pool-BN-act neural net layer that divise the width and height by 2 with default parameters.
		
		It does a conv, a max_pool, bias add,batch normalize and then uses relu to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			
			# Convolution+BN :	
			conv, out_dim = self.layer_conv2dBNAct(input_tensor=input_tensor, input_dim=input_dim, output_dim=output_dim, phase=phase, layer_name=layer_name, act=tf.identity, filter_size=filter_size, stride=stride,padding=convpadding)
			# Max Pooling :
			actmaxpool, out_dim = self.layer_maxpoolBNAct(input_tensor=conv, input_dim=out_dim, phase=phase, layer_name=layer_name, act=tf.nn.relu, pool_size=pooldim, poolstride=poolstride,padding=poolpadding)
			
			return actmaxpool, out_dim		
			
			
			
			
			
			
			
			
			
			
	def build_model_middle(self) :
		with tf.variable_scope(self.scope):
			# DROPOUT + BATCH NORMALIZATION :
			self.keep_prob = tf.placeholder(tf.float32)
			self.phase = tf.placeholder(tf.bool,name='phase')
			self.summary_ops.append( tf.summary.scalar('dropout_keep_probability', self.keep_prob) )
		
			#Input and visual encoding layers
			#PLACEHOLDER :
			self.inputs = tf.placeholder(shape=[None,self.s_size],dtype=tf.float32,name='inputs')
			#
			self.imageIn = tf.reshape(self.inputs,shape=[-1,self.imagesize[0],self.imagesize[1],self.imagesize[2]])
			
			# CONV LAYER 1 :
			shape_input = self.imageIn.get_shape().as_list()
			input_dim1 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
			nbr_filter1 = 32
			output_dim1 = [ nbr_filter1]
			relumaxpoolconv1, input_dim2 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=self.imageIn, input_dim=input_dim1, output_dim=output_dim1, phase=self.phase, layer_name='conv0MaxPool0', act=tf.nn.relu, filter_size=5, stride=3, pooldim=2, poolstride=2)
			rmpc1_do = tf.nn.dropout(relumaxpoolconv1,self.keep_prob)
		
			#LAYER STN 1 :
			#shape_inputstn = rmpc1_do.get_shape().as_list()
			#shape_inputstn = self.x_tensor.get_shape().as_list()
			#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
			#layerstn_name = 'stn1'
			#h_trans_def1, out_size1, self.thetas1 = self.nn_layer_stn( rmpc1_do, inputstn_dim, layerstn_name, self.keep_prob)
			#h_trans_def1, out_size1, self.thetas1 = self.nn_layer_stn( self.x_tensor, inputstn_dim, layerstn_name, self.keep_prob)
			#shape_input = h_trans_def1.get_shape().as_list()
			#input_dim2 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
			# CONV LAYER 2 :
			nbr_filter2 = 32
			output_dim2 = [ nbr_filter2]
			relumaxpoolconv2, input_dim3 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc1_do, input_dim=input_dim2, output_dim=output_dim2, phase=self.phase, layer_name='conv1MaxPool1', act=tf.nn.relu, filter_size=3, stride=2, pooldim=2, poolstride=2)
			rmpc2_do = tf.nn.dropout(relumaxpoolconv2,self.keep_prob)
		
			#LAYER STN 2 :
			#shape_inputstn = rmpc2_do.get_shape().as_list()
			#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
			#layerstn_name = 'stn2'
			#h_trans_def2, out_size2, self.thetas2 = self.nn_layer_stn( rmpc2_do, inputstn_dim, layerstn_name, self.keep_prob)
			#shape_input = h_trans_def2.get_shape().as_list()
			#input_dim3 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
			# CONV LAYER 3 :
			nbr_filter3 = 32
			output_dim3 = [ nbr_filter3]
			relumaxpoolconv3, input_dim4 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc2_do, input_dim=input_dim3, output_dim=output_dim3, phase=self.phase, layer_name='conv2MaxPool2', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
			rmpc3_do = tf.nn.dropout(relumaxpoolconv3,self.keep_prob)
		
			#LAYER STN 3 :
			#shape_inputstn = rmpc3_do.get_shape().as_list()
			#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
			#layerstn_name = 'stn3'
			#h_trans_def3, out_size3, self.thetas3 = self.nn_layer_stn( rmpc3_do, inputstn_dim, layerstn_name, self.keep_prob)
			#shape_input = h_trans_def3.get_shape().as_list()
			#input_dim4 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
			# CONV LAYER 4 :
			'''
			nbr_filter4 = 128
			output_dim4 = [ nbr_filter4]
			relumaxpoolconv4, input_dim5 = self.layer_conv2dBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv3', act=tf.nn.relu, filter_size=3, stride=1,padding='SAME')
			rmpc4_do = tf.nn.dropout(relumaxpoolconv4,self.keep_prob)		
			'''
			
			shape_conv = rmpc3_do.get_shape().as_list()
			shape_fc = [-1, shape_conv[1]*shape_conv[2]*shape_conv[3] ]
			out1 = 256
			fc_x_input = tf.reshape( rmpc3_do, shape_fc )
			hidden1 = self.nn_layerBN(fc_x_input, shape_fc[1], out1, self.phase, 'layer1')
			dropped1 = tf.nn.dropout(hidden1, self.keep_prob)
			
			out2 = 256
			hidden2 = self.nn_layerBN(dropped1, out1, out2, self.phase,'layer2')
			dropped2 = tf.nn.dropout(hidden2, self.keep_prob)
			
			self.y = self.nn_layer(dropped2, out2, self.nbrOutput, 'layerOutput', act=tf.identity)	
				
		
		
		
		
		
		
	def build_model_top(self) :
		with tf.variable_scope(self.scope):
			hidden = self.y

			#Recurrent network for temporal dependencies
			#CAREFUL :
			#	- self.state_init
			#	- self.state_in
			# - self.state_out
			# PLACEHOLDER :
			#	- c_in
			# - h_in
			
			if self.rec :
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.h_size,state_is_tuple=True)
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
				rnn_out = tf.reshape(lstm_outputs, [-1, self.h_size])
			else :
				rnn_out = hidden
				
			shape_out = rnn_out.get_shape().as_list()

			
			#Output layers for policy and value estimations
			#self.policy = slim.fully_connected(rnn_out, self.a_size, activation_fn=None, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
			self.policy = self.nn_layer(rnn_out, shape_out[1], self.a_size, 'policy', act=tf.identity)	
			#self.Vvalue = slim.fully_connected(rnn_out,1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)						              
			self.Vvalue = self.nn_layer(rnn_out, shape_out[1], 1, 'V-value', act=tf.identity)	
			
			#PLACEHOLDER :
			self.actions = tf.placeholder(shape=[None,self.a_size],dtype=tf.float32,name='actions')
			#
			vvalueadvantage = self.nn_layerBN(rnn_out, shape_out[1], self.nbrOutput, self.phase, 'vvalue-advantage')
			
			concat = tf.concat( [vvalueadvantage, self.actions], axis=1,name='concat-Vvalue-actions')
			concat_shape = concat.get_shape().as_list()
			#self.Qvalue = slim.fully_connected(actionadvantage+self.Vvalue,1,activation_fn=None,weights_initializer=normalized_columns_initializer(0.01),biases_initializer=None)
			#self.Qvalue = self.nn_layer(actionadvantage+vvalueadvantage, self.nbrOutput, 1, 'Q-value', act=tf.identity)	
			
			hidden = self.nn_layerBN(concat, concat_shape[1], self.nbrOutput, self.phase, 'Q-value-hidden', act=tf.nn.relu)	
			self.Qvalue = self.nn_layer(hidden, self.nbrOutput, 1, 'Q-value', act=tf.identity)	
			# use with 01 :
			#self.Qvalue = self.nn_layer(policyadvantage+vvalueadvantage, self.nbrOutput, 1, 'Q-value', act=tf.identity)	
			
			#self.Qvalue_policy = self.nn_layer(policyadvantage+vvalueadvantage, self.nbrOutput, 1, 'Q-value-policy', act=tf.identity)	
			#print(self.value.get_shape().as_list())	
		
		
		
		
		
		
		
		
		
	def build_model(self) :
		with tf.variable_scope(self.scope):
			# DROPOUT + BATCH NORMALIZATION :
			self.keep_prob = tf.placeholder(tf.float32)
			self.phase = tf.placeholder(tf.bool,name='phase')
			self.summary_ops.append( tf.summary.scalar('dropout_keep_probability', self.keep_prob) )
		
			#Input and visual encoding layers
			#PLACEHOLDER :
			self.inputs = tf.placeholder(shape=[None,self.s_size],dtype=tf.float32)
			#
			self.imageIn = tf.reshape(self.inputs,shape=[-1,self.imagesize[0],self.imagesize[1],self.imagesize[2]])
			self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
				inputs=self.imageIn,num_outputs=32,
				kernel_size=[5,5],stride=[3,3],padding='VALID')
			self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
				inputs=self.conv1,num_outputs=32,
				kernel_size=[5,5],stride=[3,3],padding='VALID')
			#hidden = slim.fully_connected(slim.flatten(self.conv2), h_size, activation_fn=tf.nn.elu)
			self.conv3 = slim.conv2d(activation_fn=tf.nn.elu,
			    inputs=self.conv2,num_outputs=64,
			    kernel_size=[3,3],stride=[1,1],padding='VALID')
			hidden = slim.fully_connected(slim.flatten(self.conv3), self.h_size, activation_fn=tf.nn.elu)

			#Recurrent network for temporal dependencies
			#CAREFUL :
			#	- self.state_init
			#	- self.state_in
			# - self.state_out
			# PLACEHOLDER :
			#	- c_in
			# - h_in
			
			if self.rec :
				lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.h_size,state_is_tuple=True)
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
				rnn_out = tf.reshape(lstm_outputs, [-1, self.h_size])
			else :
				rnn_out = hidden

			
			#Output layers for policy and value estimations
			self.policy = slim.fully_connected(rnn_out, self.a_size, activation_fn=None, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
			self.Vvalue = slim.fully_connected(rnn_out,1, activation_fn=None, weights_initializer=normalized_columns_initializer(1.0), biases_initializer=None)						              
			#PLACEHOLDER :
			self.actions = tf.placeholder(shape=[None,self.a_size],dtype=tf.float32)
			#
			actionadvantage = slim.fully_connected(self.actions, 10*self.a_size,	activation_fn=None, weights_initializer=normalized_columns_initializer(0.01), biases_initializer=None)
			self.Qvalue = slim.fully_connected(actionadvantage+self.Vvalue,1,activation_fn=None,weights_initializer=normalized_columns_initializer(0.01),biases_initializer=None)
			#self.Qvalue_policy = slim.fully_connected(self.policy+self.Vvalue,1,activation_fn=None,weights_initializer=normalized_columns_initializer(0.01),biases_initializer=None)
			#print(self.value.get_shape().as_list())
			
			
			
			
			
			
			
			
			
			
			
			
	def build_loss_functions(self):
		with tf.variable_scope(self.scope) :
			#Only the worker network need ops for loss functions and gradient updating.
			if self.scope != 'global':
				#PLACEHOLDER :
				self.target_qvalue = tf.placeholder(shape=[None,1],dtype=tf.float32,name='target_qvalues')
				#
				#Gradients :
				qreshaped = tf.reshape(self.Qvalue,[-1])
				self.Qvalue_loss = 0.5 * tf.reduce_sum(tf.square(self.target_qvalue - qreshaped))
				self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
				#self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
				#self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01
				# MINIMIZATION/MAXIMIZATION OF THE REWARD(PENALTY..) :
				self.policy_loss = -tf.reduce_sum(self.Qvalue)
				#self.policy_loss = -tf.reduce_sum(self.Qvalue_policy)
				self.loss = 0.5*self.Qvalue_loss + self.policy_loss - 0.01*self.entropy + self.lambda_regL2*self.l2_loss

				
				#Get gradients from local network using local losses
				local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
				self.var_norms = tf.global_norm(local_vars)
				'''
				#local_vars = tf.local_variables()
				self.gradients = tf.gradients(self.loss,local_vars)
				grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
				'''
				# PLACEHOLDER :
				self.critic_gradients_action = tf.placeholder(tf.float32,[None,self.a_size],name='critic_gradients_action')#tf.gradients(tf.reduce_sum(self.Qvalue),self.actions)
				self.critic_gradients_action_op = tf.gradients(self.Qvalue,self.actions)
				self.actor_gradients = tf.gradients(self.policy,local_vars,-self.critic_gradients_action)
				
				self.critic_gradients = tf.gradients(self.Qvalue_loss,local_vars)
				actor_grads,self.actor_grad_norms = tf.clip_by_global_norm(self.actor_gradients,40.0)
				critic_grads,self.critic_grad_norms = tf.clip_by_global_norm(self.critic_gradients,40.0)
				self.grad_norms = self.actor_grad_norms + self.critic_grad_norms
				
				#Apply local gradients to global network
				global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
				#global_vars = tf.trainable_variables()
				#self.apply_grads = self.trainer.apply_gradients(zip(grads,global_vars))
				print(len(global_vars))
				print(len(critic_grads))
				print(len(actor_grads))
				self.apply_grads = { 'critic':self.trainer['critic'].apply_gradients(zip(critic_grads,global_vars)), 'actor':self.trainer['actor'].apply_gradients(zip(actor_grads,global_vars)) }























class Worker():
	def __init__(self,master_network,game,replayBuffer,name,imagesize,s_size,h_size, a_size,trainer,model_path,global_episodes,rec=False,updateT=100,nbrStepPerReplay=100):
		self.master_network = master_network
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
		self.test_summary_writer = tf.summary.FileWriter(self.model_path+"/training_logs/train_test"+str(self.number))

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
		observations = np.vstack(rollout[:,0]) #np.reshape( rollout[:,0], newshape=(-1,s_size) )
		actions = np.vstack(rollout[:,1]) #np.reshape( rollout[:,1], newshape=(-1,a_size) )
		rewards = np.vstack(rollout[:,2]) #np.reshape( rollout[:,2], newshape=(-1,1) )
		next_observations = np.vstack(rollout[:,3]) #np.reshape( rollout[:,3], newshape=(-1,s_size) )
		values = np.vstack(rollout[:,4]) #np.reshape(rollout[:,4],newshape=(-1,1) )
		
		self.target_qvalue = rewards+gamma*bootstrap_value

		# Update the global network using gradients from loss
		# Generate network statistics to periodically save
		#vobs = np.vstack(observations)
		vobs = observations
		#print(discounted_rewards.shape,vobs.shape)
		if self.rec :
			rnn_state = self.local_AC.state_init
			feed_dict = {self.local_AC.target_qvalue:self.target_qvalue,
				self.local_AC.inputs:vobs,
				self.local_AC.actions:actions,
				self.local_AC.state_in[0]:rnn_state[0],
				self.local_AC.state_in[1]:rnn_state[1],
				self.local_AC.keep_prob:self.local_AC.dropoutK,
				self.local_AC.phase:True}
		else :
			feed_dict = {self.local_AC.inputs:vobs,
				self.local_AC.actions:actions,
				self.local_AC.keep_prob:self.local_AC.dropoutK,
				self.local_AC.phase:False}
			critic_gradients_action = sess.run([self.local_AC.critic_gradients_action_op],
				feed_dict = feed_dict)[0][0]
				
			feed_dict = {self.local_AC.target_qvalue:self.target_qvalue,
				self.local_AC.inputs:vobs,
				self.local_AC.actions:actions,
				self.local_AC.keep_prob:self.local_AC.dropoutK,
				self.local_AC.phase:True,
				self.local_AC.critic_gradients_action:critic_gradients_action}
			v_l,p_l,e_l,g_n,v_n,_ = sess.run([self.local_AC.Qvalue_loss,
				self.local_AC.policy_loss,
				self.local_AC.entropy,
				self.local_AC.grad_norms,
				self.local_AC.var_norms,
				self.local_AC.apply_grads['critic'],
				self.local_AC.apply_grads['actor']],
				feed_dict=feed_dict)
		
		return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
		    
	def work(self,max_episode_length,gamma,sess,coord,saver):
		episode_count = sess.run(self.global_episodes)
		summary_count = 0
		total_steps = 0
		logit = 0
		dummy_action = np.zeros(a_size)
		print ("Starting worker " + str(self.number))
		make_gif_log = False
		with sess.as_default(), sess.graph.as_default():                 
			#Let us first synchronize this worker with the global network :
			if self.number != 0:
				sess.run(self.update_local_ops)
				print('Worker synchronized...')
			while not coord.should_stop():
				try :
					episode_buffer = []
					episode_values = []
					episode_frames = []
					episode_reward = 0
					episode_step_count = 0
					d = False

					#Let us start a new episode :
					if self.number == 0 :
						if not useGAZEBO :
							s = self.env.reset()
							self.env.render()
							s = process_frame(s)
						else :
							s = self.env.reset()
							rospy.loginfo('ENVIRONMENT RESETTED !')
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
									self.local_AC.state_in[1]:rnn_state[1],
									self.local_AC.keep_prob:1.0,
									self.local_AC.phase:False})
								#summary, q = sess.run([self.local_AC.merged_summary, self.local_AC.Qvalue], 
								q = sess.run([ self.local_AC.Qvalue], 
										feed_dict={self.local_AC.inputs:s,
									self.local_AC.state_in[0]:rnn_state_q[0],
									self.local_AC.state_in[1]:rnn_state_q[1],
									self.local_AC.actions:a,
									self.local_AC.keep_prob:1.0,
									self.local_AC.phase:False})
							else :
								
								a,v = sess.run([self.local_AC.policy,self.local_AC.Vvalue], 
									feed_dict={self.local_AC.inputs:s,
									self.local_AC.keep_prob:1.0,
									self.local_AC.phase:False})
								#summary, q = sess.run([self.local_AC.merged_summary, self.local_AC.Qvalue], 
								q = sess.run([self.local_AC.Qvalue], 
									feed_dict={self.local_AC.inputs:s,
									self.local_AC.actions:a,
									self.local_AC.keep_prob:1.0,
									self.local_AC.phase:False})
								#NETWORK-RELATED SUMMARIES :
								summary = sess.run(self.local_AC.summary_ops, 
									feed_dict={self.local_AC.inputs:s,
									self.local_AC.actions:a,
									self.local_AC.keep_prob:1.0,
									self.local_AC.phase:False})
								'''
								a,v = sess.run([self.master_network.policy,self.master_network.Vvalue], 
									feed_dict={self.master_network.inputs:s,
									self.master_network.keep_prob:1.0,
									self.master_network.phase:False})
								#summary, q = sess.run([self.local_AC.merged_summary, self.local_AC.Qvalue], 
								q = sess.run([self.master_network.Qvalue], 
									feed_dict={self.master_network.inputs:s,
									self.master_network.actions:a,
									self.master_network.keep_prob:1.0,
									self.master_network.phase:False})
								#NETWORK-RELATED SUMMARIES :
								summary = sess.run(self.master_network.summary_ops, 
									feed_dict={self.master_network.inputs:s,
									self.master_network.actions:a,
									self.master_network.keep_prob:1.0,
									self.master_network.phase:False})
								'''
						
							self.test_summary_writer.add_summary(summary,summary_count)
							self.test_summary_writer.flush()
							summary_count +=1
						
						
							#logfile = open('./logfile.txt', 'w+')
							if logit > 50 :
								logit = 0
								sentence = 'episode:{} / step:{} / action:'.format(episode_count,remainingSteps)+str(a)
								rospy.loginfo(sentence)
							else :
								logit += 1
							#logfile.write(sentence+'\n')
							#logfile.close()
						
							#EXPLORATION NOISE :
							eps_greedy_prob = 0.3
							if np.random.rand() < eps_greedy_prob :
								scale = 0.1
								a_noise = np.random.normal(loc=0.0,scale=scale,size=a[0].shape)
								a[0] += a_noise

							if useGAZEBO :
								s1, r, d, _ = envstep(self.env, a[0])
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
						
							if remainingSteps < 0 :
								d = True
							if d == True:
								break
							
							LoopRate.sleep()
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
						if episode_count % 5 == 0 and self.name == 'worker_0':
							saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
							print ("Saved Model")

						mean_reward = np.mean(self.episode_rewards[-5:])
						mean_length = np.mean(self.episode_lengths[-5:])
						mean_value = np.mean(self.episode_mean_values[-5:])
						summary = tf.Summary()
						summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
						summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
						summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
						#summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
						#summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
						#summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
						#summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
						#summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
						self.summary_writer.add_summary(summary, episode_count)
						self.summary_writer.flush()
						
						sess.run(self.increment)

					#END OF IF SELF.NUMBER == 0
					else :
						# Update the network using the experience replay buffer:
						if len(self.rBuffer) != 0:
							a1 = None
							idxEpisode = np.random.choice(len(self.rBuffer),1)[0]
							maxIdxStep = len(self.rBuffer[idxEpisode])
							idxSteps = np.random.randint(low=0,high=max(1,maxIdxStep -self.nbrStepPerReplay) )
						
							rollout = np.vstack(self.rBuffer[idxEpisode][int(idxSteps):int(idxSteps+self.nbrStepPerReplay)  ] )
							s1 = np.vstack(rollout[:,3])
							# Since we don't know what the true final return is, we "bootstrap" from our current
							# q value estimation that is done by the local network which acts as a target network for the master network on which we apply the gradients...
							if self.rec :
								a1 = sess.run(self.local_AC.policy,
								feed_dict={self.local_AC.inputs:s1,
								self.local_AC.state_in[0]:rnn_state[0],
								self.local_AC.state_in[1]:rnn_state[1],
								self.local_AC.keep_prob:1.0,
								self.local_AC.phase:False})
								q1 = sess.run(self.local_AC.Qvalue, 
								feed_dict={self.local_AC.inputs:s1,
								self.local_AC.state_in[0]:rnn_state[0],
								self.local_AC.state_in[1]:rnn_state[1],
								self.local_AC.actions:a,
								self.local_AC.keep_prob:1.0,
								self.local_AC.phase:False})[0,0]
							else :
								a1 = sess.run(self.local_AC.policy,
								feed_dict={self.local_AC.inputs:s1,
								self.local_AC.keep_prob:1.0,
								self.local_AC.phase:False})
								q1 = sess.run(self.local_AC.Qvalue, 
								feed_dict={self.local_AC.inputs:s1,
								self.local_AC.actions:a1,
								self.local_AC.keep_prob:1.0,
								self.local_AC.phase:False})[0,0]
								
							v_l,p_l,e_l,g_n,v_n = self.train( rollout,sess,gamma,q1)
							
							'''
							#NETWORK-RELATED SUMMARIES :
							summary = sess.run(self.local_AC.summary_ops, 
								feed_dict={self.local_AC.inputs:s1,
								self.local_AC.actions:a,
								self.local_AC.keep_prob:1.0,
								self.local_AC.phase:False})
						
							self.test_summary_writer.add_summary(summary,summary_count)
							self.test_summary_writer.flush()
							summary_count +=1
							'''
							

							#Let us update the global network :
							if episode_count % self.updateT == 0 :
								sess.run(self.update_local_ops)

					episode_count += 1
		      
				except Exception as e :
					print('EXCEPTION HANDLED : '+str(e))


tf.reset_default_graph()


with tf.device("/cpu:0"): 
#with tf.device("/gpu:0"): 
	global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
	trainer = { 'actor':tf.train.AdamOptimizer(learning_rate=lr*10.0), 'critic':tf.train.AdamOptimizer(learning_rate=lr)}
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
		workers.append(Worker(master_network,game,replayBuffer,i,imagesize,s_size,h_size,a_size,trainer,model_path,global_episodes,rec,updateT,nbrStepsPerReplay))
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

