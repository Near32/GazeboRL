import numpy as np
from utils import INPUT_SHAPE_R, Dataset
import argparse
import tensorflow as tf



nbrinput = INPUT_SHAPE_R
nbroutput = 2
filepath_base = './logs/'
dropoutK = 0.5
batch_size = 1024

useMINI = True


def load_dataset(args) :
	dataset = Dataset(args.data_dir)
	return dataset


def BNlayer(x, is_training, scope):
   bn_train = tf.contrib.layers.batch_norm(x, decay=0.999, center=True, scale=True, updates_collections=None, is_training=True, scope=scope)
   bn_inference = tf.contrib.layers.batch_norm(x, decay=0.999, center=True, scale=True, updates_collections=None, is_training=False, scope=scope, reuse=True)
   bn = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)
   return bn


class NN :
	def __init__(self,filepath_base,nbrinput,nbroutput,lr,filepathin=None) :
		  self.filepathin = filepathin
		  #build_modelMINI
		  #self.filepath = filepath_base+'archiYOLO_'+str(nbrinput[0])+'_'+str(nbrinput[1])+'--'
		  
		  #build_modelMINI1
		  self.filepath = filepath_base+'archiYOLO1_'+str(nbrinput[0])+'_'+str(nbrinput[1])+'--'
		  self.filepath = self.filepath + str(nbroutput) + '-'+str(lr)
		  
		  self.lr = lr
		  self.nbrInput = nbrinput
		  self.nbrOutput = nbroutput
		  
		  self.l2_loss = tf.constant(0.0)
		  self.lambda_regL2 = 0.0
		  print( 'Session is beginning...')
		  # Input placeholders
		  with tf.name_scope('input'):
		      self.x = tf.placeholder(tf.float32, [None, self.nbrInput[0]*self.nbrInput[1]*self.nbrInput[2]], name='x-input')
		      self.y_ = tf.placeholder(tf.float32, [None, self.nbrOutput], name='y-input')

		  print('Model building : ...')
		  if useMINI :
		  	#self.build_modelMINI()
		  	self.build_modelMINI1()
		  else :
		  	self.build_modelYOLO()
		  print('Model building : OK.')
		  print('Model initialization : ...')
		  self.init_model(self.lr)
		  print('Model initialization : OK.')

	def weight_variable(self,shape, name=None):
		  initial = tf.Variable(tf.truncated_normal(shape, stddev=1e-3))
		  self.l2_loss += tf.nn.l2_loss(initial)
		  return initial

	def bias_variable(self,shape):
		  initial = tf.constant(1e-2, shape=shape)
		  var = tf.Variable(initial)
		  return var

	def variable_summaries(self,var, name):
		  """Attach a lot of summaries to a Tensor."""
		  with tf.name_scope('summaries'):
		      mean = tf.reduce_mean(var)
		      tf.summary.scalar('mean/' + name, mean)
		      with tf.name_scope('stddev'):
		          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		      tf.summary.scalar('stddev/' + name, stddev)
		      tf.summary.scalar('max/' + name, tf.reduce_max(var))
		      tf.summary.scalar('min/' + name, tf.reduce_min(var))
		      tf.summary.histogram(name, var)

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
			tf.summary.histogram(layer_name + '/activations', activations)
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
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			activations = act(preactivate, name='activation')
			tf.summary.histogram(layer_name + '/activations', activations)
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
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			
			# Activation :
			activations = act(preactivate, name='activation')
			tf.summary.histogram(layer_name + '/activations', activations)
			
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
			tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			
			# Activation :
			activations = act(preactivate, name='activation')
			tf.summary.histogram(layer_name + '/activations', activations)
			
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
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			
			# Activation :
			activations = act(preactivate, name='activation')
			tf.summary.histogram(layer_name + '/activations', activations)
			
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
				tf.summary.histogram(layer_name + '/pre_activations', preactivate)
			
			# Activation :
			activations = act(preactivate, name='activation')
			tf.summary.histogram(layer_name + '/activations', activations)
			
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
			
	def build_model(self,archi=[128,256,32]) :
		  self.keep_prob = tf.placeholder(tf.float32)
		  self.phase = tf.placeholder(tf.bool,name='phase')
		  tf.summary.scalar('dropout_keep_probability', self.keep_prob)

		  shapein = self.nbrInput
		  layers = list()
		  layers.append(self.x)
		  nbrlayers = len(archi)
		  for l in range(nbrlayers) :
		      nbrout = archi[l]
		      inputl = None
		      
		      inputl = layers[l]
		      
		      print('layer :: {} : inputs::{}::{} ; outputs::{}'.format(l,inputl,shapein,nbrout))
		      
		      layers.append( 
		          tf.nn.dropout(
		              self.nn_layerBN(inputl,shapein,nbrout,self.phase, 'layer'+str(l)),
		              self.keep_prob
		          )
		      )
		      
		      shapein = nbrout
		  
		  self.y = self.nn_layerBN(layers[-1], shapein, self.nbrOutput, self.phase, layer_name='layerOutput', act=tf.identity)
		  
	def build_modelYOLO(self) :
		#with tf.name_scope('dropout'):
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.bool,name='phase')
		tf.summary.scalar('dropout_keep_probability', self.keep_prob)
		
		self.x_tensor = tf.reshape( self.x, [-1, self.nbrInput[0], self.nbrInput[1], self.nbrInput[2]] )
		
		# CONV LAYER 1 :
		shape_input = self.x_tensor.get_shape().as_list()
		input_dim1 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		nbr_filter1 = 32
		output_dim1 = [ nbr_filter1]
		relumaxpoolconv1, input_dim2 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=self.x_tensor, input_dim=input_dim1, output_dim=output_dim1, phase=self.phase, layer_name='conv0MaxPool0', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
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
		#relumaxpoolconv2, input_dim3 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=h_trans_def1, input_dim=input_dim2, output_dim=output_dim2, phase=self.phase, layer_name='conv1MaxPool1', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv2, input_dim3 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc1_do, input_dim=input_dim2, output_dim=output_dim2, phase=self.phase, layer_name='conv1MaxPool1', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		rmpc2_do = tf.nn.dropout(relumaxpoolconv2,self.keep_prob)
		
		#LAYER STN 2 :
		#shape_inputstn = rmpc2_do.get_shape().as_list()
		#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
		#layerstn_name = 'stn2'
		#h_trans_def2, out_size2, self.thetas2 = self.nn_layer_stn( rmpc2_do, inputstn_dim, layerstn_name, self.keep_prob)
		#shape_input = h_trans_def2.get_shape().as_list()
		#input_dim3 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
		# CONV LAYER 3 :
		nbr_filter3 = 64
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
		nbr_filter4 = 128
		output_dim4 = [ nbr_filter4]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv4, input_dim5 = self.layer_conv2dBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv3', act=tf.nn.relu, filter_size=3, stride=1,padding='SAME')
		rmpc4_do = tf.nn.dropout(relumaxpoolconv4,self.keep_prob)
		
		#LAYER STN 4 :
		#shape_inputstn = rmpc4_do.get_shape().as_list()
		#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
		#layerstn_name = 'stn4'
		#h_trans_def4, out_size4, self.thetas4 = self.nn_layer_stn( rmpc4_do, inputstn_dim, layerstn_name, self.keep_prob)
		#shape_input = h_trans_def4.get_shape().as_list()
		#input_dim5 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
		# CONV LAYER 5 :
		#nbr_filter5 = 256
		#output_dim5 = [ nbr_filter5]
		#relumaxpoolconv5, input_dim6 = self.nn_layer_actMaxpoolConv2dDivide2(input_tensor=rmpc4_do, input_dim=input_dim5, output_dim=output_dim5, layer_name='conv5', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		#rmpc5_do = tf.nn.dropout(relumaxpoolconv5,self.keep_prob)
		nbr_filter5 = 64
		output_dim5 = [ nbr_filter5]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv5, input_dim6 = self.layer_conv2dBNAct(input_tensor=rmpc4_do, input_dim=input_dim5, output_dim=output_dim5, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=1, stride=1,padding='VALID')
		rmpc5_do = tf.nn.dropout(relumaxpoolconv5,self.keep_prob)
		
		
		# CONV LAYER 6 :
		nbr_filter6 = 128
		output_dim6 = [ nbr_filter6]
		relumaxpoolconv6, input_dim7 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc5_do, input_dim=input_dim6, output_dim=output_dim6, phase=self.phase, layer_name='conv5MaxPool3', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2,convpadding='SAME',poolpadding='VALID')
		rmpc6_do = tf.nn.dropout(relumaxpoolconv6,self.keep_prob)
		
		# CONV LAYER 7 :
		nbr_filter7 = 256
		output_dim7 = [ nbr_filter7]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv7, input_dim8 = self.layer_conv2dBNAct(input_tensor=rmpc6_do, input_dim=input_dim7, output_dim=output_dim7, phase=self.phase, layer_name='conv6', act=tf.nn.relu, filter_size=3, stride=1, padding='VALID')
		rmpc7_do = tf.nn.dropout(relumaxpoolconv7,self.keep_prob)
		
		
		# CONV LAYER 8 :
		nbr_filter8 = 128
		output_dim8 = [ nbr_filter8]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv8, input_dim9 = self.layer_conv2dBNAct(input_tensor=rmpc7_do, input_dim=input_dim8, output_dim=output_dim8, phase=self.phase, layer_name='conv7', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc8_do = tf.nn.dropout(relumaxpoolconv8,self.keep_prob)
		
		# CONV LAYER 9 :
		nbr_filter9 = 256
		output_dim9 = [ nbr_filter9]
		relumaxpoolconv9, input_dim10 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc8_do, input_dim=input_dim9, output_dim=output_dim9, phase=self.phase, layer_name='conv8MaxPool4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		#relumaxpoolconv8, input_dim9 = self.layer_conv2dBNAct(input_tensor=rmpc7_do, input_dim=input_dim8, output_dim=output_dim8, phase=self.phase, layer_name='conv7', act=tf.nn.relu, filter_size=1, stride=1)
		rmpc9_do = tf.nn.dropout(relumaxpoolconv9,self.keep_prob)
		
		# CONV LAYER 10 :
		nbr_filter10 = 512
		output_dim10 = [ nbr_filter10]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv10, input_dim11 = self.layer_conv2dBNAct(input_tensor=rmpc9_do, input_dim=input_dim10, output_dim=output_dim10, phase=self.phase, layer_name='conv9', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc10_do = tf.nn.dropout(relumaxpoolconv10,self.keep_prob)
		
		# CONV LAYER 11 :
		nbr_filter11 = 256
		output_dim11 = [ nbr_filter11]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv11, input_dim12 = self.layer_conv2dBNAct(input_tensor=rmpc10_do, input_dim=input_dim11, output_dim=output_dim11, phase=self.phase, layer_name='conv10', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc11_do = tf.nn.dropout(relumaxpoolconv11,self.keep_prob)
		
		# CONV LAYER 12 :
		nbr_filter12 = 512
		output_dim12 = [ nbr_filter12]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv12, input_dim13 = self.layer_conv2dBNAct(input_tensor=rmpc11_do, input_dim=input_dim12, output_dim=output_dim12, phase=self.phase, layer_name='conv11', act=tf.nn.relu, filter_size=3, stride=1)
		rmpc12_do = tf.nn.dropout(relumaxpoolconv12,self.keep_prob)
		
		# CONV LAYER 13 :
		nbr_filter13 = 256
		output_dim13 = [ nbr_filter13]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv13, input_dim14 = self.layer_conv2dBNAct(input_tensor=rmpc12_do, input_dim=input_dim13, output_dim=output_dim13, phase=self.phase, layer_name='conv12', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc13_do = tf.nn.dropout(relumaxpoolconv13,self.keep_prob)
		
		# CONV LAYER 14 :
		nbr_filter14 = 512
		output_dim14 = [ nbr_filter14]
		relumaxpoolconv14, input_dim15 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2,convpadding='VALID',poolpadding='SAME')
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13', act=tf.nn.relu, filter_size=3, stride=3)
		rmpc14_do = tf.nn.dropout(relumaxpoolconv14,self.keep_prob)
		
		# CONV LAYER 15 :
		nbr_filter15 = 1024
		output_dim15 = [ nbr_filter15]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv15, input_dim16 = self.layer_conv2dBNAct(input_tensor=rmpc14_do, input_dim=input_dim15, output_dim=output_dim15, phase=self.phase, layer_name='conv14', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc15_do = tf.nn.dropout(relumaxpoolconv15,self.keep_prob)
		
		# CONV LAYER 16 :
		nbr_filter16 = 512
		output_dim16 = [ nbr_filter16]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv16, input_dim17 = self.layer_conv2dBNAct(input_tensor=rmpc15_do, input_dim=input_dim16, output_dim=output_dim16, phase=self.phase, layer_name='conv15', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc16_do = tf.nn.dropout(relumaxpoolconv16,self.keep_prob)
		
		# CONV LAYER 17 :
		nbr_filter17 = 1024
		output_dim17 = [ nbr_filter17]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv17, input_dim18 = self.layer_conv2dBNAct(input_tensor=rmpc16_do, input_dim=input_dim17, output_dim=output_dim17, phase=self.phase, layer_name='conv16', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc17_do = tf.nn.dropout(relumaxpoolconv17,self.keep_prob)
		
		# CONV LAYER 18 :
		nbr_filter18 = 512
		output_dim18 = [ nbr_filter18]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv18, input_dim19 = self.layer_conv2dBNAct(input_tensor=rmpc17_do, input_dim=input_dim18, output_dim=output_dim18, phase=self.phase, layer_name='conv17', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc18_do = tf.nn.dropout(relumaxpoolconv18,self.keep_prob)
		
		# CONV LAYER 19 :
		nbr_filter19 = 1024
		output_dim19 = [ nbr_filter19]
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=3, pooldim=2, poolstride=2)
		relumaxpoolconv19, input_dim20 = self.layer_conv2dBNAct(input_tensor=rmpc18_do, input_dim=input_dim19, output_dim=output_dim19, phase=self.phase, layer_name='conv18', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc19_do = tf.nn.dropout(relumaxpoolconv19,self.keep_prob)
		
		# CONV LAYER 20 :
		nbr_filter20 = 512
		output_dim20 = [ nbr_filter20]
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=3, pooldim=2, poolstride=2)
		relumaxpoolconv20, input_dim21 = self.layer_conv2dBNAct(input_tensor=rmpc19_do, input_dim=input_dim20, output_dim=output_dim20, phase=self.phase, layer_name='conv19', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc20_do = tf.nn.dropout(relumaxpoolconv20,self.keep_prob)
		
		#shape_conv = relumaxpoolconv2.get_shape().as_list()
		#shape_conv = relumaxpoolconv3.get_shape().as_list()
		#shape_conv = relumaxpoolconv4.get_shape().as_list()
		
		#shape_conv = rmpc4_do.get_shape().as_list()
		shape_conv = rmpc20_do.get_shape().as_list()
		
		#shape_conv = rmpc5_do.get_shape().as_list()
		#shape_conv = h_trans_def4.get_shape().as_list()
		
		shape_fc = [-1, shape_conv[1]*shape_conv[2]*shape_conv[3] ]
		out1 = 128
		#fc_x_input = tf.reshape( relumaxpoolconv2, shape_fc )
		#fc_x_input = tf.reshape( relumaxpoolconv3, shape_fc )
		#fc_x_input = tf.reshape( relumaxpoolconv4, shape_fc )
		
		#fc_x_input = tf.reshape( rmpc4_do, shape_fc )
		fc_x_input = tf.reshape( rmpc20_do, shape_fc )
		
		#fc_x_input = tf.reshape( rmpc5_do, shape_fc )
		#fc_x_input = tf.reshape( h_trans_def4, shape_fc )
		hidden1 = self.nn_layerBN(fc_x_input, shape_fc[1], out1, self.phase, 'layer1')
		dropped1 = tf.nn.dropout(hidden1, self.keep_prob)
	
		#out2 = 512
		#hidden2 = self.nn_layerBN(dropped1, out1, out2, self.phase,'layer2')
		#dropped2 = tf.nn.dropout(hidden2, self.keep_prob)

		#out3 = 256
		#hidden3 = self.nn_layerBN(dropped2, out2, out3, self.phase,'layer3')
		#dropped3 = tf.nn.dropout(hidden3, self.keep_prob)

		# Do not apply softmax activation yet, see below.
		#y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
		#y = nn_layer(dropped2, out2, nbrOutput, 'layer3', act=tf.identity)
		self.y = self.nn_layerBN(dropped1, out1, self.nbrOutput, self.phase, layer_name='layerOutput', act=tf.identity)	
		#self.y = self.nn_layer(dropped2, out2, nbrOutput, 'layerOutput', act=tf.identity)	
		#self.y = self.nn_layer(dropped3, out3, nbrOutput, 'layerOutput', act=tf.identity)	
	
	
	def build_modelMINI(self) :
		#with tf.name_scope('dropout'):
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.bool,name='phase')
		tf.summary.scalar('dropout_keep_probability', self.keep_prob)
		
		self.x_tensor = tf.reshape( self.x, [-1, self.nbrInput[0], self.nbrInput[1], self.nbrInput[2]] )
		
		# CONV LAYER 1 :
		shape_input = self.x_tensor.get_shape().as_list()
		input_dim1 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		nbr_filter1 = 24
		output_dim1 = [ nbr_filter1]
		relumaxpoolconv1, input_dim2 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=self.x_tensor, input_dim=input_dim1, output_dim=output_dim1, phase=self.phase, layer_name='conv0MaxPool0', act=tf.nn.relu, filter_size=5, stride=3, pooldim=2, poolstride=2)
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
		nbr_filter2 = 36
		output_dim2 = [ nbr_filter2]
		#relumaxpoolconv2, input_dim3 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=h_trans_def1, input_dim=input_dim2, output_dim=output_dim2, phase=self.phase, layer_name='conv1MaxPool1', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
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
		nbr_filter3 = 48
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
		nbr_filter4 = 128
		output_dim4 = [ nbr_filter4]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv4, input_dim5 = self.layer_conv2dBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv3', act=tf.nn.relu, filter_size=3, stride=1,padding='SAME')
		rmpc4_do = tf.nn.dropout(relumaxpoolconv4,self.keep_prob)
		
		'''
		#LAYER STN 4 :
		#shape_inputstn = rmpc4_do.get_shape().as_list()
		#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
		#layerstn_name = 'stn4'
		#h_trans_def4, out_size4, self.thetas4 = self.nn_layer_stn( rmpc4_do, inputstn_dim, layerstn_name, self.keep_prob)
		#shape_input = h_trans_def4.get_shape().as_list()
		#input_dim5 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
		# CONV LAYER 5 :
		#nbr_filter5 = 256
		#output_dim5 = [ nbr_filter5]
		#relumaxpoolconv5, input_dim6 = self.nn_layer_actMaxpoolConv2dDivide2(input_tensor=rmpc4_do, input_dim=input_dim5, output_dim=output_dim5, layer_name='conv5', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		#rmpc5_do = tf.nn.dropout(relumaxpoolconv5,self.keep_prob)
		nbr_filter5 = 256
		output_dim5 = [ nbr_filter5]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv5, input_dim6 = self.layer_conv2dBNAct(input_tensor=rmpc4_do, input_dim=input_dim5, output_dim=output_dim5, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1,padding='VALID')
		rmpc5_do = tf.nn.dropout(relumaxpoolconv5,self.keep_prob)
		
		
		# CONV LAYER 6 :
		nbr_filter6 = 128
		output_dim6 = [ nbr_filter6]
		relumaxpoolconv6, input_dim7 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc5_do, input_dim=input_dim6, output_dim=output_dim6, phase=self.phase, layer_name='conv5MaxPool3', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2,convpadding='SAME',poolpadding='VALID')
		rmpc6_do = tf.nn.dropout(relumaxpoolconv6,self.keep_prob)
		
		# CONV LAYER 7 :
		nbr_filter7 = 256
		output_dim7 = [ nbr_filter7]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv7, input_dim8 = self.layer_conv2dBNAct(input_tensor=rmpc6_do, input_dim=input_dim7, output_dim=output_dim7, phase=self.phase, layer_name='conv6', act=tf.nn.relu, filter_size=3, stride=1, padding='VALID')
		rmpc7_do = tf.nn.dropout(relumaxpoolconv7,self.keep_prob)
		
		
		# CONV LAYER 8 :
		nbr_filter8 = 128
		output_dim8 = [ nbr_filter8]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv8, input_dim9 = self.layer_conv2dBNAct(input_tensor=rmpc7_do, input_dim=input_dim8, output_dim=output_dim8, phase=self.phase, layer_name='conv7', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc8_do = tf.nn.dropout(relumaxpoolconv8,self.keep_prob)
		
		# CONV LAYER 9 :
		nbr_filter9 = 256
		output_dim9 = [ nbr_filter9]
		relumaxpoolconv9, input_dim10 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc8_do, input_dim=input_dim9, output_dim=output_dim9, phase=self.phase, layer_name='conv8MaxPool4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		#relumaxpoolconv8, input_dim9 = self.layer_conv2dBNAct(input_tensor=rmpc7_do, input_dim=input_dim8, output_dim=output_dim8, phase=self.phase, layer_name='conv7', act=tf.nn.relu, filter_size=1, stride=1)
		rmpc9_do = tf.nn.dropout(relumaxpoolconv9,self.keep_prob)
		
		# CONV LAYER 10 :
		nbr_filter10 = 512
		output_dim10 = [ nbr_filter10]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv10, input_dim11 = self.layer_conv2dBNAct(input_tensor=rmpc9_do, input_dim=input_dim10, output_dim=output_dim10, phase=self.phase, layer_name='conv9', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc10_do = tf.nn.dropout(relumaxpoolconv10,self.keep_prob)
		
		# CONV LAYER 11 :
		nbr_filter11 = 256
		output_dim11 = [ nbr_filter11]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv11, input_dim12 = self.layer_conv2dBNAct(input_tensor=rmpc10_do, input_dim=input_dim11, output_dim=output_dim11, phase=self.phase, layer_name='conv10', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc11_do = tf.nn.dropout(relumaxpoolconv11,self.keep_prob)
		
		# CONV LAYER 12 :
		nbr_filter12 = 512
		output_dim12 = [ nbr_filter12]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv12, input_dim13 = self.layer_conv2dBNAct(input_tensor=rmpc11_do, input_dim=input_dim12, output_dim=output_dim12, phase=self.phase, layer_name='conv11', act=tf.nn.relu, filter_size=3, stride=1)
		rmpc12_do = tf.nn.dropout(relumaxpoolconv12,self.keep_prob)
		
		# CONV LAYER 13 :
		nbr_filter13 = 256
		output_dim13 = [ nbr_filter13]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv13, input_dim14 = self.layer_conv2dBNAct(input_tensor=rmpc12_do, input_dim=input_dim13, output_dim=output_dim13, phase=self.phase, layer_name='conv12', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc13_do = tf.nn.dropout(relumaxpoolconv13,self.keep_prob)
		
		# CONV LAYER 14 :
		nbr_filter14 = 512
		output_dim14 = [ nbr_filter14]
		relumaxpoolconv14, input_dim15 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2,convpadding='VALID',poolpadding='SAME')
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13', act=tf.nn.relu, filter_size=3, stride=3)
		rmpc14_do = tf.nn.dropout(relumaxpoolconv14,self.keep_prob)
		
		# CONV LAYER 15 :
		nbr_filter15 = 1024
		output_dim15 = [ nbr_filter15]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv15, input_dim16 = self.layer_conv2dBNAct(input_tensor=rmpc14_do, input_dim=input_dim15, output_dim=output_dim15, phase=self.phase, layer_name='conv14', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc15_do = tf.nn.dropout(relumaxpoolconv15,self.keep_prob)
		
		# CONV LAYER 16 :
		nbr_filter16 = 512
		output_dim16 = [ nbr_filter16]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv16, input_dim17 = self.layer_conv2dBNAct(input_tensor=rmpc15_do, input_dim=input_dim16, output_dim=output_dim16, phase=self.phase, layer_name='conv15', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc16_do = tf.nn.dropout(relumaxpoolconv16,self.keep_prob)
		
		# CONV LAYER 17 :
		nbr_filter17 = 1024
		output_dim17 = [ nbr_filter17]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv17, input_dim18 = self.layer_conv2dBNAct(input_tensor=rmpc16_do, input_dim=input_dim17, output_dim=output_dim17, phase=self.phase, layer_name='conv16', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc17_do = tf.nn.dropout(relumaxpoolconv17,self.keep_prob)
		
		# CONV LAYER 18 :
		nbr_filter18 = 512
		output_dim18 = [ nbr_filter18]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv18, input_dim19 = self.layer_conv2dBNAct(input_tensor=rmpc17_do, input_dim=input_dim18, output_dim=output_dim18, phase=self.phase, layer_name='conv17', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc18_do = tf.nn.dropout(relumaxpoolconv18,self.keep_prob)
		
		# CONV LAYER 19 :
		nbr_filter19 = 1024
		output_dim19 = [ nbr_filter19]
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=3, pooldim=2, poolstride=2)
		relumaxpoolconv19, input_dim20 = self.layer_conv2dBNAct(input_tensor=rmpc18_do, input_dim=input_dim19, output_dim=output_dim19, phase=self.phase, layer_name='conv18', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc19_do = tf.nn.dropout(relumaxpoolconv19,self.keep_prob)
		
		# CONV LAYER 20 :
		nbr_filter20 = 512
		output_dim20 = [ nbr_filter20]
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=3, pooldim=2, poolstride=2)
		relumaxpoolconv20, input_dim21 = self.layer_conv2dBNAct(input_tensor=rmpc19_do, input_dim=input_dim20, output_dim=output_dim20, phase=self.phase, layer_name='conv19', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc20_do = tf.nn.dropout(relumaxpoolconv20,self.keep_prob)
		
		'''
		
		
		#shape_conv = rmpc20_do.get_shape().as_list()
		shape_conv = rmpc4_do.get_shape().as_list()
		
		#shape_conv = rmpc5_do.get_shape().as_list()
		#shape_conv = h_trans_def4.get_shape().as_list()
		
		shape_fc = [-1, shape_conv[1]*shape_conv[2]*shape_conv[3] ]
		out1 = 256
		#fc_x_input = tf.reshape( relumaxpoolconv2, shape_fc )
		#fc_x_input = tf.reshape( relumaxpoolconv3, shape_fc )
		#fc_x_input = tf.reshape( relumaxpoolconv4, shape_fc )
		
		fc_x_input = tf.reshape( rmpc4_do, shape_fc )
		#fc_x_input = tf.reshape( rmpc20_do, shape_fc )
		hidden1 = self.nn_layerBN(fc_x_input, shape_fc[1], out1, self.phase, 'layer1')
		dropped1 = tf.nn.dropout(hidden1, self.keep_prob)
	
		
		out2 = 128
		hidden2 = self.nn_layerBN(dropped1, out1, out2, self.phase,'layer2')
		dropped2 = tf.nn.dropout(hidden2, self.keep_prob)

		#out3 = 256
		#hidden3 = self.nn_layerBN(dropped2, out2, out3, self.phase,'layer3')
		#dropped3 = tf.nn.dropout(hidden3, self.keep_prob)

		# Do not apply softmax activation yet, see below.
		#y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
		#y = nn_layer(dropped2, out2, nbrOutput, 'layer3', act=tf.identity)
		#self.y = self.nn_layerBN(dropped1, out1, self.nbrOutput, self.phase, layer_name='layerOutput', act=tf.identity)	
		self.y = self.nn_layer(dropped2, out2, self.nbrOutput, 'layerOutput', act=tf.identity)	
		#self.y = self.nn_layer(dropped3, out3, nbrOutput, 'layerOutput', act=tf.identity)	
		
	def build_modelMINI1(self) :
		#with tf.name_scope('dropout'):
		self.keep_prob = tf.placeholder(tf.float32)
		self.phase = tf.placeholder(tf.bool,name='phase')
		tf.summary.scalar('dropout_keep_probability', self.keep_prob)
		
		self.x_tensor = tf.reshape( self.x, [-1, self.nbrInput[0], self.nbrInput[1], self.nbrInput[2]] )
		
		# CONV LAYER 1 :
		shape_input = self.x_tensor.get_shape().as_list()
		input_dim1 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		nbr_filter1 = 24
		output_dim1 = [ nbr_filter1]
		relumaxpoolconv1, input_dim2 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=self.x_tensor, input_dim=input_dim1, output_dim=output_dim1, phase=self.phase, layer_name='conv0MaxPool0', act=tf.nn.relu, filter_size=5, stride=3, pooldim=2, poolstride=2)
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
		nbr_filter2 = 36
		output_dim2 = [ nbr_filter2]
		#relumaxpoolconv2, input_dim3 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=h_trans_def1, input_dim=input_dim2, output_dim=output_dim2, phase=self.phase, layer_name='conv1MaxPool1', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
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
		nbr_filter3 = 48
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
		nbr_filter4 = 128
		output_dim4 = [ nbr_filter4]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv4, input_dim5 = self.layer_conv2dBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv3', act=tf.nn.relu, filter_size=3, stride=1,padding='SAME')
		rmpc4_do = tf.nn.dropout(relumaxpoolconv4,self.keep_prob)
		
		
		#LAYER STN 4 :
		#shape_inputstn = rmpc4_do.get_shape().as_list()
		#inputstn_dim = [-1, shape_inputstn[1], shape_inputstn[2], shape_inputstn[3]]
		#layerstn_name = 'stn4'
		#h_trans_def4, out_size4, self.thetas4 = self.nn_layer_stn( rmpc4_do, inputstn_dim, layerstn_name, self.keep_prob)
		#shape_input = h_trans_def4.get_shape().as_list()
		#input_dim5 = [shape_input[0], shape_input[1], shape_input[2], shape_input[3]]
		
		# CONV LAYER 5 :
		#nbr_filter5 = 256
		#output_dim5 = [ nbr_filter5]
		#relumaxpoolconv5, input_dim6 = self.nn_layer_actMaxpoolConv2dDivide2(input_tensor=rmpc4_do, input_dim=input_dim5, output_dim=output_dim5, layer_name='conv5', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		#rmpc5_do = tf.nn.dropout(relumaxpoolconv5,self.keep_prob)
		nbr_filter5 = 256
		output_dim5 = [ nbr_filter5]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv5, input_dim6 = self.layer_conv2dBNAct(input_tensor=rmpc4_do, input_dim=input_dim5, output_dim=output_dim5, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1,padding='VALID')
		rmpc5_do = tf.nn.dropout(relumaxpoolconv5,self.keep_prob)
		
		'''
		# CONV LAYER 6 :
		nbr_filter6 = 128
		output_dim6 = [ nbr_filter6]
		relumaxpoolconv6, input_dim7 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc5_do, input_dim=input_dim6, output_dim=output_dim6, phase=self.phase, layer_name='conv5MaxPool3', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2,convpadding='SAME',poolpadding='VALID')
		rmpc6_do = tf.nn.dropout(relumaxpoolconv6,self.keep_prob)
		
		# CONV LAYER 7 :
		nbr_filter7 = 256
		output_dim7 = [ nbr_filter7]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv7, input_dim8 = self.layer_conv2dBNAct(input_tensor=rmpc6_do, input_dim=input_dim7, output_dim=output_dim7, phase=self.phase, layer_name='conv6', act=tf.nn.relu, filter_size=3, stride=1, padding='VALID')
		rmpc7_do = tf.nn.dropout(relumaxpoolconv7,self.keep_prob)
		
		
		# CONV LAYER 8 :
		nbr_filter8 = 128
		output_dim8 = [ nbr_filter8]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv8, input_dim9 = self.layer_conv2dBNAct(input_tensor=rmpc7_do, input_dim=input_dim8, output_dim=output_dim8, phase=self.phase, layer_name='conv7', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc8_do = tf.nn.dropout(relumaxpoolconv8,self.keep_prob)
		
		# CONV LAYER 9 :
		nbr_filter9 = 256
		output_dim9 = [ nbr_filter9]
		relumaxpoolconv9, input_dim10 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc8_do, input_dim=input_dim9, output_dim=output_dim9, phase=self.phase, layer_name='conv8MaxPool4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		#relumaxpoolconv8, input_dim9 = self.layer_conv2dBNAct(input_tensor=rmpc7_do, input_dim=input_dim8, output_dim=output_dim8, phase=self.phase, layer_name='conv7', act=tf.nn.relu, filter_size=1, stride=1)
		rmpc9_do = tf.nn.dropout(relumaxpoolconv9,self.keep_prob)
		
		# CONV LAYER 10 :
		nbr_filter10 = 512
		output_dim10 = [ nbr_filter10]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv10, input_dim11 = self.layer_conv2dBNAct(input_tensor=rmpc9_do, input_dim=input_dim10, output_dim=output_dim10, phase=self.phase, layer_name='conv9', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc10_do = tf.nn.dropout(relumaxpoolconv10,self.keep_prob)
		
		# CONV LAYER 11 :
		nbr_filter11 = 256
		output_dim11 = [ nbr_filter11]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv11, input_dim12 = self.layer_conv2dBNAct(input_tensor=rmpc10_do, input_dim=input_dim11, output_dim=output_dim11, phase=self.phase, layer_name='conv10', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc11_do = tf.nn.dropout(relumaxpoolconv11,self.keep_prob)
		
		# CONV LAYER 12 :
		nbr_filter12 = 512
		output_dim12 = [ nbr_filter12]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv12, input_dim13 = self.layer_conv2dBNAct(input_tensor=rmpc11_do, input_dim=input_dim12, output_dim=output_dim12, phase=self.phase, layer_name='conv11', act=tf.nn.relu, filter_size=3, stride=1)
		rmpc12_do = tf.nn.dropout(relumaxpoolconv12,self.keep_prob)
		
		# CONV LAYER 13 :
		nbr_filter13 = 256
		output_dim13 = [ nbr_filter13]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv13, input_dim14 = self.layer_conv2dBNAct(input_tensor=rmpc12_do, input_dim=input_dim13, output_dim=output_dim13, phase=self.phase, layer_name='conv12', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc13_do = tf.nn.dropout(relumaxpoolconv13,self.keep_prob)
		
		# CONV LAYER 14 :
		nbr_filter14 = 512
		output_dim14 = [ nbr_filter14]
		relumaxpoolconv14, input_dim15 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2,convpadding='VALID',poolpadding='SAME')
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13', act=tf.nn.relu, filter_size=3, stride=3)
		rmpc14_do = tf.nn.dropout(relumaxpoolconv14,self.keep_prob)
		
		# CONV LAYER 15 :
		nbr_filter15 = 1024
		output_dim15 = [ nbr_filter15]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv15, input_dim16 = self.layer_conv2dBNAct(input_tensor=rmpc14_do, input_dim=input_dim15, output_dim=output_dim15, phase=self.phase, layer_name='conv14', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc15_do = tf.nn.dropout(relumaxpoolconv15,self.keep_prob)
		
		# CONV LAYER 16 :
		nbr_filter16 = 512
		output_dim16 = [ nbr_filter16]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv16, input_dim17 = self.layer_conv2dBNAct(input_tensor=rmpc15_do, input_dim=input_dim16, output_dim=output_dim16, phase=self.phase, layer_name='conv15', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc16_do = tf.nn.dropout(relumaxpoolconv16,self.keep_prob)
		
		# CONV LAYER 17 :
		nbr_filter17 = 1024
		output_dim17 = [ nbr_filter17]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv17, input_dim18 = self.layer_conv2dBNAct(input_tensor=rmpc16_do, input_dim=input_dim17, output_dim=output_dim17, phase=self.phase, layer_name='conv16', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc17_do = tf.nn.dropout(relumaxpoolconv17,self.keep_prob)
		
		# CONV LAYER 18 :
		nbr_filter18 = 512
		output_dim18 = [ nbr_filter18]
		#relumaxpoolconv4, input_dim5 = self.layer_conv2dBNMaxpoolBNAct(input_tensor=rmpc3_do, input_dim=input_dim4, output_dim=output_dim4, phase=self.phase, layer_name='conv4', act=tf.nn.relu, filter_size=3, stride=1, pooldim=2, poolstride=2)
		relumaxpoolconv18, input_dim19 = self.layer_conv2dBNAct(input_tensor=rmpc17_do, input_dim=input_dim18, output_dim=output_dim18, phase=self.phase, layer_name='conv17', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc18_do = tf.nn.dropout(relumaxpoolconv18,self.keep_prob)
		
		# CONV LAYER 19 :
		nbr_filter19 = 1024
		output_dim19 = [ nbr_filter19]
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=3, pooldim=2, poolstride=2)
		relumaxpoolconv19, input_dim20 = self.layer_conv2dBNAct(input_tensor=rmpc18_do, input_dim=input_dim19, output_dim=output_dim19, phase=self.phase, layer_name='conv18', act=tf.nn.relu, filter_size=3, stride=1, padding='SAME')
		rmpc19_do = tf.nn.dropout(relumaxpoolconv19,self.keep_prob)
		
		# CONV LAYER 20 :
		nbr_filter20 = 512
		output_dim20 = [ nbr_filter20]
		#relumaxpoolconv14, input_dim15 = self.layer_conv2dBNAct(input_tensor=rmpc13_do, input_dim=input_dim14, output_dim=output_dim14, phase=self.phase, layer_name='conv13Maxpool5', act=tf.nn.relu, filter_size=3, stride=3, pooldim=2, poolstride=2)
		relumaxpoolconv20, input_dim21 = self.layer_conv2dBNAct(input_tensor=rmpc19_do, input_dim=input_dim20, output_dim=output_dim20, phase=self.phase, layer_name='conv19', act=tf.nn.relu, filter_size=1, stride=1, padding='VALID')
		rmpc20_do = tf.nn.dropout(relumaxpoolconv20,self.keep_prob)
		
		'''
		
		
		#shape_conv = rmpc20_do.get_shape().as_list()
		shape_conv = rmpc5_do.get_shape().as_list()
		
		#shape_conv = rmpc5_do.get_shape().as_list()
		#shape_conv = h_trans_def4.get_shape().as_list()
		
		shape_fc = [-1, shape_conv[1]*shape_conv[2]*shape_conv[3] ]
		out1 = 1024
		#fc_x_input = tf.reshape( relumaxpoolconv2, shape_fc )
		#fc_x_input = tf.reshape( relumaxpoolconv3, shape_fc )
		#fc_x_input = tf.reshape( relumaxpoolconv4, shape_fc )
		
		fc_x_input = tf.reshape( rmpc5_do, shape_fc )
		#fc_x_input = tf.reshape( rmpc20_do, shape_fc )
		hidden1 = self.nn_layerBN(fc_x_input, shape_fc[1], out1, self.phase, 'layer1')
		dropped1 = tf.nn.dropout(hidden1, self.keep_prob)
	
		
		out2 = 256
		hidden2 = self.nn_layerBN(dropped1, out1, out2, self.phase,'layer2')
		dropped2 = tf.nn.dropout(hidden2, self.keep_prob)

		#out3 = 256
		#hidden3 = self.nn_layerBN(dropped2, out2, out3, self.phase,'layer3')
		#dropped3 = tf.nn.dropout(hidden3, self.keep_prob)

		# Do not apply softmax activation yet, see below.
		#y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)
		#y = nn_layer(dropped2, out2, nbrOutput, 'layer3', act=tf.identity)
		#self.y = self.nn_layerBN(dropped1, out1, self.nbrOutput, self.phase, layer_name='layerOutput', act=tf.identity)	
		self.y = self.nn_layer(dropped2, out2, self.nbrOutput, 'layerOutput', act=tf.identity)	
		#self.y = self.nn_layer(dropped3, out3, nbrOutput, 'layerOutput', act=tf.identity)	
		
	def init_model(self,lr=1e-4) :
		  sqdiff = tf.square(self.y_ - self.y)
		  self.loss = tf.reduce_mean(sqdiff)
		  tf.summary.scalar('loss', self.loss)

		  # With decaying learning rate :
		  starter_learning_rate = lr;
		  self.global_step = tf.Variable(0,trainable=False)
		  decay_steps = 1000
		  decay_rate = 0.99
		  #every decay_steps, the learning_rate is decayed of decay_rate
		  self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, decay_steps, decay_rate)

		  tflambda_regL2 = tf.constant(self.lambda_regL2,name='lambda_regL2')
		  tf.summary.scalar('L2_loss',self.l2_loss)

		  self.total_loss = self.loss+tflambda_regL2*self.l2_loss
		  tf.summary.scalar('total_loss',self.total_loss)

		  self.train_step = tf.train.AdamOptimizer(self.learning_rate,name='Adam').minimize(self.total_loss, global_step=self.global_step)
		  #self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)
		  #self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)
		  
		  #with tf.name_scope('Gradients'):
		  #self.gradients = tf.gradients(self.total_loss,tf.trainable_variables())
		  #for grads,i in enumerate(self.gradients):
		  #    tf.summary.scalar('gradients'+str(i), grads)
		  #    tf.summary.histogram('gradients'+str(i), grads)
		      #self.variable_summaries(grads,'gradients'+str(i))

	def train(self,args,dataset, filepathIn=None):
		  self.dataset = dataset
		  self.batch_size = args.batch_size
		  nbr_epoch = args.nb_epoch
		  iter_per_epoch = args.samples_per_epoch
		  
		  
		  trainsize = self.dataset.getTrainSize()
		  testsize = self.dataset.getTestSize()
		  
		  nbrtest = 5
		  testbatch_size = int(testsize/nbrtest)+1

		  print("NBR OF SAMPLES :: train:"+str(trainsize)+" ; test:"+str(testsize))
		  testiteration = 0
		  freqTest = 25
		  
		  with tf.Session() as self.sess :
		      # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
		      self.merged = tf.summary.merge_all()
		      self.train_writer = tf.summary.FileWriter(self.filepath + '/train', self.sess.graph)
		      self.test_writer = tf.summary.FileWriter(self.filepath + '/test', self.sess.graph)

		      self.init_op = tf.global_variables_initializer()
		      self.sess.run(self.init_op)

		      # Add ops to save and restore all the variables.
		      self.saver = tf.train.Saver()
		      # START FROM THE MODEL PRE TRAINED :
		      if filepathIn is not None :
		      	self.saver.restore(self.sess, filepathIn)

		      for epoch_i in range(nbr_epoch):
		          for i in range(iter_per_epoch - 1):
		              summary, _, loss = self.sess.run([self.merged, self.train_step, self.total_loss], feed_dict=self.feed_dict(True,i))
		              print('TRAINING : Loss at step %s: %s' % ((epoch_i)*iter_per_epoch+(i), loss))
		              self.train_writer.add_summary(summary, epoch_i*iter_per_epoch+i)

		              if i % freqTest == 0:  # Record summaries and test-set accuracy
		                  testiteration += 1
		                  for itest in range(nbrtest) :
		                      summary, loss = self.sess.run([self.merged, self.total_loss], feed_dict=self.feed_dict(False,itest) )
		                      self.test_writer.add_summary(summary, testiteration*nbrtest+itest)
		                      print('TESTING : Loss at step %s: %s' % ((testiteration-1)*nbrtest+itest, loss))
		                  print('Learning rate = '+str(self.sess.run(self.learning_rate)))
		                  # Save the variables to disk.
		                  save_path = self.saver.save(self.sess, self.filepath+'.ckpt')
		                  print("Model saved in file: %s" % save_path)

		      self.train_writer.close()
		      self.test_writer.close()

		      # Save the variables to disk.
		      save_path = self.saver.save(self.sess, self.filepath)
		      print("Model saved in file: %s" % save_path)
		  
	def inference(self,x):
		with tf.Session() as self.sess :
		    self.init_op = tf.global_variables_initializer()
		    self.sess.run(self.init_op)

		    # Add ops to save and restore all the variables.
		    self.saver = tf.train.Saver()
		    # START FROM THE MODEL PRE TRAINED :
		    self.saver.restore(self.sess, self.filepath+'.ckpt')
		    
		    outputs = self.sess.run([self.y], feed_dict={self.x: x, self.keep_prob: 0.5, self.phase: False})
		    print('INFERENCE : y:')
		    print(outputs)
		    
		return outputs
		        
	
	def feed_dict(self,train, iteration=0):
		  """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		  if train :
		      xs, ys = self.dataset.batch_generator(self.batch_size, train)
		      #print(list(self.dataset.batch_generator(self.batch_size, train)))
		      k = dropoutK#FLAGS.dropout
		  else:
		      xs, ys = self.dataset.batch_generator(self.batch_size, train)
		      k = 1.0
		  return {self.x: xs, self.y_: ys, self.keep_prob: k, self.phase: train}


	
def build_model(args) :
	#model = Sequential()
	#model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
	#model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
	#model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
	#model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
	#model.add(Conv2D(64, 3, 3, activation='elu'))
	#model.add(Conv2D(64, 3, 3, activation='elu'))
	#model.add(Dropout(args.keep_prob))
	#model.add(Flatten())
	#model.add(Dense(100, activation='elu'))
	#model.add(Dense(50, activation='elu'))
	#model.add(Dense(10, activation='elu'))
	#model.add(Dense(2))
	#model.summary()
	
	model = NN( filepath_base,nbrinput=nbrinput,nbroutput=nbroutput,lr=args.learning_rate,filepathin=None)
	return model

def train_model(model,args,dataset,filepathIn=None) :
	model.train(args, dataset, filepathIn=filepathIn)


def s2b(s) :
	s = s.lower()
	return s=='true' or s=='yes'
	
def main() :
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	#parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='./datasets/dataset1Robot.ImagesCmdsOdoms.npz')
	parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='/home/kevin/rosbuild_ws/sandbox/GazeboRL/images')
	parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
	parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
	parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=100)
	parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=50)
	if useMINI :
		parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=4)
	else :
		parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=12)
	parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
	parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-2)
	args = parser.parse_args()
	
	#print parameters
	print('-' * 30)
	print('Parameters')
	print('-' * 30)
	for key, value in vars(args).items():
		  print('{:<20} := {}'.format(key, value))
	print('-' * 30)
	
	dropoutK = args.keep_prob
	
	dataset = load_dataset(args)
	model = build_model(args)
	train = True
	
	#modelYOLO1
	#240x1280
	filepathIn = None#'./logs/archiYOLO1_240_1280--2-0.01.ckpt'
	
	#240x640
	#filepathIn = './logs/archiYOLO1_240_1280--2-0.005.ckpt'
	
	
	
	if train :
		train_model(model, args, dataset,filepathIn=filepathIn)
	else :
		testbatchx, testbatchy = dataset.batch_generator(batch_size=10, is_training=False)
		model.inference(x=testbatchx)
		print('desired output :', testbatchy)
	
if __name__ == '__main__' :
	main()
	
	

												

