import numpy as np
import tensorflow as tf

class Pd :
	def sample(self, state) :
		raise NotImplementedError
		
	def likelyhood(self, x) :
		raise NotImplementedError
	
	def logpd(self, x) :
		'''
		log L(param1, param2, ... | sample1, sample2, ... ) = log \prod_{sample} pd( sample | param1, param2 ) 
		'''
		raise NotImplementedError
		
	def sample(self) :
		raise NotImplementedError
		
	def kl(self, other) :
		raise NotImplementedError
		
		
class DiagGaussianPd(Pd) :
	def __init__(self, mean, sigma) :
		self.mean = mean
		self.sigma = sigma
		self.sigma2 = tf.square(self.sigma)
		self.logstd2 = tf.log(self.sigma2)
		
	def likelyhood(self, x) :
		return tf.exp( (-0.5) * tf.square( x - self.mean) / self.sigma2 ) / tf.sqrt( 2.0*np.pi*self.sigma2 )
	
	def logpd(self, x) :
		return -0.5*tf.log(2.0*np.pi) -0.5*self.logstd2 -0.5*tf.square( x - self.mean)/self.sigma2
		
	def sample(self) :
		return self.mean + tf.sqrt(self.sigma2)*tf.random_normal(shape=tf.shape(self.mean))
	
	def kl(self, other) :
		assert isinstance(other, DiagGaussianPd)
		#TODO :
		return 0.5
	
	
