import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from utils import INPUT_SHAPE, imgh, imgw, imgch, batch_generator
import argparse

def load_data(args) :
	data = np.load(args.data_dir)
	X = data['images']
	Y = data['cmds']
	
	X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size=args.test_size, random_state=0)
	
	del X
	del Y
	del data
	
	print(len(X_valid),len(X_train))
	
	return X_train, X_valid, Y_train, Y_valid
	
def build_model(args) :
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
	model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Conv2D(64, 3, 3, activation='elu'))
	model.add(Dropout(args.keep_prob))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(2))
	model.summary()
	
	return model

def train_model(model,args,X_train,X_valid,Y_train,Y_valid) :
	
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
																monitor='val_loss',
																verbose=0,
																save_best_only=args.save_best_only,
																mode='auto')
																
	tensorboard = TensorBoard(log_dir='./logs', 
														histogram_freq=0, 
														write_graph=True, 
														write_images=False, 
														embeddings_freq=0, 
														embeddings_layer_names=None, 
														embeddings_metadata=None)
														
	
	model.compile(loss='mean_squared_error',optimizer=Adam(lr=args.learning_rate) )
	
	model.fit_generator( batch_generator(X_train,Y_train, args.batch_size,True),
												args.samples_per_epoch,
												args.nb_epoch,
												max_q_size=1,
												validation_data=batch_generator(X_valid,Y_valid,args.batch_size,False),
												nb_val_samples=1000,#len(X_valid),
												callbacks=[checkpoint,tensorboard],
												verbose=1)


def s2b(s) :
	s = s.lower()
	return s=='true' or s=='yes'
	
def main() :
	parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
	parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='./datasets/dataset1Robot.ImagesCmdsOdoms.npz')
	parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.5)
	parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.25)
	parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
	parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=1000)
	parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=20)
	parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
	parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=3.0e-4)
	args = parser.parse_args()
	
	#print parameters
	print('-' * 30)
	print('Parameters')
	print('-' * 30)
	for key, value in vars(args).items():
		  print('{:<20} := {}'.format(key, value))
	print('-' * 30)
	
	data = load_data(args)
	model = build_model(args)
	train_model(model, args, *data)
	
if __name__ == '__main__' :
	main()
	
	

												
