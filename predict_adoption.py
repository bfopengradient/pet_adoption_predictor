 

import numpy as np
import pandas as pd
import keras
from keras import callbacks
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.engine import Input
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout 
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM  
import matplotlib.pyplot as plt
import pymysql.cursors
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
#Establish connection to mysql database hosted locally as dbpets.
from sqlalchemy import create_engine
from sqlalchemy.sql import select
engine = create_engine("mysql+pymysql://root:bf_op12345@localhost/dbpets")
con = engine.connect()

 
  

class pets:
	''' Class that takes the data prepared via the prep_pets class for use in the machine learning algorithm below ''' 


    #Print accuracy score
	def print_metrics(self,y):
	    pred_round_1 = np.round(self, decimals=0,out=None)     
	    acc= accuracy_score(y, pred_round_1) 
	    print('Model accuracy score:', acc)       
	    
	#Plot model convergence path    
	def plot_model(self):	     
	    plt.plot(self.history['acc'])
	    plt.plot(self.history['val_acc'])
	    plt.title('model accuracy')
	    plt.ylabel('accuracy')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'val'], loc='upper left')
	    plt.show()
	    # summarize history for loss
	    plt.plot(self.history['loss'])
	    plt.plot(self.history['val_loss'])
	    plt.title('model loss')
	    plt.ylabel('loss')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'val'], loc='upper left')
	    plt.show() 


    #Main method to run model and generate accuracy score with plot of model learning history, feed in X(data) and y(labels) matrices.
	def run_model(data,labels):		  
        #Tokenize and pad records to uniform length oof 23 tokens per record.
		t = Tokenizer()
		embed= data.astype(str)
		t.fit_on_texts(embed)
		t_size = len(t.word_index) + 1
		#Integer encode the feature set 
		encoded_docs = t.texts_to_sequences(embed) 
		#Pad records to a max length of 23 tokens per record.
		max_length = 23
		#padded docs is the predictor matrix that will be later split into train/test data.
		padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

		#Splitting train/test data 80/20 but will also further split the training data to use 20% for validation during training. 
		X_train, X_test, y_train, y_test = train_test_split(padded_docs,labels, test_size=0.20, shuffle=True)
		print('Target class(training data)')
		print(y_train.value_counts())
		print('')
		print('Baseline: Target class(test data)')
		print(y_test.value_counts())
		print('')

        #Keep record of actual embeddings used in the model. Save table as 'X_train' in the dbpets database.    
		pd.DataFrame(X_train).to_sql(name='X_train',con=con,if_exists='replace')
 


		#Structure of neural net: CNN with Max pooling layer feeding into two LSTM layers. Longer term memory kept on in first LSTM layer
		#second LSTM layer uses heavier regularisation before final dense fully connected layer votes. The idea is to reduce weight matrix and complexity
		#before the final layer is asked to predict whether the pet is going to be adopted or not.

		#Using Keras sequential API 
		model_1 = Sequential()
		#1st layer
		#Embedding layer
		#Embedding weights not frozen during training!
		e = Embedding(t_size, 150, input_length=max_length,trainable=True)
		model_1.add(e)

		#2nd layer
		#One dimension Convolutional neural net(CNN) layer 
		#Testing narrower normalization of initializer as default has wider normalisation tolerance.
		c = Conv1D(filters=50, kernel_size=23,padding='same',  kernel_initializer='glorot_normal', activation='relu')
		model_1.add(c)

		#3rd Layer
		#Maxpool layer reducing output dimenaionality from CNN layer ahead of LSTM layers.
		mp= MaxPooling1D(pool_size=4)
		model_1.add(mp)
		              
		#4th layer     
		#LSTM 1st layer. Using some random dropout and some regularisation ahead of second LSTM layer. 
		#Expanding dimensionality again before shrinking it again below.
		ls1= LSTM(100, input_shape=(50, 1),kernel_initializer='glorot_normal',
		          return_sequences=True,activation='relu',dropout=0.1,
		          recurrent_dropout=0.1)
		model_1.add(ls1)
		              		    
		#5th layer    
		#LSTM 2nd layer, using more regularization before weights are passed to final fully connected layer.
		ls2= LSTM(100, input_shape=(15, 1),kernel_initializer='glorot_normal',activation='relu',kernel_regularizer=regularizers.l2(0.5), 
		           bias_regularizer=regularizers.l2(0.5),activity_regularizer=regularizers.l2(0.5),
		          recurrent_regularizer=regularizers.l2(0.5))
		model_1.add(ls2)

		#6th layer
		#Final fully Connected Dense layer with sigmoidal logistic regression function.
		d = Dense(1, activation='sigmoid')
		model_1.add(d)

		#Using reduce learning rate on val_accuracy plateau with a patience of one Epoch
		lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.15, 
		                                  patience=1, verbose=0, mode='auto', cooldown=0, min_lr=0)
		 
		#For this exercise I am usding online batch learning where weights are updated after each record is feed to algorithm.  
		#Compile,fit model and evaluate on test data. fit and evaluation.
		model_1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
		#Spliting the trainning data into a 30/20 split where 20% of records will be used in training to help the model learn from it's errors.
		history_1 = model_1.fit(X_train, y_train, epochs=7, callbacks=[lr],batch_size=1, 
		           validation_split=0.20, shuffle=True, verbose=0)		 
		model_1.evaluate(X_test, y_test, verbose=0,batch_size=1)
		pred_cnn = model_1.predict(X_test, batch_size=1,verbose=0)

        #Produce accuracy and plot of model learning.
		pets.print_metrics(pred_cnn,y_test) 
		pets.plot_model(history_1)
	 


 
