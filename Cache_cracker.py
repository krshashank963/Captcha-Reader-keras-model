# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:05:05 2020

@author: SHASHANK RAJPUT
"""


import os
import cv2
import random
import numpy as np
import string
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout

from tensorflow.keras.layers import Flatten, Dense, Layer, BatchNormalization

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model,Input
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import argparse

symbols =string.ascii_lowercase +'0123456789'
len_symbols =len(string.ascii_lowercase +'0123456789')
Data ='samples'
def preprocessing(path):


	print("[INFO] Processing Images.......")
	n_samples= len(os.listdir(path))
	

	# variables for data and labels 
	X = np.zeros((n_samples , 50 , 200 ,1 )) 
	y = np.zeros((n_samples,5, 36 ))     

	for i , image in enumerate(os.listdir(path)):
		img = cv2.imread(os.path.join(path, image) , cv2.IMREAD_GRAYSCALE)

		targets = image.split('.')[0]

		if len(targets)<6:

			img = img/255.0
			img = np.reshape(img , (50,200,1))

			#find the char and one hot encode it to the target
			targ = np.zeros((5,36))

			for l , char in enumerate(targets):

				idx = symbols.find(char)
				targ[l , idx] = 1

			X[i] = img
			y[i,: ,:] = targ

	print("[INFO] Processing Finishes.....")

	return X,y

def my_model():
    inputs =Input(shape=(50,200,1),name ='image')
    x= Conv2D(16, (3,3),padding='same',activation='relu')(inputs)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x= Conv2D(32, (3,3),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x= Conv2D(32, (3,3),padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2) , padding='same')(x)
    x = BatchNormalization()(x)
    out_flat= Flatten()(x)
    
     #char-1
    dense_1 = Dense(64 , activation='relu')(out_flat)
    dropout_1= Dropout(0.5)(dense_1)
    out_1 = Dense(len_symbols , activation='sigmoid' , name='char_1')(dropout_1)
    
    #char-2
    dense_2 = Dense(64 , activation='relu')(out_flat)
    dropout_2= Dropout(0.5)(dense_2)
    out_2 = Dense(len_symbols , activation='sigmoid' , name='char_2')(dropout_2)
    
    #char-3
    dense_3 = Dense(64 , activation='relu')(out_flat)
    dropout_3= Dropout(0.5)(dense_3)
    out_3 = Dense(len_symbols , activation='sigmoid' , name='char_3')(dropout_3)
    
    #char-4
    dense_4 = Dense(64 , activation='relu')(out_flat)
    dropout_4= Dropout(0.5)(dense_4)
    out_4 = Dense(len_symbols , activation='sigmoid' , name='char_4')(dropout_4)
    
    #char-5
    dense_5 = Dense(64 , activation='relu')(out_flat)
    dropout_5= Dropout(0.5)(dense_5)
    out_5 = Dense(len_symbols , activation='sigmoid' , name='char_5')(dropout_5)
    
    model_out = Model(inputs=inputs , outputs=[out_1 , out_2 , out_3 , out_4 , out_5])
    
    return model_out

model =my_model()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
X, y = preprocessing(Data)    
trainX , testX , trainY , testY = train_test_split(X, y , test_size=0.2 , random_state=42)

#target values
labels = {'char_1': trainY[:,0,:], 
         'char_2': trainY[:,1,:],
         'char_3': trainY[:,2,:],
         'char_4': trainY[:,3,:],
         'char_5': trainY[:,4,:]}

test_labels = {'char_1': testY[:,0,:], 
         'char_2': testY[:,1,:],
         'char_3': testY[:,2,:],
         'char_4': testY[:,3,:],
         'char_5': testY[:,4,:]}
history = model.fit(trainX ,labels , epochs=30 , batch_size=64 , validation_split=0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot( loss, 'b', label='Training loss')
plt.plot(val_loss, 'r', label='Validation loss')

 
plt.show()

#prection
def predictions(image):
    
    image = np.reshape(image , (50,200))
    result = model.predict(np.reshape(image , (1,50,200,1)))
    result = np.reshape(result ,(5,36))
    indexes =[]
    for i in result:
        indexes.append(np.argmax(i))
        
    label=''
    for i in indexes:
        label += symbols[i]
        
    plt.imshow(image)
    plt.title(label)



predictions('testX[10]')
