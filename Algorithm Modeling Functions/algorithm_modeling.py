
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
import re
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense,LSTM, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def LSTM_model(units,train_generator,test_generator,dactivation='linear', \
               n_outputs=1,epochs=100,learning_rate=0.001,dropout=0.1,alpha=0.05):
    
    trainX,trainY = train_generator[0]
    testX,testY = test_generator[0]
    
    model = Sequential()
    model.add(LSTM(units,input_shape=(trainX.shape[1],trainX.shape[2]),activation='tanh',#return_sequences=True,
                  recurrent_activation='sigmoid',recurrent_dropout=0,unroll=False,use_bias=True))
   # model.add(LSTM(units,activation='tanh',
    #          recurrent_activation='sigmoid',recurrent_dropout=0,unroll=False,use_bias=True))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout))
    model.add(Dense(n_outputs,activation=dactivation))
    
    callback = EarlyStopping(monitor='loss',patience = 10, mode='auto')
    
    adam = Adam(learning_rate=learning_rate)
    
    model.compile(loss='mse',optimizer=adam, metrics=['mae','mse'])
    
    #with tf.device('XLA_GPU'):
        #print(tf.device('/gpu:0'))
    #    print(tf.test.is_built_with_cuda())
        
    hist = model.fit(train_generator,epochs=epochs, validation_data=test_generator,callbacks=callback,\
                     verbose=2,shuffle=False,batch_size=10)
    
    return model,hist
