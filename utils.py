#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os, sys
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup as bs
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
from pandas.plotting import lag_plot


# In[13]:
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Dense, SimpleRNN, LSTM


from sklearn.metrics import r2_score
from math import sqrt


# In[14]:


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[15]:


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD

import math
from sklearn.metrics import mean_squared_error


# In[16]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# In[19]:


# !pip install gif


# In[20]:


from pandas.plotting import autocorrelation_plot
import json


# In[21]:


from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


# In[22]:


import tensorflow as tf
tf.__version__

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm


# In[23]:


from sklearn import metrics
import statsmodels.api as sm

import sys
import os
from os.path import dirname
parent = (dirname(os.path.abspath('')))
sys.path.insert(0, parent)

# # Read Datam

# In[107]:


class TimeSeries:
    """
    read data and preprocess it for time series projects
    """

    def __init__(self, *args, **kwargs):
        
        self.address = kwargs.get("address")
        self.df = self.read_data()
        self.lags = kwargs.get("lags") or 60
        self.split_fraction = kwargs.get("split_fraction") or 0.69
        
        self.sc = MinMaxScaler(feature_range=(0,1))
        
        self.training_set, self.test_set = self._time_series_train_test_split()
        self.X_train, self.y_train, self.X_test = self._prepare_train_test_dataset()
        
        self.regressor = kwargs.get("regressor") or None
        self.history = None
        
        self.predicted_ICME_reshaped = None
        
    def read_data(self):
        
        df = pd.read_pickle(self.address)
        return df
    
    
    def _time_series_train_test_split(self):
        
#         self.lags = lags
#         self.split_fraction = split_fraction
        # use first 347 timestamps for training 
#         print(self.split_fraction)
        size_of_training = int(self.split_fraction*len(self.df))

        # set the start of training row date
        train_start = self.df[0:1].index[0]
        train_start = str(train_start)

        a = self.df[0:1].index[0]

        # set the end of training row date
        train_end = self.df[(size_of_training-1):size_of_training].index[0]
        train_end = str(train_end)

        # use last 60 time stamps for prediction one timestamps ahead (18 days ahead)

        next_timestamps_test_num = len(self.df) - size_of_training # 347

        test_start = self.df[(size_of_training-self.lags-1):(size_of_training-self.lags)].index[0]
        test_start = str(test_start)

        test_end = self.df[(size_of_training+next_timestamps_test_num-1):(size_of_training+next_timestamps_test_num)].index[0]
        test_end = str(test_end)

        training_set = self.df[train_start:train_end]
        test_set = self.df[test_start:test_end]
        self.training_set, self.test_set = training_set, test_set
        return self.training_set, self.test_set
        
    
    def plot(self):
        
        self.training_set.plot(figsize=(16,6),legend=True)
        self.test_set.plot(figsize=(16,6),legend=True)
        plt.legend([f'Training set (first {self.training_set.shape[0]})',
                    f'Test set (next {self.test_set.shape[0]} time stamps)'])
        plt.title('ICME km/s')
        plt.show()
    
    

    def _prepare_train_test_dataset(self):

#         self.sc = MinMaxScaler(feature_range=(0,1))

        size_of_training = int(self.split_fraction*len(self.df))
        next_timestamps_test_num = len(self.df) - size_of_training
        
        training_set = self.training_set
        test_set = self.test_set
        # training_set.values
        training_set_np = np.array(training_set)
        # training_set_np
        training_set_np = training_set_np.reshape(-1,1)
        # training_set_np

        # Scaling the training set
    #     sc = MinMaxScaler(feature_range=(0,1))
        training_set_scaled = self.sc.fit_transform(training_set_np)

    #     Using 60 timestamps for pridction next timestamp.

        X_train = []
        y_train = []

        num_use_past_data = self.lags # 60
        size_of_training = training_set_scaled.size 



        for i in range(num_use_past_data, size_of_training):
            X_train.append(training_set_scaled[i-num_use_past_data:i, 0])
            y_train.append(training_set_scaled[i,0])
        X_train, y_train = np.array(X_train), np.array(y_train)


        # Reshaping X_train for efficient modelling
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))



        # test_set.values
        test_set_np = np.array(test_set)
        # test_set_np
        test_set_np = test_set_np.reshape(-1,1)
        # test_set_np

        # Scaling the test set
        inputs = self.sc.fit_transform(test_set_np)

        # Preparing X_test and prediction
        X_test = []
        for i in range(self.lags,(next_timestamps_test_num+self.lags)):
            X_test.append(inputs[i-self.lags:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

        self.X_train, self.y_train, self.X_test = X_train, y_train, X_test
        
        return self.X_train, self.y_train, self.X_test
    
    
    def _setup_lstm_model(self, 
                   optimizer:str='rmsprop', 
                   loss:str='mean_squared_error',
                   print_summary:bool = False):
        
        # lstm source https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru
        # The LSTM architecture
        self.regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        self.regressor.add(LSTM(units=5, 
                                return_sequences=True, 
                                input_shape=(self.X_train.shape[1],1)))
        self.regressor.add(Dropout(0.2))

        # Second LSTM layer
        self.regressor.add(LSTM(units=5, 
                                return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Third LSTM layer
        self.regressor.add(LSTM(units=5, 
                                return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Fourth LSTM layer
        self.regressor.add(LSTM(units=5))
        self.regressor.add(Dropout(0.2))

        # The output layer
        self.regressor.add(Dense(units=1))

        # Compiling the RNN
        self.regressor.compile(optimizer=optimizer, 
                                loss=loss)
        if print_summary:
            self.regressor.summary()
    

    def fit_lstm(self, 
                 monitor:str="loss",
                 min_delta:int=0,
                 patience:int = 10,
                 verbose = 0,
                 epochs = 25,
                 validation_split = 0.2):

        self._setup_lstm_model()
        
        self.path_checkpoint = parent+ f"/Models/LSTM_monitor_{monitor}_epochs_{epochs}_validation_split_{validation_split}.h5"
            
        es_callback = keras.callbacks.EarlyStopping(monitor, min_delta, patience)

        modelckpt_callback = keras.callbacks.ModelCheckpoint(monitor=monitor,
                                                            filepath=self.path_checkpoint,
                                                            verbose=verbose,
                                                            #save_weights_only=True,
                                                            #save_best_only=True,
                                                            )

        # Fitting to the training set
        self.history = self.regressor.fit(self.X_train, self.y_train,
                            epochs=epochs, validation_split=validation_split,
                            callbacks=[es_callback, modelckpt_callback],
                        )
    

    
    def plot_history(self):
        
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.ylim(0,)
        plt.xlabel('Epoch')
        plt.ylabel('Error V_ICME (km/s) (i)')
        plt.legend()
        plt.grid(True)
        
        
    def _load_saved_model(self, print_summary:bool = False):
        path = self.path_checkpoint
        self.regressor = keras.models.load_model(path)
        if print_summary:
            self.regressor.summary()

    def model_predict(self):
        self._load_saved_model()
        predicted_ICME = self.regressor.predict(self.X_test)
        predicted_ICME = self.sc.inverse_transform(predicted_ICME)
        predicted_ICME_reshaped = predicted_ICME.reshape(-1, )
        self.predicted_ICME_reshaped = predicted_ICME_reshaped
    
    def compare_test_result(self, figsize=(15,5)):
        
        plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        plt.plot(self.history.history['loss'], label='loss')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.ylim(0,)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)
        
        # make a series from predictions
        series = pd.Series(self.predicted_ICME_reshaped) 
        # set the index names
        series.index = self.test_set[self.lags+1:].index
        
        real_values = self.test_set[self.lags+1:]
        predited_values = series
         
        
        plt.subplot(1,3,2)
        plt.plot(series)
        plt.plot(self.test_set[self.lags+1:])
        coefficient_of_dermination = r2_score(real_values, predited_values)
        coefficient_of_dermination
        rs2 = round(coefficient_of_dermination,2)
        plt.xlabel("date")
        plt.ylabel("ICME km/s")

        plt.subplot(1,3,3)
        plt.scatter(real_values, predited_values, color='r', 
            alpha=0.4, label=f'Model vs Data | R2 = {rs2}',s=100)
        plt.xlabel("ICME km/s data", fontsize=14)
        plt.ylabel("ICME km/s predicted", fontsize=14)
        plt.xticks(fontsize=12), plt.yticks(fontsize=12)

        plt.xlim(300,650), plt.ylim(300,650)
        plt.tight_layout()
        plt.legend(fontsize=14)

