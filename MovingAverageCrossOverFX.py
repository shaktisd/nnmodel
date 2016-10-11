# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 06:30:23 2016
Moving average crossover
@author: Shakti
"""
import pandas as pd
import talib
import numpy as np


#df0 = pd.read_csv('IOC.csv',parse_dates=['Date'])
df0 = pd.read_csv('FX_TRAIN.csv',sep=";",parse_dates=['Date'])
#df0 = df0.reindex(index=df0.index[::-1])

# Normalize input
#cols_to_norm = ['Open','High','Low','Last','Close','Total Trade Quantity','Turnover (Lacs)']
#df0[cols_to_norm] = df0[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

#Split data into train & validate & test
#df = df0[df0.Date < '2016-09-01']
df = df0
#dfValidate =df0[df0.Date > '2010-01-01']




#SMA Crossover
df['SMA_20'] = pd.rolling_mean(df['Close'],20,min_periods=20)
df['SMA_50'] = pd.rolling_mean(df['Close'],50,min_periods=50)
previous_20 = df['SMA_20'].shift(1)
previous_50 = df['SMA_50'].shift(1)
df['SMA_CROSSING_SIGNAL']  = np.where(((df['SMA_20'] >= df['SMA_50']) & (previous_20 <= previous_50)),
1,
0)

#Interday Difference
df['INTERDAY_DIFF_SIGNAL'] = np.where((df['Close'] - df['Close'].shift(1)) > 0,1,0)

#Intraday Diff
df['INTRADAY DIFF_SIGNAL'] = np.where(df['Close'] - df['Open'] > 0,1,0)


#RSI
df['RSI'] = talib.RSI(df['Close'],14)
df['RSI_SIGNAL'] = np.where((df['RSI'] <= 30) & (df['RSI'].shift(1) > 30),
1,
0)

#Bollinger bands
upper, middle, lower = talib.BBANDS(df['Close'],timeperiod=10,
        # number of non-biased standard deviations from the mean
        nbdevup=2,
        nbdevdn=2,
        # Moving average type: simple moving average here
        matype=0)

df['BB_UPPER'] = upper
df['BB_MIDDLE'] = middle
df['BB_LOWER'] = lower

df['BB_SIGNAL'] = np.where(((df['Close']) <= (df['BB_LOWER'])),
1,
0)

#OBV
  
df['OBV'] = 0 
df['OBV_SIGNAL'] = np.where(((df['Close']) > (df['Close'].shift(1))), 
df['OBV'].shift(1) + df['Total Trade Quantity'],
df['OBV'].shift(1) - df['Total Trade Quantity'] )  

#MACD
macd, signal, hist = talib.MACD(df['Close'],fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD_SIGNAL'] = np.where((macd - signal) > 0,1,0)

#PRICE CHANNEL
df['PRICE_CHANNEL_MAX'] = pd.rolling_max(df['Close'],20,min_periods=20)
df['PRICE_CHANNEL_MIN'] = pd.rolling_min(df['Close'],20,min_periods=20)

df['PRICE_CHANNEL_SIGNAL'] = np.where(((df['Close']) > (df['PRICE_CHANNEL_MAX'].shift(1))),
1,
np.where(((df['Close']) < (df['PRICE_CHANNEL_MIN'].shift(1))),-1,0))

#WILLR
df['WILLR'] = talib.WILLR(high=df['High'], low=df['Low'],close=df['Close'],timeperiod=14)
df['WILLR_SIGNAL'] =  np.where(((df['WILLR']) <= (-80)),
1,
0)

#ULTOSC - Ultimate Oscillator
#df['ULTOSC'] = talib.ULTOSC(high=df['High'], low=df['Low'],close=df['Close'],timeperiod=14)

#STD DEV
df['STD'] =  pd.rolling_std(df['Close'],window=20, min_periods=20)

#MFI
#df['MFI'] = talib.MFI(high=df['High'], low=df['Low'],close=df['Close'],volume=df['Total Trade Quantity'],timeperiod=14)
#df['MFI_SIGNAL'] =  np.where(((df['MFI']) < (20)),
#1,
#np.where(((df['MFI']) >= (80)),-1,0))

df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
window = 5
df['PCT_CHANGE_SIGNAL'] = np.where((pd.rolling_max(df['Close'], window).shift(-window) - df['Close'])  > 0  , 1, 0)



new_cols =  ['SMA_CROSSING_SIGNAL','RSI_SIGNAL','BB_SIGNAL','MACD_SIGNAL','PRICE_CHANNEL_SIGNAL','WILLR_SIGNAL','PCT_CHANGE_SIGNAL']

newdf = df[new_cols]

newdf.to_csv('TRAIN.csv', sep=',',header=False, index=False)


