
# coding: utf-8

# In[10]:

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
import numpy
import pandas as pd
import numpy as np

df = pd.read_csv('IOC_TRAIN.csv', header = None)
#Calculate returns 1 , 0 , -1
#df['BSEC'] = np.where(df['BSEC'].shift(-1) - df['BSEC'] > 0, 1 , -1)

# Drop Date column
#df = df.drop(['DATE'], axis=1)
#Take out last column
#df1 = df[['BSEC']] 

#futureReturns = df.groupby('TICKER')['BSEC'].transform(lambda x: np.where(x.shift(-1) - x > 0, 1 , 0))
#print('Future Returns')
#print(futureReturns)
#Normalize data
#grouped = df.groupby(df.columns[0]).transform(lambda x: (x - x.mean()) / x.std())
#Replace last column
#grouped['BSEC'] = futureReturns
#pd.set_option('display.max_columns', 44)
#pd.set_option('display.max_rows', 500)

#Remove all na rows
#grouped = grouped.dropna()
#print('pandas df is')
#print(grouped)
#Change to numpy array
dataset = df.as_matrix()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# split into input (X) and output (Y) variables
X = dataset[:,0:6]
print(X.shape)
X = X.reshape(4601,1,6)
print(X.shape)
Y = dataset[:,6]
print(Y.shape)
print(Y)
# create model
model = Sequential()
#model.add(Dense(30, input_dim=10, init='normal', activation='relu'))
#model.add(Dense(10, init='normal', activation='relu'))
#model.add(Dense(1, init='normal', activation='sigmoid'))
model = Sequential()
model.add(LSTM(20,
               batch_input_shape=(1, 1, 6), return_sequences=False,
               stateful=True))
model.add(Dense(1, activation='sigmoid'))
# Compile model

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=1)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:




# In[ ]:



