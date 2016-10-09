# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
import numpy as np

df = pd.read_csv('truncated_nn_stocks.csv')
#Calculate returns 1 , 0 , -1
#df['BSEC'] = np.where(df['BSEC'].shift(-1) - df['BSEC'] > 0, 1 , -1)

# Drop Date column
df = df.drop(['DATE'], axis=1)
#Take out last column
#df1 = df[['BSEC']] 

futureReturns = df.groupby('TICKER')['BSEC'].transform(lambda x: np.where(x.shift(-1) - x > 0, 1 , 0))
#print('Future Returns')
#print(futureReturns)
#Normalize data
grouped = df.groupby(df.columns[0]).transform(lambda x: (x - x.mean()) / x.std())
#Replace last column
grouped['BSEC'] = futureReturns
pd.set_option('display.max_columns', 44)
pd.set_option('display.max_rows', 500)

#Remove all na rows
grouped = grouped.dropna()
#print('pandas df is')
#print(grouped)
#Change to numpy array
dataset = grouped.as_matrix()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# split into input (X) and output (Y) variables
X = dataset[:,0:10]
Y = dataset[:,10]
print(Y)
# create model
model = Sequential()
model.add(Dense(20, input_dim=10, init='uniform', activation='relu'))
model.add(Dense(10, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
