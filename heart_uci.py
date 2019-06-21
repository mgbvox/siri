import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#These two lines deal with a bug on some mac computers.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Read in the data.
data = pd.read_csv('heart.csv')

#Get all non-target columns
x = data[[c for c in data.columns if c != 'target']]
#Get JUST the target column
y = data['target']

#Establish input shape for network (needs to be a tuple - syntax thing.)
input_shape = (x.shape[1],)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Make the middle section of hidden layers absurdly big:
layer_dims = [128]*20 #Set the 20 to 1 or 2 for a more appropriately-sized model


#Initialize the model.
model = Sequential()

#Add layers.
model.add(Dense(64, input_shape=input_shape, activation = 'relu'))
#Layers can be added via loops!
for l in layer_dims:
    model.add(Dense(l, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
#Final layer should be 1 output node;
#Sigmoid forces predictions to be between 0 and 1
model.add(Dense(1, activation = 'sigmoid'))

#Create the optimizer:
adam = Adam(lr=0.0001)

#Compile the model. Binary Crossentropy is the best loss for binary classification.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
#PS: For more classes, you'd use 'categorical-crossentropy' - look it up!

#Get a summary of the compiled model!
#Commented out for .py version.
#model.summary()

#Finally, fit the model to the data - and store the metrics in a history variable
history = model.fit(x, y, batch_size = 64, epochs = 100, shuffle = True)

#Save the model!
model.save('heart_uci_siri')

#View loss over time - should be decreasing.
plt.plot(history.history['loss'])
plt.title('Training Loss over Time')
plt.show()


model = keras.models.load_model('heart_uci_siri')

y_pred = model.predict(x)
plt.plot(y_pred, 'o')
plt.title('Unrounded Predictions')
plt.show()

plt.plot(np.round(y_pred), 'o')
plt.title('Predictions w/ Round')
plt.show()