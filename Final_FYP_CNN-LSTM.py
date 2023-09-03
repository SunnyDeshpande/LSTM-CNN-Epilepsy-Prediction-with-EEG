# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 14:54:35 2018

@author: Sunny
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 23:06:46 2018

@author: Sunny
"""
import keras, numpy
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LSTM
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import load_model

###############################################################################
trainingset = numpy.loadtxt("TrainCNN2.dat", delimiter=",")
testingset = numpy.loadtxt("TestCNN.dat", delimiter=",")
validationset = numpy.loadtxt("ValidateCNN2.dat", delimiter=",")

Xtrain = trainingset[:,0:len(trainingset[0])-2]
Ytrain = trainingset[:,len(trainingset[0])-2:len(trainingset[0]-1)]
Xtest = testingset[:,0:len(testingset[0])-2]
Ytest = testingset[:,len(testingset[0])-2:len(testingset[0]-1)]
Xvalid = validationset[:,0:len(validationset[0])-2]
Yvalid = validationset[:,len(validationset[0])-2:len(validationset[0]-1)]

Xtrain = numpy.expand_dims(Xtrain, axis=2)
Xtest = numpy.expand_dims(Xtest, axis=2)
Xvalid = numpy.expand_dims(Xvalid, axis=2)
###############################################################################


modelX = load_model('Final_FYP_CNN-LSTM.h5')
modelY = modelX
history = modelY.fit(Xtrain, Ytrain, epochs=10, batch_size=64, verbose=1, validation_data=(Xvalid, Yvalid), shuffle=True)
score = modelY.evaluate(Xtest, Ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#predictions = modelY.predict(Xtest)
## TRY SGD OPTIMIZER SOON
#print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()



