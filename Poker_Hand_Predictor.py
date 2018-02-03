from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import csv       
import time
import sys

start_time = time.time()

batch_size = 16
num_classes = 10
epochs = 100
dropout_rate = 0.2
nodes = 512

def load_data(filename):
    """Loads in test and tranining data"""
    label_data = []
    data = []

    csv_file = csv.reader(open(filename, 'r'))

    for row in csv_file:
        
        if len(row) > 1:
            
            value = str(row[-1])
    
            # Adds out the class value
            label_data.append(value)

            # Removes the label and saves the data
            del row[-1]
            data.append(row)   

    data = group(data)
    return data, label_data    

def group(_list):
    """Groups suit and card number into a unique ID by concat"""

    total_list = []

    for row in _list:

        grouped_list = []

        for x in range(1, len(row), 2):
            grouped_list.append(row[x - 1] + row[x])

        total_list.append(grouped_list)

    return total_list
    
x_test, y_test = load_data('TestData.csv')
x_train, y_train = load_data('TrainingData.csv')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(nodes, activation='sigmoid', input_shape=(5,)))
model.add(Dropout(dropout_rate))
model.add(Dense(nodes, activation='sigmoid'))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(0.001),
              metrics=['accuracy'])

try:
    history = model.fit(np.array(x_train), np.array(y_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(np.array(x_test), np.array(y_test)))

    score = model.evaluate(np.array(x_test), np.array(y_test), verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("--- %s seconds ---" % (time.time() - start_time))

except KeyboardInterrupt:
    print('\n## Close Signal ##')


