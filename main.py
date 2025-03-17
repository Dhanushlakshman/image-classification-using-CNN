# -*- coding: utf-8 -*-


import tensorflow as tf
import matplotlip.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = training_datagen.flow_from_directory("D:\CNN\CNN\training_set", target_size=(64,64), batch_size=32, class_mode='binay')

test_datagen = ImageDataGenerator (rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_set = training_datagen.flow_from_directory("D:\CNN\CNN\test_set", target_size=(64,64), batch_size=32, class_mode='binary')

cnn = tf.keras.model.sequential()
cnn.add(tf.keras.layers.ConV2D(filters=32,kernal_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernal_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
cnn.fit(training_set,validation_data=test_set,epochs=5)

import numpy as np 
from keras.preprocessing import image
test_image = tf.keras.utils.load.img_to_array("D:\CNN\CNN\single_prediction/cat_or_dog_1.jpg", target_size=(64,64))
test_image =tf.keras.utils.img_to_array(test_image)
test_image = np.extend_dims(test_image,axis=0)
result= cnn.prediction(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction= 'dog'
else:
    prediction='cat'
print (prediction)

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(rescale=1./55, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set =training_datagen.flow_from_directory('D:\CNN\CNN\training_set', target_size=(64,64), batch_size=32, class_mode='binary')

test_datagen =ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_set = training_datagen.flow_from_directory('D:\CNN\CNN\test_set', target_size=(64,64), batch_size=32, class_mode= 'binary')

cnn= tf.keras.models.sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernal_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

cnn.add(tf.keras.Conv2D(filters=32, kernal_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, acctivation='relu'))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer='adam', loss='binary_crossentropy', matrics=['acc'])
cnn.fit(training_set, validation_data=test_set,  epochs=5)


import numpy as np
from keras.preprocessing import image
test_image= tf.keras.utills.load_img('D:\CNN\CNN\single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
test_image.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:                              
    prediction ='cat'
else:    
    prediction= 'dog'
print(prediction)


import matplotlip.pyplot as tf
history = cnn.fit(training_set,alidation_set= test_set, epochs=5)
test_loss,test_acc = cnn.evaluate(test_set)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


#plot accuracy
plt.figure(dpi=300)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train','val'],loc='best')
plt.tight_layout()
plt.show()

#plot loss
plt.figure(dpi=300)
plt.polt(history.history['loss'])
plt.plot(history.history['val_acc'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['train','val'],loc="best")
plt.tight_layout()
plt.show()
