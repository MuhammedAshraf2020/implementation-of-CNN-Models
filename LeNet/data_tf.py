import keras
import numpy as np
from keras.utils import to_categorical
from keras.datasets.cifar10 import load_data
def data_tf():
  (x_train , y_train) , (x_test , y_test) = load_data()
  x_train = x_train.reshape(x_train.shape + (1,)) 
  x_test  = x_test.reshape(x_test.shape + (1, )) 
  print("x train shape = " , x_train.shape)
  print("x test  shape = " , x_test.shape)
  y_train = to_categorical(y_train)
  y_test  = to_categorical(y_test)
  print("y train shape = " , y_train.shape)
  print("y test  shape = " , y_test.shape)
  return (x_train , y_train) , (x_test , y_test)
