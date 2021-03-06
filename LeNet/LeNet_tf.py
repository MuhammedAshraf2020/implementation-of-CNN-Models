
from keras.layers import Conv2D , AveragePooling2D , Input ,  Dense , Flatten 
from keras.models import Sequential


def LeNet_tf():
  model = Sequential()
  model.add(Input(shape = (32 , 32 , 3) ))
  model.add(Conv2D(filters = 6 , kernel_size = (5 , 5) , strides = (1 , 1)  , activation = "relu"))
  model.add(AveragePooling2D(2 , 2))
  model.add(Conv2D(filters = 16 , kernel_size = (5 , 5) , strides = (1 , 1)  , activation = "relu"))
  model.add(AveragePooling2D(2 , 2))
  model.add(Conv2D(filters = 128 , kernel_size = (5 , 5) , strides = (1 , 1)  , activation = "relu"))
  model.add(Flatten())
  model.add(Dense(84 , activation = "relu"))
  model.add(Dense(10 , activation = "softmax"))
  return model
