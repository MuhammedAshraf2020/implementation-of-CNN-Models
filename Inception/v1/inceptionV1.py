
from keras.layers import Dense , Conv2D , concatenate  
from keras.layers import Flatten , Dropout , Input 
from keras.layers import AveragePooling2D , MaxPooling2D , GlobalAveragePooling2D
from keras.models import Sequential , Model


def inception_module(x , filter_1X1 , filter_3X3 , filter_5X5 ,
                     filter_3X3_reduce , filter_5X5_reduce , pool_proj ):

  X_1X1 = Conv2D(filters = filter_1X1 , kernel_size = (1 , 1) , padding = "same" , activation = "relu")(x)

  X_3X3 = Conv2D(filters = filter_3X3_reduce , kernel_size = (1 , 1)  , padding = "same" , activation = "relu")(x)
  X_3X3 = Conv2D(filters = filter_3X3 , kernel_size = (3 , 3)  , padding = "same" , activation = "relu")(X_3X3)

  X_5X5 = Conv2D(filters = filter_5X5_reduce , kernel_size = (1 , 1)  , padding = "same" , activation = "relu")(x)
  X_5X5 =  Conv2D(filters = filter_5X5 , kernel_size = (5 , 5)  , padding = "same" , activation = "relu")(X_5X5)

  X_MAX = MaxPooling2D(pool_size = (3 , 3) , strides = 1  , padding = "same")(x)
  X_MAX = Conv2D(filters = pool_proj , kernel_size = (1 , 1)  , padding = "same" , activation = "relu")(X_MAX)

  output = concatenate([X_1X1, X_3X3, X_5X5, X_MAX], axis=3 )

  return output


  def InceptionV1(classes = 1000 , input_shape = (224 , 224 , 3)):
  data_input = Input(shape = input_shape)

  output = Conv2D(filters = 64 , kernel_size = (7 , 7) , strides = 2 , padding = "same" , activation = "relu")(data_input)
  output = MaxPooling2D(pool_size = (3 , 3) , strides = 2  , padding = "same")(output)
  output = Conv2D(filters = 64 , kernel_size = (1 , 1 ) , padding = "same" , activation = "relu")(output)
  output = Conv2D(filters = 192 , kernel_size = (3 , 3) , padding = "same", activation = "relu")(output)
  output = MaxPooling2D(pool_size = (3 , 3) , strides = 2  , padding = "same")(output)

  
  output = inception_module(x = output ,
                            filter_1X1 = 64 ,
                            filter_3X3 = 128,
                            filter_5X5 = 32 ,
                            filter_3X3_reduce = 96 ,
                            filter_5X5_reduce = 16 , 
                            pool_proj = 32 )

  output = inception_module(x = output ,
                            filter_1X1 = 128 ,
                            filter_3X3 = 192 ,
                            filter_5X5 = 96 ,
                            filter_3X3_reduce = 128 ,
                            filter_5X5_reduce = 32 ,
                            pool_proj = 64 )

  output = MaxPooling2D(pool_size = (3 , 3) , strides = 2  , padding = "same")(output)


  output = inception_module(x = output ,
                            filter_1X1 = 192 ,
                            filter_3X3 = 208 ,
                            filter_5X5 = 48 ,
                            filter_3X3_reduce = 96 ,
                            filter_5X5_reduce = 16 ,
                            pool_proj = 64 )
  
  output1 = AveragePooling2D((5, 5), strides=3)(output)
  
  output1 = Conv2D(128, (1, 1), padding='same', activation='relu')(output1)
  
  output1 = Flatten()(output1)
  
  output1 = Dense(1024, activation='relu')(output1)
  
  output1 = Dropout(0.7)(output1)
  
  output1 = Dense(10, activation='softmax', name='auxilliary_output_1')(output1) 

  output = inception_module(x = output ,
                            filter_1X1 = 160 ,
                            filter_3X3 = 224 ,
                            filter_5X5 = 64 ,
                            filter_3X3_reduce = 112 ,
                            filter_5X5_reduce = 24 ,
                            pool_proj = 64 )

  output = inception_module(x = output ,
                            filter_1X1 = 128 ,
                            filter_3X3 = 256 ,
                            filter_5X5 = 64 ,
                            filter_3X3_reduce = 128 ,
                            filter_5X5_reduce = 24 ,
                            pool_proj = 64 )

  output = inception_module(x = output ,
                            filter_1X1 = 112 ,
                            filter_3X3 = 288 ,
                            filter_5X5 = 64 ,
                            filter_3X3_reduce = 144 ,
                            filter_5X5_reduce = 32 ,
                            pool_proj = 64 )

  
  output2 = AveragePooling2D((5, 5), strides=3)(output)
  
  output2 = Conv2D(128, (1, 1), padding='same', activation='relu')(output2)
  
  output2 = Flatten()(output2)
  
  output2 = Dense(1024, activation='relu')(output2)
  
  output2 = Dropout(0.7)(output2)

  output2 = Dense(classes , activation='softmax', name='auxilliary_output_2')(output2) 

  
  output = inception_module(x = output ,
                            filter_1X1 = 256  ,
                            filter_3X3 = 320 ,
                            filter_5X5 = 128 ,
                            filter_3X3_reduce = 160 ,
                            filter_5X5_reduce = 32 ,
                            pool_proj = 128 )

  output = MaxPooling2D(pool_size = (3 , 3) , strides = 2  , padding = "same")(output)


  output = inception_module(x = output ,
                            filter_1X1 = 256 ,
                            filter_3X3 = 320 ,
                            filter_5X5 = 128 ,
                            filter_3X3_reduce = 160 ,
                            filter_5X5_reduce = 32 ,
                            pool_proj = 128 )

  output = inception_module(x = output ,
                            filter_1X1 = 384  ,
                            filter_3X3 = 384 ,
                            filter_5X5 = 128 ,
                            filter_3X3_reduce = 192 ,
                            filter_5X5_reduce = 48 ,
                            pool_proj = 128 )

  output = GlobalAveragePooling2D()(output)

  output = Dropout(0.4)(output)

  output = Dense(classes , activation = "softmax")(output)

  model = Model(data_input , [output , output1 , output2])
  return model