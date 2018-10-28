import keras
from keras.models import Model
from keras.layers import Input,Dense, Activation, Dropout, Flatten, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
import os 


def resnet(inputs, depth=20, num_classes=3):
    num_filters=16
    num_res_blocks= int((depth-2)/6)
    
    x= resnet_block(inputs=inputs)
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides=1
            
            if stack>0 and res_block==0:
                strides=2
            
            
            
            y=resnet_block(inputs=x,
                          num_filters=num_filters,
                          strides=strides)
            y=resnet_block(inputs=y,
                           num_filters=num_filters,
                           activation=None)
            
            
            if stack>0 and res_block==0:
                x=resnet_block(inputs=x,
                               num_filters=num_filters,
                               kernel_size=1,
                               strides=strides,
                               activation=None,
                               batch_normalization=False
                              )
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
            
        num_filters*=2  
        
        
    x= AveragePooling2D(pool_size=8)(x)
    y=Flatten()(x)
    outputs=Dense(num_classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(y)
    #model =Model(inputs=inputs, output=outputs)
    return outputs
          

def resnet_block(inputs, 
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    
    conv= Conv2D(num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))
    x=inputs
    if conv_first:
        x=conv(x)
        if batch_normalization:
            x= BatchNormalization()(x)
        if activation is not None:
            x=Activation(activation)(x)
    else:
        if batch_normalization:
            x=BatchNormalization()(x)
        if activation is not None:
            x= Activation(activation)(x)
        x=conv(x)
    return x 


def VGG16(input_img,num_classes=3):
  
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
  x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
  x = MaxPooling2D((2, 2), strides=(2, 2))(x)
  
  x= Conv2D(128, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(128, (3,3), activation='relu', padding='same')(x)
  x= MaxPooling2D((2,2), strides=(2,2))(x)
 
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x= MaxPooling2D((2,2), strides=(2,2))(x)
  
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= MaxPooling2D((2,2), strides=(2,2))(x)
  
  x= Flatten()(x)
  x= Dense(4096, activation='relu')(x)
  x= Dense(4096, activation='relu')(x)
  x= Dense(num_classes, activation='softmax')(x)
  
  return x


def VGG19(input_img, num_classes=3):
  
  x= Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
  x= Conv2D(64, (3,3), activation='relu', padding='same')(x)
  x= MaxPooling2D((2,2), strides=(2,2))(x)
  
  x= Conv2D(128, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(128, (3,3), activation='relu', padding='same')(x)
  x=MaxPooling2D((2,2), strides=(2,2))(x)
  
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)
  x=MaxPooling2D((2,2), strides=(2,2))(x)
  
  
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x=MaxPooling2D((2,2), strides=(2,2))(x)
  
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)
  x=MaxPooling2D((2,2), strides=(2,2))(x)
  
  x= Flatten()(x)
  x=Dense(4096, activation='relu')(x)
  x=Dense(4096, activation='relu')(x)
  x=Dense(num_classes,activation='softmax')(x)
  
  
  return x
  