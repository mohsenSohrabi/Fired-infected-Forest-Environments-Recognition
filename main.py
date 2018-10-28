import numpy as np 
import myDS as ds 
import keras 
import RecogModels as mymodels
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define some hyperparameters and parameters
num_classes = 3
subtract_pixel_mean=True

   # define a simple function to set learning rate 
def lr_schedule(epoch):
    lr = 1e-3
    if(epoch> 150):
        lr*=1e-2
    elif(epoch>100):
        lr*=0.5e-1
    elif(epoch>50):
        lr*=1e-1
    return lr

batch_size = 16
epochs = 100
depth = 20 # it is only for resnet
model_name ="resnet" # vgg19 , resnet


# Read Data from Dataset 
X_train,y_train,X_val,y_val,X_test,y_test = ds.read_data()
assert np.shape(X_train)==(1923,120,160,3)

#Convert datatye to float32 and normalize Data
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_val /=255
X_test /= 255

# We have 3 classes so we change the labels to one-hot form 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# Subtract data from mean it can improve accuracy 
if subtract_pixel_mean==True:
  X_train_mean = np.mean(X_train,axis=0)
  X_train -= X_train_mean
  X_test -= X_train_mean
  X_val -= X_train_mean

# Define Input for network
input_img= Input(shape=X_train[0].shape)

# Get output from one of the defined models
if(model_name == "vgg16"):
    output = mymodels.VGG16(input_img)
elif(model_name == "vgg19"):
    output = mymodels.VGG19(input_img)
else:
    output = mymodels.resnet(input_img,depth)    





if model_name =="vgg16" or model_name == "vgg19":
    # we use SGD optimizer for vgg 
    optimizer=SGD(lr=lr_schedule(0), decay=1e-6, momentum=0.9, nesterov=True)
else:
    # we use Adam optimizer for resnet 
    optimizer=Adam(lr=lr_schedule(0))

model= Model(input_img, output)

# bind learning rate scheduler to the defined function
lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks=[lr_scheduler]

# Show a summary of model
model.summary()

# Complile the model
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,
              metrics=['accuracy'])

# Fit model with data
hist=model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_val, y_val),
              shuffle=True)

# Draw a plot for loss and accuracy 
plt.subplot(2,1,1)
x=np.arange(0,len(hist.history['val_loss']))
y1=hist.history['val_loss']
y2=hist.history['loss']
plt.plot(x,y1,'r--.')
plt.plot(x,y2,'b--.')

plt.subplot(2,1,2)
y3=hist.history['val_acc']
y4=hist.history['acc']
plt.plot(x,y3,'r--.')
plt.plot(x,y4,'b--.')

# Evaluate model with some unseen data
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
