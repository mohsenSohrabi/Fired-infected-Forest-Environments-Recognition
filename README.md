# Fired Infected Envirnment Recognition Using Keras
**Note**: In order to run this project you should download dataset from [here](https://drive.google.com/file/d/1c1ADHs3uPMgQgoaAg36N--5-06FbFM23/view) and simply put it into "Dataset" folder (It is about 167MB and I can not put the file here directly to Dataset directory). 

It is a simple classification program. We use three models to classify data. 

### In this project we have 3 classes:
* Fired: There are lots of fired forest images here. In some of the pictures fire is obvious but in some other, the fire takes only a few parts of the picture. So that it is really hard to recognize fire in some samples. 

![some fired pictures ](https://github.com/mohsenSohrabi/Fired_Infected_Envirnment_Recognition/blob/master/sample_images/fired_samples.jpg)
___
* Infected: It includes pictures that there is some garbage accumulated in. Like the fired class, some of the samples are sophisticated to recognize.

 ![some infected images](https://github.com/mohsenSohrabi/Fired_Infected_Envirnment_Recognition/blob/master/sample_images/infected_samples.JPG)
 ___
* Clean: I includes a series of natural clean forest images.

 ![some natural images](https://github.com/mohsenSohrabi/Fired_Infected_Envirnment_Recognition/blob/master/sample_images/clean_samples.jpg)
 ___

### objective
The main objective of this program is classify fired, infected and clean places in correct class. 
In order to acheive the goal. We use following three models:
* ResNet
* VGG16
* VGG19 
