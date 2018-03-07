#run python shell with bash command (python -i)
#install packages with bash command (pip instal pack)
#insalling pillow , numpy , opencv-python (cv2), matplotlib , scipy 
# import time
#from PIL import Image
# import h5py

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import csv 

from NN_Model import L_layer_model , predict 
from Web_scraping import load_data 

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

start = time.clock() 
#dataset .. (object , #train , #test , rate.false , shape(X,Y) , )
train_x_orig, train_y, test_x_orig, test_y, classes = load_data("dog")
print ("Loading data time : %f sec"%((time.clock() - start)))

# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
# Example of a picture
index = int(np.random.rand(1)*m_train)

plt.imshow(train_x_orig[index])
plt.title("Class =" + "y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]] + " picture.")
plt.show()
# print ("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]] + " picture.")

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

start = time.clock() 
layers_dims = [train_x.shape[0],7,1]
# layers_dims = [12288, 20, 7, 5, 1] #  4-layer model
#			two_layer_...[12288,5,1]
parameters = L_layer_model(train_x, train_y, layers_dims = layers_dims ,learning_rate=0.0075, num_iterations = 1000, print_cost=True)
print ("Learning token time : %f sec"%((time.clock() - start)))

start = time.clock() 
print("Train:")
predictions_train = predict(train_x, train_y, parameters)
print("Test:")
predictions_test = predict(test_x, test_y, parameters)
print ("Prediction Train / Test token time : %f sec"%(time.clock() - start))

start = time.clock() 
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [0] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "Images/" + my_image
fname = os.path.join(os.path.dirname(__file__),fname) #the dir of current file 
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

print ("Prediction token time : %f sec"%(time.clock() - start))

plt.imshow(image)
plt.title("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),]+  "\" picture.")
# print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),]+  "\" picture.")
plt.show()

#save parameters 
fieldnames = {}
L = len(parameters) // 2 
for l in range (L) :
    fieldnames["W"+str(l+1)] = 0 #writting using dictionary
    fieldnames["b"+str(l+1)] = 0 

PATH = "parameters_data/data6"
PATH = os.path.join(os.path.dirname(__file__),PATH) #the dir of current file 
with open(PATH+".csv",'w') as csvfile :   
    writer = csv.writer(csvfile)
    writer.writeheader()  #to write the header of the dataset  or not dont use it 
    for l in range(L):
     	writer.writerow({"W" + str(l + 1):parameters["W" + str(l + 1)]})
     	writer.writerow({"b" + str(l + 1):parameters["b" + str(l + 1)]})
