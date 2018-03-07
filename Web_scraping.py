#construct NN Data Web Scrapping from google.com 
#save parameters so we can use many times for each obj 
#convert images to h5 dataset 
#scraping images data from web and save it 
#reshaping images (64,64,3) or other uniform shape and save it 

#web packs
import requests #lib to perform HTTP request from any site
from bs4 import BeautifulSoup #lib to perform formattion for strings
import urllib.request
#os packs
import os 
import time 
#imgae packs 
import numpy as np 
from scipy import ndimage 
import scipy 


def scraping (Obj,num_wanted=200):	
	print ('starting scraping request ...' )

	# Obj = "pen"
	item = 1
	#making directory 
	directory = "Images/{Obj}/".format(Obj=Obj)
	directory = os.path.join(os.path.dirname(__file__),directory) #the dir of current file 
	if not os.path.exists(directory):
		os.makedirs(directory)

	print ("{Obj} Images' being saved in {dir}".format(Obj=Obj,dir=directory))

	num_items = 0 
	while num_items < num_wanted :
		url = "https://www.google.com.eg/search?q={obj}&source=lnms&tbm=isch&start={num}".format(obj=Obj,num=num_items)
		num_items += 20 
		site_req =requests.get(url)
		print ("scrape page : %i @ response %i "%(num_items/20 ,site_req.status_code)) #200 if request resposed
		site_soup = BeautifulSoup(site_req.text,'html.parser')
		# print(dir(site_soup))
		# print(site_soup)

		div_images = site_soup.findAll('img')
		# print(len(div_images))
		# print(div_images)

		images = []
		for i in div_images : 
			src = i['src']
			images.append(src)

		# print(len(images))		
		# print (images)

		for img in images:
			try :
				PATH = "{dir}/ img{i}.jpg".format(dir = directory , i = str(item))
				urllib.request.urlretrieve(img,PATH)
				item += 1 
			except:
				pass 

		time.sleep(1) #sleep 5 sec so can load images 
	reshaping(Obj)

def reshaping(Obj,dim=128):
	num_px = dim 
	PATH = "Images/"+Obj+"/"
	PATH = os.path.join(os.path.dirname(__file__),PATH) #the dir of current file 
	images = os.listdir(PATH)
	for my_image in images :
		fname = PATH + my_image
		print (fname)
		image = np.array(ndimage.imread(fname, flatten=False)) 
		my_image = scipy.misc.imresize(image, size=(num_px,num_px))#.reshape((num_px*num_px*3,1))
		scipy.misc.imsave(fname,my_image)


def load_data(Obj,false="false",train_ratio=0.8):
	PATH = "Images/"+Obj+"/"
	PATH = os.path.join(os.path.dirname(__file__),PATH) #the dir of current file 
	try:
		images = os.listdir(PATH)
	except:
		print("No Data Scraping ")
		scraping(Obj)
		images = os.listdir(PATH)

	n_imgs = len(images) 
	print ("number of true dataset {n}".format(n=n_imgs))

	max_train = int(train_ratio * n_imgs)
	max_test = n_imgs - max_train 

	false_PATH = "Images/"+ false + "/"
	false_PATH = os.path.join(os.path.dirname(__file__),false_PATH) #the dir of current file 
	false_images = os.listdir(false_PATH)
	max_false_train_sapce  = int(3)

	if images is not None :
		train_set_x_orig = []
		train_set_y_orig = []

		count = 0 
		# print ("loading {n} {Obj} Train images...".format(n=max_train,Obj=Obj))
		for my_image in images[0:max_train] :
			fname = PATH + my_image
			train_set_x_orig.append(np.array(ndimage.imread(fname, flatten=False,mode='RGB'))) # your train set features .. flattern ture mean gray 
			train_set_y_orig.append(np.array(1)) # your train set label
			if count % max_false_train_sapce == 0: 
				fname = false_PATH + false_images[int (count/max_false_train_sapce)] 
				train_set_x_orig.append(np.array(ndimage.imread(fname, flatten=False,mode='RGB'))) # your train set features
				train_set_y_orig.append(np.array(0)) # your train set label
			count += 1 
		print (np.shape(train_set_x_orig))
		#print (train_set_y_orig.shape) 	
		train_set_x_orig = np.array(train_set_x_orig)
		train_set_y_orig = np.array(train_set_y_orig)
		print ("loaded {n} {Obj} Train images with {m} false {false}".format(n=train_set_x_orig.shape[0],Obj=Obj,m = train_set_x_orig.shape[0]-max_train, false=false))

		print (train_set_x_orig.shape)
		print (train_set_y_orig.shape)

		test_set_x_orig = []
		test_set_y_orig = []
		print ("loading {n} {Obj} Test images...".format(n=max_test,Obj=Obj))
		for my_image in images[max_train:max_train+max_test] :
			fname = PATH + my_image
			test_set_x_orig.append(np.array(ndimage.imread(fname, flatten=False,mode='RGB'))) # your train set features
			test_set_y_orig.append(np.array(1)) # your train set label
			if count % max_false_train_sapce == 0: 
				fname = false_PATH + false_images[int (count/max_false_train_sapce)] 
				test_set_x_orig.append(np.array(ndimage.imread(fname, flatten=False,mode='RGB'))) # your train set features
				test_set_y_orig.append(np.array(0)) # your train set label
			count += 1 

		test_set_x_orig = np.array(test_set_x_orig)
		test_set_y_orig = np.array(test_set_y_orig)
		print ("loaded {n} {Obj} Test images with {m} false {false}".format(n=test_set_x_orig.shape[0],Obj=Obj,m = test_set_x_orig.shape[0]-max_test, false=false))

		# print (test_set_x_orig.shape)
		# print (test_set_y_orig.shape)

		classes = np.array([false,Obj]) # the list of classes 0 non-cat , 1 cat 
		print ("classes") 
		print(classes)

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	# print (train_set_y_orig.shape)
	# print (test_set_y_orig.shape)


	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# scraping("car",200)
# reshaping("dog")
# load_data("rat")
# reshaping("rat")
# print(os.listdir("Images/car")[0:10])
