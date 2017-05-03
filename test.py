import numpy as np
import random
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt
import recog_char
import Main_cnn
from os import walk
import os


# recognize all files in the root directory, all files should all be image file
count = 0;
root = "/Users/fangyanxu/Documents/car_plate/Character_Recognition/my_data/2/"

for (path, dirname, name) in walk(root):
	for file in name:

		print(root, file)
		file = os.path.join(root, file)
		
		result = Main_cnn.run(file)

# test the error rate for a set of sample data, given contour matrix and the label
# ClassToChar = {0:48, 1:49, 2:50, 3:51, 4:52, 5:53, 6:54, 7:55, 8:56, 9:57, 10:65, 11:66, 12:67, 
#                13:68, 14:69, 15:70, 16:71, 17:72, 18:73, 19:74, 20:75, 21:76, 22:77, 23:78, 24:79, 25:80,
#                26:81, 27:82, 28:83, 29:84, 30:85, 31:86, 32:87, 33:88, 34:89, 35:90}
# sess = tf.Session()
# recog_char.load_model(sess, "my_model.meta", "my_model.data-00000-of-00001")
# all_var = tf.get_collection('validation_nodes')
# test_data = np.loadtxt("flattened_images_test_restore.txt")
# test_label = np.loadtxt("classifications_test.txt")
# size = test_data.shape[0]
# 	if(ClassToChar[result[0]] != test_label[i]):
# 		print(ClassToChar[result[0]], test_label[i])
# 		count += 1

# acc = count / size

# print(acc)