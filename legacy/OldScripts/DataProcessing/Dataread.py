## Reading data from the raw data

import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import svm
import random
import math

def magic_box(x, y):
	return [42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42,42];



# arg0: the maximum number of data wa
def main():

	print " \n Enter Main";

	# assume reading 500 sets of data as feature.
	processedData = [];
	data_index = 0;
	x_index = 0;


	yaw_block_x = [];
	pitch_block_x = [];
	roll_block_x = [];
	block_y = [];


	# 10 is the maximum number of data we want to create, subject to the 
	# number of raw data available. Candidate for parameter of main()
	while data_index < 10:
		if(x_index >= len(x)):
			break;
		if x[x_index][0] == 666:
			
			x_index += 1;
			
			temp_index = 0;
			
			while x[x_index][0] != 999:
				yaw_block_x.append(x[x_index][0]);
				pitch_block_x.append(x[x_index][1]);
				roll_block_x.append(x[x_index][2]);
				block_y.append(temp_index);
				temp_index += 1;
				
				print x[x_index][0];
				
				x_index += 1;
				
				if(x_index >= len(x)):
					break;

			processedData.append(magic_box(yaw_block_x, block_y));
			processedData[data_index] = np.concatenate((processedData[data_index], 
				magic_box(pitch_block_x, block_y)));
			processedData[data_index] = np.concatenate((processedData[data_index], 
				magic_box(roll_block_x, block_y)));

			yaw_block_x = [];
			yaw_block_y = [];
			data_index += 1;
		else:
			x_index+=1;


	print "The resultant shape of the data would look like: (# of data, features of each data) = " + str(np.array(processedData).shape);



if __name__ == '__main__':

	# data = read Data based on its file type
	# data X has should be in a n by 3 matrix, and data Y is label with 1 and -1
	# assume the 666 is the starting, and 999 is ending.

	# x = data['X'];
	# y = data['Y'];

	# y = y.astype(None).transpose()[0]

	x = [np.array([121212,121212,121212])];
	y = [1]

	for i in range (100):
		x.append(np.array([int(math.floor(random.random()*180)),int(math.floor(random.random()*180)),int(math.floor(random.random()*180))]));
		y.append(random.choice([1,-1]));
		if(i % 20 == 0):
			x.append(np.array([999,999,999]));
			y.append(random.choice([1,-1]));
			x.append(np.array([42222,42222,42222]))
			y.append(1)
			x.append(np.array([666,666,666]))
			y.append(1)
		

	x = np.array(x);
	y = np.array(y);
	print x;
	print y;
	print x.shape;
	print y.shape;

	main();
