#!/usr/bin/env python
# coding: utf-8

# # main

# In[8]:


import dataSet as ds
import selectivesearch
import numpy as np
import utils
import model
import solver
import tensorflow as tf
from PIL import Image
import argparse
import slidingWindow as sw
import augmentation
from xml.etree.ElementTree import ElementTree

'''
selective search에 들어갈 이미지는 np객체
'''

parser = argparse.ArgumentParser()
parser.add_argument("-training", "--training", type=bool, default=False)
parser.add_argument("-epochs", "--epochs", type=int, default=2000)
args = parser.parse_args()

print("Phase 0 : Load data")
data = ds.dataSet()
j = ds.jsonObject('para.json')
param = j.para

target_index = []
for cl in param['label_dic']:
    if cl in param['target_list']:
        target_index.append(param['label_dic'][cl])

training = args.training

if training:
    data_dir = 'image'
else:
    data_dir = None
data.load_data(dir=data_dir, test_dir='test_image')

#data.grayscale()
data.augmentation()
#data.edge()

if training:
    data.sep_train_test()
_, *model_input_size = data.img_set.shape

sess = tf.Session()
model = model.four_layer_CNN(sess=sess, input_shape=data.img_set.shape, n_class=len(param['label_dic']))
sv = solver.Solver(sess=sess, name='op', model=model, dataset=data, optimizer=tf.train.AdamOptimizer)

epochs = args.epochs
batch_size = param['batch_size']
learning_rate = param['lr']
path = '/home/paperspace/Dropbox/krlee/easy-yolo/devkit/2017/Images'

sess.run(tf.global_variables_initializer())

if not training:
    print("Phase 1 : Load model")
    sv.model_load()

else:
    print("Phase 1 : Train model")
    sv.train(epoch=epochs, batch_size=batch_size, lr=learning_rate, print_frequency=100)
    sv.model_save()

def cf(x):
    return sv.predict(x)[0] in target_index

def score_cf(imag):
    scores = np.array(sv.predict_softmax_score(imag))
    return (np.argmax(scores) in target_index) and (np.max(scores) > param['score_bottom_line'])

for i, img in enumerate(data.test_img):
    print("{}th image in progress".format(i))
    temp_img, matrix = sw.sliding_window(img, score_cf, window_size=param['ss_dic']['window_size'], stride=16, boundary=param['sw_dic']['boundary'])
    temp_img.save('./test_result/sw'+str(i)+'.jpg', 'jpeg')
    img.save('./Images/sw'+str(i)+'.jpg', 'jpeg')
    box_set = sw.make_box(matrix)
    for box in box_set:
        img = utils.draw_rectangle(img, box, label='block')
    xml = utils.box_to_xml('sw'+str(i)+'.xml', path+str(i)+'.jpg', img.size+(3,), box_set)
    ElementTree(xml).write('./Annotation/sw'+str(i)+'.xml')
    txt = utils.box_to_txt([0 for _ in range(len(box_set))], box_set, img.size)
    with open('./labels/sw'+str(i)+'.txt', 'w') as f:
        f.write(txt)
    img.save('./test_result/sw_box'+str(i)+'.jpg', 'jpeg')


# # dataSet

# In[9]:


import pickle
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import json
import augmentation
import utils
from skimage import feature

class jsonObject():
	def __init__(self, dir):
		with open(dir, 'r') as f:
			self.para = json.load(f)

class dataSet():
	'''
	 - Data 원본을 저장하고, 매번 필요한 경우 리사이징을 하자?
	'''
	def __init__(self, para_filename='para.json'):
		self.train_image = np.empty(0)
		self.train_label = np.empty(0)
		self.test_image = np.empty(0)
		self.test_label = np.empty(0)
		self.label_dic = {}
		self.ori_img_set = []
		self.img_set = []
		self.label_set = []
		self.width, self.height, self.color = 0, 0, 0

		self.batch_size = 32
		self.batch_index = []
		self.n_example = 0
		self.n_class = 0
		self.load_para(para_filename)

	def load_para(self, filename='para.json'):
		j = jsonObject(filename)
		js = j.para
		self.label_dic = js['label_dic']
		self.resizing = js['resizing']
		self.target_list = js['target_list']
		self.score_bottom_line = js['score_bottom_line']
		self.test_rate = js['test_rate']
		self.width, self.height = self.resizing
		self.n_size_reset()

	def n_size_reset(self):
		self.test_size = int(self.n_example * self.test_rate)
		self.train_size = self.n_example - self.test_size

	def load_data(self, dir=None, test_dir=None): # 현경
		sys_file = []
		self.img_set = np.empty((0,) + tuple(self.resizing) + (3,))

		# 만약 label_dic이 비어있으면, 폴더 이름을 불러옴
		if self.label_dic == {}:
			for i, filename in enumerate(os.listdir(os.getcwd() + '/' + dir)):
				self.label_dic[filename] = i

		if dir is not None:
			for label in self.label_dic:
				img_dir = os.getcwd()+ '/' + dir + '/' + label

				for path, _, files in os.walk(img_dir):
					for file in files:
						img_dir = path + '/' + file
						try:
							img = Image.open(img_dir)
						except OSError as e:
							sys_file.append(e)
						else:
							# 만약 image가 RGB 포맷이 아닐경우, RGB로 변경
							if not img.format == "RGB":
								img = img.convert("RGB")

							self.ori_img_set.append(img)
							self.img_set = np.append(self.img_set, np.array([np.array(img.resize(self.resizing))]), axis=0)
							self.label_set = np.append(self.label_set, self.label_dic[label])


			self.n_example, self.width, self.height, self.color = self.img_set.shape
			self.n_size_reset()
			self.n_class = len(self.label_dic)

			
		else:
			self.img_set = np.empty((0,) + tuple(self.resizing) + (3,))

		if test_dir is not None:
			self.test_img = []
			img_dir = os.getcwd() + '/' + test_dir

			for path, _, files in os.walk(img_dir):
				for file in files:
					img_dir = path + '/' + file
					try:
						img = Image.open(img_dir)
						# 사이즈 일괄조정을 위해 하드코딩함
					except OSError as e:
						sys_file.append(e)
					else:
						# 만약 image가 RGB 포맷이 아닐경우, RGB로 변경
						if not img.format == "RGB":
							img = img.convert("RGB")

						# 만약 image가 정사각형이 아닐경우, 정사각형 두 개로 자른다.
						if img.size[0] > img.size[1]:
							img1 = img.crop((0,0,img.size[1],img.size[1]))
							img2 = img.crop((img.size[0]-img.size[1],0,img.size[0],img.size[1]))
							img1 = img1.resize((416,416))
							img2 = img2.resize((416,416))
							self.test_img.append(img1)
							self.test_img.append(img2)
						elif img.size[0] < img.size[1]:
							img1 = img.crop((0,0,img.size[0],img.size[0]))
							img2 = img.crop((0,img.size[1]-img.size[0],img.size[0],img.size[1]))
							img1 = img1.resize((416,416))
							img2 = img2.resize((416,416))
							self.test_img.append(img1)
							self.test_img.append(img2)
						else:
							img = img.resize((416,416))
							self.test_img.append(img)

	def sep_train_test(self):
		'''
			train / test로 분할 함, size는 para.json에 저장된 값 부름
		:return:
		'''
		print('debug : ', self.img_set.shape)
		ind = np.random.randint(self.n_example, size=self.train_size+self.test_size)
		self.train_image = self.img_set[ind[:self.train_size]]
		self.train_label = self.label_set[ind[:self.train_size]]
		self.test_image = self.img_set[ind[self.train_size:]]
		self.test_label = self.label_set[ind[self.train_size:]]

	def _resize(self, size=None):
		# size를 따로 받지 않았을 경우, para.json에 저장되어 있는 값으로 resizing
		if size is None:
			size = self.resizing
		resizing_size = tuple(size) + (self.img_set[0].shape[2],)
		self.img_set = np.empty((0,) + resizing_size)
		for img in self.ori_img_set:
			temp = np.array(img.resize(size)).reshape(((-1,) + resizing_size))
			self.img_set = np.append(self.img_set, temp, axis=0)

	def save(self, dir): # 현경
		temp_dataset = {
			'train_image' : self.train_image,
			'train_label' : self.train_label,
			'test_image' : self.test_image,
			'test_label' : self.test_label
		}
		with(open(dir, 'wb')) as f:
			pickle.dump(temp_dataset, f)

	def one_hot_encoding(self):  # 현경
		temp_list = np.zeros((self.n_example, self.n_class))
		temp_list[np.arange(self.n_example), np.array(self.label_set)] = 1
		self.label_set = temp_list

	def one_hot_decoding(self):
		self.label_set = [np.where(i==1)[0][0] for i in self.label_set]

	def print_informaton(self): # 광록
		print('train_data : {}, test_data : {}'.format(self.train_size, self.test_size))
		print('image_size : ({}, {})'.format(self.width, self.height))
		for i in np.unique(self.train_label):
			print('sample image : {}'.format(i))
			ind = np.where(self.train_label == i)
			self.sample_image(ind[0][0])

	def mini_batch(self, batch_size): # 광록
		pass

	def _make_batch_index(self):
		self.batch_index = np.split(np.random.permutation(self.train_size), np.arange(1, int(self.train_size // self.batch_size) + 1) * self.batch_size)
		if self.batch_index[-1] == []:
			self.batch_index = self.batch_index[:-1]

	def next_batch(self): # 광록
		if len(self.batch_index) == 0:
			self._make_batch_index()
		ind = self.batch_index[0]
		self.batch_index = self.batch_index[1:]
		return self.train_image[ind], self.train_label[ind]

	def grayscale(self):
		print('---grayscale---')
		RGB_to_L = np.array([[[[0.299,0.587,0.114]]]])
		self.img_set = np.sum(self.img_set * RGB_to_L, axis=3, keepdims=True)
		_, self.width, self.height, self.color = self.img_set.shape

	def augmentation(self):
		print('---Image augmentation---')
		self.img_set = augmentation.augmentation(self.img_set,3,3,3)
		print('debug : ', self.img_set.shape)
		_, self.width, self.height, self.color = self.img_set.shape
		self.n_size_reset()

	def hog(self):
		print('---HOG---')
		img_to_hog = lambda img: utils.hog(img)
		self.img_set = np.array(list(map(img_to_hog, self.img_set)))
		with open('para.json', 'r') as f:
			js = json.load(f)
			block_wid = js['hog_dic']['block_width']
			block_hei = js['hog_dic']['block_height']
			n_ori = js['hog_dic']['n_orient']
		self.img_set = self.img_set.reshape((-1, n_ori, int(self.width / block_wid + 0.5), int(self.height / block_hei + 0.5)))
		self.img_set = np.transpose(self.img_set, (0,2,3,1))
		self.n_example, self.width, self.height, self.color = self.img_set.shape
		self.n_size_reset()

	def edge(self):
		print('---EDGE---')
		self.img_set = np.sum(self.img_set, axis=3, keepdims=False)
		self.img_set = np.array(list(map(feature.canny, self.img_set)))
		self.img_set = self.img_set.reshape(self.img_set.shape+(1,))
		self.n_example, self.width, self.height, self.color = self.img_set.shape
		self.n_size_reset()

	def sample_image(self, index=0):
		if self.train_image.shape[3] == 3:
			plt.imshow(self.train_image[index]/255)
		elif self.train_image.shape[3] == 1:
			plt.imshow(self.train_image[index].reshape((self.width, self.height))/255)
		plt.show()

	def object_detect(self):
		pass


# # model

# In[10]:


import tensorflow as tf

class two_layer_CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L2_flat = tf.reshape(L2, [-1, w * h * 64])

        W3 = tf.get_variable("W3", shape=[w * h * 64, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L2_flat, W3) + b
        self._hypothesis = tf.nn.softmax(logits)
        self._prediction = tf.argmax(input=logits, axis=-1)

        self._xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(self._xentropy, name="loss")

class three_layer_CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W3 = tf.get_variable(name="W3", shape=[3, 3, 64, 32], dtype=tf.float32, initializer=initializer)
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = activation_fn(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L3_flat = tf.reshape(L3, [-1, w * h * 32])

        W4 = tf.get_variable("W4", shape=[w * h * 32, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L3_flat, W4) + b
        self._hypothesis = tf.nn.softmax(logits)
        self._prediction = tf.argmax(input=logits, axis=-1)

        self._xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(self._xentropy, name="loss")


class four_layer_CNN:
    '''
     - X : 4차원 이미지 데이터, 사이즈에 유동적
     - y : 1차원 class 데이터
    '''
    def __init__(self, sess, input_shape, n_class,
                 activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()):

        _, w, h, d = input_shape
        self._sess = sess
        self._x = tf.placeholder(tf.float32, [None, w, h, d])
        self._y = tf.placeholder(tf.int32, [None])
        y_one_hot = tf.one_hot(self._y, n_class)
        y_one_hot = tf.reshape(y_one_hot, [-1, n_class])

        W1 = tf.get_variable(name="W1", shape=[3, 3, d, 32], dtype=tf.float32, initializer=initializer)
        L1 = tf.nn.conv2d(self._x, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = activation_fn(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W2 = tf.get_variable(name="W2", shape=[3, 3, 32, 64], dtype=tf.float32, initializer=initializer)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = activation_fn(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W3 = tf.get_variable(name="W3", shape=[3, 3, 64, 64], dtype=tf.float32, initializer=initializer)
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = activation_fn(L3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        W4 = tf.get_variable(name="W4", shape=[3, 3, 64, 32], dtype=tf.float32, initializer=initializer)
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = activation_fn(L4)
        L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        w = int(w / 2 + 0.5)
        h = int(h / 2 + 0.5)

        L4_flat = tf.reshape(L4, [-1, w * h * 32])

        W5 = tf.get_variable("W5", shape=[w * h * 32, n_class], initializer=initializer)
        b = tf.Variable(tf.random_normal([n_class]))
        logits = tf.matmul(L4_flat, W5) + b
        self._hypothesis = tf.nn.softmax(logits)
        self._prediction = tf.argmax(input=logits, axis=-1)

        self._xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
        self._loss = tf.reduce_mean(self._xentropy, name="loss")


# # augmentation

# In[11]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np

def augmentation(images_array,rotate=30,shift=30,shear=30,gray=True):
	
	# count : image 생성 갯수
	count_set = [rotate, shift, shear]

	# 함수 정의 
	gen_rotation = ImageDataGenerator(rotation_range=180)
	gen_shift = ImageDataGenerator(featurewise_center=True, width_shift_range=0.25, height_shift_range=0.25)
	gen_shear = ImageDataGenerator(shear_range=0.5) # 0.8이면 더 길어짐

	# create infinite flow of images
	images_flow_set = []
	images_flow_set.append(gen_rotation.flow(images_array, batch_size=1)) 
	images_flow_set.append(gen_shift.flow(images_array, batch_size=1))
	images_flow_set.append(gen_shear.flow(images_array, batch_size=1)) 

	aug_result = np.empty((0,) + images_array.shape[1:])

	for j,images_flow in enumerate(images_flow_set):
		count = counts_set[j] 
		for i, new_images in enumerate(images_flow):
	    # we access only first image because of batch_size=1
			print(new_images.shape)
			#new_image = array_to_img(new_images[0], scale=True)
			aug_result = np.append(aug_result, new_image)
			#new_image.save(output_path.format(30*(j)+i + 1))
			if i >= count:
				break

	return aug_result

#augmentation('sohee.jpg',gray=False)


# # CNN test

# In[12]:


import tensorflow as tf
import numpy as np
import pickle
import solver
import model
import dataSet as ds

epochs = 2000
batch_size = 128
learning_rate = 1e-4
#data = pickle.load(open('bnb.p', 'rb'))

d = ds.dataSet()
#d.temp_load_data()
d.load_data(dir='image', test_dir='test_image')
d.grayscale()

sess = tf.Session()
CNN_model = model.two_layer_CNN(sess = sess, input_shape=d.train_image.shape, n_class = d.n_class)
adam_opt = solver.Solver(sess = sess, name ='Adam', dataset=d, model = CNN_model, optimizer = tf.train.AdamOptimizer)

sess.run(tf.global_variables_initializer())
adam_opt.train(epoch=epochs, batch_size=batch_size, lr=learning_rate)
adam_opt.print_result()


# # slidingwindow

# In[13]:


import numpy as np
from PIL import Image
import utils

def plus1(np_array, region):
	aw, ah = np_array.shape
	x, y, w, h = region
	X = np.array([[i for i in range(ah)] for _ in range(aw)])
	Y = np.array([[i for _ in range(ah)] for i in range(aw)])
	X = (X >= x) & (X < x+w)
	Y = (Y >= y) & (Y < y+h)
	XY = X & Y
	return(np_array + XY)

def sliding_window(img, classifier, window_size, stride, boundary):
	np_img = np.array(img)
	img_w, img_h, _ = np_img.shape
	score_board = np.zeros(np_img.shape[0:2])
	count_board = np.zeros(np_img.shape[0:2])
	for window in window_size:
		w, h = window
		for x in range(0, img_h-h, stride):
			for y in range(0, img_w-w, stride):
				count_board = plus1(count_board, (x,y,w,h))
				croped_img = utils.seperate_region(img, (x, y, w, h))
				croped_img = croped_img.resize((64,64))
				croped_img = np.array(croped_img).reshape((1,64,64,3))
				if classifier(croped_img):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	c_img = b_img * img
	result_image = Image.fromarray(c_img)
	return result_image, b_img.reshape((img_w, img_h))

def edge_sliding_window(img, classifier, window_size, stride, boundary):
	def image_preprocessing(img):
	    from skimage import feature
	    img = np.sum(img, axis=2, keepdims=False)
	    return feature.canny(img)

	np_img = np.array(img)
	img_w, img_h, _ = np_img.shape
	score_board = np.zeros(np_img.shape[0:2])
	count_board = np.zeros(np_img.shape[0:2])
	for window in window_size:
		w, h = window
		for x in range(0, img_h-h, stride):
			for y in range(0, img_w-w, stride):
				count_board = plus1(count_board, (x,y,w,h))
				croped_img = utils.seperate_region(img, (x, y, w, h))
				croped_img = croped_img.resize((64,64))
				croped_img = np.array(croped_img).reshape((64,64,3))
				croped_img = image_preprocessing(croped_img)
				croped_img = np.array(croped_img).reshape((1,64,64,1))
				if classifier(croped_img):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	c_img = b_img * img
	result_image = Image.fromarray(c_img)
	return result_image, b_img.reshape((img_w, img_h))

def score_sliding_window(img, classifier, window_size, stride, boundary, target_index, limit):
	np_img = np.array(img)
	img_w, img_h, _ = np_img.shape
	score_board = np.zeros(np_img.shape[0:2])
	count_board = np.zeros(np_img.shape[0:2])
	for window in window_size:
		w, h = window
		for x in range(0, img_h-h, stride):
			for y in range(0, img_w-w, stride):
				count_board = plus1(count_board, (x,y,w,h))
				croped_img = utils.seperate_region(img, (x, y, w, h))
				croped_img = croped_img.resize((64,64))
				croped_img = np.array(croped_img).reshape((1,64,64,3))
				scores = classifier(croped_img)
				if argmax(scores) in target_index and max(scores) >= limit:
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	c_img = b_img * img
	result_image = Image.fromarray(c_img)
	return result_image, b_img.reshape((img_w, img_h))

def box_search_dfs(matrix,sy,sx):
	a = matrix[sy][sx]
	i = 0
	switch = True
	while switch:
		i = i+1
		for j in range(i):
			if any([(matrix.shape[0] <= sy+i), (matrix.shape[1] <= sx+i)]):
				switch = False
				break
			if (matrix[sy+i][sx+j] != a) or (matrix[sy+j][sx+i] != a):
				switch = False
				break
	return i, i

def make_box(matrix, limit_size = (64,64)):
	box_set = []
	h, w = matrix.shape
	for y in range(h):
		for x in range(w):
			if matrix[y][x]:
				i0, i1 = box_search_dfs(matrix, y, x)
				if i0 >= limit_size[0] and i1 >= limit_size[1]:
					box_set.append([x, y, i1, i0])
					for iy in range(i0):
						for ix in range(i1):
							matrix[y+iy][x+ix] = 0
	return box_set

def hog_sliding_window(img, classifier, window_size, stride, boundary):
	np_img = np.array(img)
	img_w, img_h, _ = np_img.shape
	score_board = np.zeros(np_img.shape[0:2])
	count_board = np.zeros(np_img.shape[0:2])
	for window in window_size:
		w, h = window
		for x in range(0, img_h-h, stride):
			for y in range(0, img_w-w, stride):
				count_board = plus1(count_board, (x,y,w,h))
				croped_img = utils.seperate_region(img, (x, y, w, h))
				croped_img = croped_img.resize((64,64))
				hog = utils.hog(np.array(croped_img))
				hog = hog.reshape((-1,8,4,4))
				hog = np.transpose(hog, (0,2,3,1))
				if classifier(hog):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	b_img = b_img * img
	result_image = Image.fromarray(b_img)
	return result_image


def gray_sliding_window(img, classifier, window_size, stride, boundary):
	np_img = np.array(img)
	img_w, img_h, _ = np_img.shape
	score_board = np.zeros(np_img.shape[0:2])
	count_board = np.zeros(np_img.shape[0:2])
	for window in window_size:
		w, h = window
		for x in range(0, img_h-h, stride):
			for y in range(0, img_w-w, stride):
				count_board = plus1(count_board, (x,y,w,h))
				croped_img = utils.seperate_region(img, (x, y, w, h))
				croped_img = croped_img.resize((64,64))
				croped_img = np.array(croped_img).reshape((1,64,64,3))
				croped_img = utils.grayscale(croped_img)
#				print(classifier(croped_img))
				if classifier(croped_img):
					score_board = plus1(score_board, (x,y,w,h))

	b_img = (score_board / count_board) > boundary
	b_img = b_img.reshape((img_w, img_h, 1))
	b_img = b_img * img
	result_image = Image.fromarray(b_img)
	return result_image


# # solver

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import os

class Solver:
    '''
    data를 이 안에 넣을까?
    '''
    def __init__(self, sess, name, model, dataset, optimizer=tf.train.AdamOptimizer):
        self._sess = sess
        self._model = model
        self._lr = tf.placeholder(dtype=tf.float32)
        self._loss_history = []
        self._train_acc_history = []
        self._test_acc_history = []

        with tf.variable_scope(name):
            self._optimizer = optimizer(self._lr)
            self._training_op = self._optimizer.minimize(self._model._loss)

        self.dataset = dataset
        self.batch_x, self.batch_y = [], []
        self.val_x, self.val_y = self.dataset.test_image, self.dataset.test_label
        

    def train(self, epoch=200, batch_size=128, lr = 1e-2, verbose=True, print_frequency=10):
        self.batch_size = batch_size
        n_batch = self.dataset.train_size // self.batch_size
        for i, iter in enumerate(range(epoch)):
            for j in range(n_batch):
                self.batch_x, self.batch_y = self.dataset.next_batch()
                feed_train = {self._model._x: self.batch_x, self._model._y: self.batch_y, self._lr: lr}
                _, recent_loss = self._sess.run(fetches=[self._training_op, self._model._loss], feed_dict=feed_train)
                self._loss_history.append(recent_loss)
                self._train_acc_history.append(self.accuracy(self.batch_x, self.batch_y))
                self._test_acc_history.append(self.accuracy(self.val_x, self.val_y))
            if verbose:
                if i % print_frequency == print_frequency-1:
                    self._print_train_process(epoch=i+1)


    def loss(self):
        feed_loss = {self._model._x: self.batch_x, self._model._y: self.batch_y}
        return self._sess.run(fetches=self._model._loss, feed_dict=feed_loss)

    def predict(self, x_data):
        feed_predict = {self._model._x: x_data}
        return self._sess.run(fetches=self._model._prediction, feed_dict=feed_predict)

    def predict_softmax_score(self, x_data):
        feed_predict = {self._model._x: x_data}
        return self._sess.run(fetches=self._model._hypothesis, feed_dict=feed_predict)

    def print_accuracy(self, x_data, y_data):
        result = y_data == self.predict(x_data=x_data)
        print('accuracy : {:.4f}'.format(sum(result) / len(result)))

    def _print_train_process(self, epoch):
        print('epoch : {:>4}, loss : {:.4f}, train_accuracy : {:.4f}, test_accuracy : {:.4f}'.format(
            epoch, self.loss(), self.accuracy(self.batch_x, self.batch_y), self.accuracy(self.val_x, self.val_y)))

    def accuracy(self, x_data, y_data):
        if x_data is None:
            return 0
        result = y_data == self.predict(x_data)
        return sum(result) / len(result)

    def print_result(self):
        plt.plot(self._loss_history)
        plt.title('loss')
        plt.show()

        l = range(len(self._train_acc_history))
        plt.plot(l, self._train_acc_history, 'b', label = 'train_acc')
        plt.plot(l, self._test_acc_history, 'r', label = 'test_acc')
        plt.legend()
        plt.title('accuracy')
        plt.show()


    def model_save(self, save_dir="saved"):
        saver = tf.train.Saver()
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        saver.save(self._sess, save_dir+"/train")

    def model_load(self, load_dir="saved"):
        saver = tf.train.Saver()
        saver.restore(self._sess, tf.train.latest_checkpoint(load_dir))


# # utils

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
import numpy as np
from xml.etree.ElementTree import Element, SubElement, dump
import xml.dom.minidom

def draw_rectangle(img, region, label='', display=False):
    '''
    그릴 직사각형의 영역 하나를 받아, 그려주는 함수
    :param img: 직사각형을 그릴 원본 이미지, PIL의 Image 객체, 깊은 복사를 통해 원본을 수정하지 않음
    :param region: 이미지 내에 그릴 직사각형의 영역 리스트, 리스트의 각 원소는 x, y, w, h로 구성
    :param label: 직사각형 내에 쓰여질 text
    :param display: 결과를 보여줄지 여부
    :return: 직사각형이 그려진 객체
    '''

    temp_img = img.copy()
    draw = ImageDraw.Draw(temp_img)
    x, y, w, h = region
    draw.rectangle((x, y, x+w, y+h), outline='red')
    draw.text((x,y), label, fill='red')

    if display:
        temp_img.show()

    return temp_img

def seperate_region(img, region, display=False):
    '''
    이미지에서 regions 값대로 crop한 결과를 리턴
    :param img: PIL의 image객체
    :param region: img를 crop할 기준
    :param display: crop된 결과를 보여줄지 여부
    :return: crop된 이미지
    '''
    x, y, w, h = region
    temp_image = img.copy()
    temp_image = temp_image.crop((x, y, x+w, y+h))

    if display:
        temp_image.show()

    return temp_image

def refining_ss_regions(ss_regions):
    '''
    selective search의 결과 중 region을 유의미한 img의 영역 부분만 남김
    selective search 라이브러리와 우리 코드와의 호환을 위함
    조건은 코드의 # -1, -2, -3 참고
    :param ss_regions: selective_search의 결과 regions
    :return: regions 중, 유의미한 결과의 numpy array
    '''
    candidates = set()
    for r in ss_regions:
        if r['rect'] in candidates: # -1
            continue
        if r['size'] < 2000 or r['size'] > 10000: # -2
            continue
        x, y, w, h = r['rect']
        if w / h > 2.5 or h / w > 2.5 : # -3
            continue
        if min(w, h) < 20:
            continue
        candidates.add(r['rect'])

    return np.array(list(candidates))

def CNN_classifier(img, softmax_classifier, input_size, label_dic, boundary):
    softmax_score = softmax_classifier(img)[0]
    ind = np.argmax(softmax_score)
    if softmax_score[ind] > boundary:
        return list(label_dic.keys())[ind]
    return None

def hog(img, pixels_per_cell = (16, 16), save = False, save_dir = 'temp.jpg'):
    '''
    img: Image 객체를 np로 변형한 객체
    return
        fd: img의 hog value np객체(1차원)
        hog_image_rescaled: img의 hog value를 이미지화 시킨 np객체
    '''
    from skimage.feature import hog
    from skimage import color, exposure

    image = color.rgb2gray(img)
    
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1), visualise=True)
    #print(fd.size)
    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    return fd

def hog_to_3d(hog, size):
    hog = hog.reshape(size)
    hog = np.transpose(hog, (2,3,1))
    return hog
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    '''

def grayscale(img_set):
    RGB_to_L = np.array([[[[0.299,0.587,0.114]]]])
    img_set = np.sum(img_set * RGB_to_L, axis=3, keepdims=True)
    return img_set

def box_to_xml(filename, path, size, box_set):
    note = Element("annotation")
    SubElement(note, "folder").text = 'Images'
    SubElement(note, "filename").text = filename
    SubElement(note, "path").text = path
    source = Element("source")
    SubElement(source, "database").text = 'Unknown'
    note.append(source)
    size_ele = Element('size')
    SubElement(size_ele, 'width').text = str(size[0])
    SubElement(size_ele, 'height').text = str(size[1])
    SubElement(size_ele, 'depth').text = str(size[2])
    note.append(size_ele)
    SubElement(note, 'segmented').text = '0'
    for box in box_set:
        obj = Element('object')
        SubElement(obj, 'name').text = 'block'
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'
        bndbox = Element('bndbox')
        SubElement(bndbox, 'xmin').text = str(box[0])
        SubElement(bndbox, 'ymin').text = str(box[1])
        SubElement(bndbox, 'xmax').text = str(box[0]+box[2])
        SubElement(bndbox, 'ymax').text = str(box[1]+box[3])
        obj.append(bndbox)
        note.append(obj)

    indent(note)
    return note

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    
    return elem

def list_to_str(ls):
    result = ''
    for line in ls:
        for ob in line:
            result = result + str(ob) + ' '
        result = result[:-1] + '\n'
    return result[:-1]

def box_to_txt(labels, box_set, image_size):
    result = []
    dw, dh = image_size
    for i, box in enumerate(box_set):
        x = (box[0]+box[2]/2)/dw
        y = (box[1]+box[3]/2)/dh
        w = box[2]/dw
        h = box[3]/dh
        result.append([labels[i], x, y, w, h])
    return list_to_str(result)


# In[7]:





# In[ ]:




