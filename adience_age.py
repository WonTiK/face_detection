from PIL import Image
import numpy as np
import math

IMAGE_LOCATION = '/home/lukken/ADIENCE/aligned'
TXT_LOCATION = '/home/lukken/ADIENCE/Folds/train_val_txt_files_per_fold/test_fold_is_0/age_train.txt'

f = open(TXT_LOCATION, 'r')
index = 0

# make image & label list.
# output: image_list, label_list
def make_image_list():
	global index
	image_list = []
	label_list = []
	for j in range(1, 301):
	# while True:
		txt = f.readline()
		if not txt: break
		index += 1
		if index % 1000 is 0:
			print('index: %s' % str(index))

		filename, label = txt.split(" ")

		# make image list.
		# crop size: 227 * 227. mirror option needed soon.
		width = 227
		height = 227

		image = Image.open(IMAGE_LOCATION + '/' + filename)
		image_mod = image.resize((width, height))
		image_np = np.array(image_mod, dtype=np.uint8)
		image_list.append(image_np)

		# make label list.
		label = int(label)
		label_list.append(label)

	classes = 8
	# make one-hot vector.
	label_list = np.eye(classes)[label_list]

	return np.array(image_list), np.array(label_list)

# train model.
def train_data(image_list, label_list):
	X = tf.placeholder(tf.float32, [None, 228, 228, 3])
	Y_ = tf.placeholder(tf.float32, [None, 8])
	lr = tf.placeholder(tf.float32)

	K = 96
	L = 256

	W1 = tf.Variable(tf.truncated_normal([7, 7, 3, K]))
	B1 = tf.Variable(tf.ones([K]) / 8)
	W2 = tf.Variable(tf.truncated_normal([5, 5, 96, L]))
	B1 = tf.Variable(tf.ones([L]) / 8)

	stride = 4
	relu1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
	pool1 = tf.nn.max_pool()


image_list, label_list = make_image_list()
print('Image amount: %d' % len(image_list))
print(np.shape(image_list))
print('Label amount: %d' % len(label_list))
print(np.shape(label_list))
print((label_list[0]))
