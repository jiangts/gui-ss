import tensorflow as tf
import glob
import random
import os
import itertools
from multiprocessing import Pool
import json

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.enable_eager_execution()


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.
    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.
    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.io.gfile.GFile(temp_file, 'r') as temp_file_object:
        with tf.io.gfile.GFile(filename, 'w') as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                file_object.write(line)
    tf.io.gfile.remove(temp_file)

def decode_img(img): 
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.convert_image_dtype(img, tf.float32)
	return img 

def process_path(file_name): 
	img = tf.io.read_file(file_name)
	img = decode_img(img)
	return img
	

def load_label_pair(file_name): 
	img = process_path(file_name)
	root, _ = os.path.split(file_name)
	num = file_name.split('/')[-1].split('.png')[0]
	data_path = os.path.join(root, num + '.json')
	with open(data_path, 'r') as f:
		labeled_data = json.load(f)
		for component in labeled_data['children']:
			x_min, x_max = component['bounds'][0], component['bounds'][0] + component['bounds'][2]
			y_min, y_max = component['bounds'][1], component['bounds'][1] + component['bounds'][3]
			img_crop = img[x_min:x_max, y_min:y_max]
			label = component['componentLabel']
			return img_crop, label
	print(data_path)

def parse_into_data_sets(local_dir, num_files, num_threads):
	images = glob.glob(os.path.join(local_dir, '*.png'))[:num_files]
	print('Number of images', len(images))
	random.shuffle(images) 
	p = Pool(processes=num_threads)
	ret = p.map(load_label_pair, images)
	valid_pairs = [x for x in ret if x is not None]
	labels = set([x[1] for x in valid_pairs])
	label_mapping = {}
	for idx, key in enumerate(labels): 
		label_mapping[key] = [idx] 
	data_points = [(x[0], label_mapping[x[1]]) for x in valid_pairs]
	
	# split 20/20/20 
	train_pts = data_points[0:int(0.6*(len(data_points)))]
	test_pts = data_points[int(0.6 * len(data_points)):int(0.8*len(data_points))]
	validation_pts = data_points[int(0.8*len(data_points)):]

	train_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in train_pts), (tf.float32, tf.int32))
	# sample one point to make sure the ds is loaded properly 
	for image, label in train_ds.take(1):
		print("Image shape: ", image.numpy().shape)
		print("Label: ", label.numpy())
	test_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in test_pts), (tf.float32, tf.int32))
	validation_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in test_pts), (tf.float32, tf.int32))
	
	return train_ds, test_ds, validation_ds	

	
