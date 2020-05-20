import tensorflow as tf
import os
import itertools
from multiprocessing import Pool
import json
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE
# tf.enable_eager_execution()
tf.random.set_seed(0)


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


def process_path(file_name):
	img = tf.io.read_file(file_name)
	img = tf.image.decode_image(img, channels=3)
	return img


def load_label_pair(file_name):
	img = process_path(file_name)
	root, _ = os.path.split(file_name)
        # generalize: subtract extension from end of path
	num = file_name.split('/')[-1].split('.png')[0]
	data_path = os.path.join(root, num + '.json')
	with open(data_path, 'r') as f:
		labeled_data = json.load(f)
		# print('fname', file_name)
                # TODO check multiple children
		for component in labeled_data['children']:
			x_min, y_min, x_max, y_max = component['bounds']
			# print('crop dims', [y_min,y_max], [x_min,x_max])
			# print('img dims', img.shape)
			try:
				img_crop = img[y_min:y_max, x_min:x_max]
			except Exception as e:
				print(e)
				traceback(print_exc())
				print()
				raise e
			# print('cropped', img_crop)
                        # currently only fetching label
			label = component['componentLabel']
			# print('label', label)
			return img_crop, label
	print(data_path)

def parse_into_data_sets(local_dir, num_files, num_threads):
	b = time.time()
	images = tf.io.gfile.glob(os.path.join(local_dir, '*.png'))[:num_files]
	print('Number of images', len(images))
	print("got glob", time.time() - b)

	b = time.time()
	tf.random.shuffle(images)
	# p = Pool(processes=num_threads)
	# ret = p.map(load_label_pair, images)
	ret = [load_label_pair(image) for image in images]
	print("finish load", time.time() - b)

	b = time.time()
	valid_pairs = [x for x in ret if x is not None]
	labels = set([x[1] for x in valid_pairs])
	label_mapping = {}
        # TODO make fixed mapping
	for idx, key in enumerate(labels):
		label_mapping[key] = [idx]
	data_points = [(x[0], label_mapping[x[1]]) for x in valid_pairs]
	print("got labels", time.time() - b)

	b = time.time()
	# split 60/20/20
	train_pts = data_points[0:int(0.6*(len(data_points)))]
	test_pts = data_points[int(0.6 * len(data_points)):int(0.8*len(data_points))]
	validation_pts = data_points[int(0.8*len(data_points)):]
	print("split dataset", time.time() - b)

	b = time.time()
	train_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in train_pts), (tf.uint8, tf.uint8))
	test_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in test_pts), (tf.uint8, tf.uint8))
	validation_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in test_pts), (tf.uint8, tf.uint8))
	# sample one point to make sure the ds is loaded properly
	for image, label in train_ds.take(1):
		print("Image shape: ", image.numpy().shape)
		print("Label: ", label.numpy())
	print("from dataset generator", time.time() - b)

	return train_ds, test_ds, validation_ds

