import tensorflow as tf
import os
import itertools
from multiprocessing import Pool
import json
import time
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(0)
print('using tensorflow', tf.__version__)


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


def decode_img(file_name):
    img = tf.io.read_file(file_name)
    img = tf.image.decode_image(img, channels=3)
    return img


def load_label_pair(file_name, img_dir):
    root, fname = os.path.split(file_name)
    num = os.path.splitext(fname)[0]
    data_path = os.path.join(root, num + '.json')

    img = decode_img(os.path.join(img_dir, num + '.jpg'))
    png_height = 2560
    png_width = 1440
    # scaling = 0.75
    # png and jpg are diff dimensions... 2560 height vs 1920
    with open(data_path, 'r') as f:
        labeled_data = json.load(f)
        # TODO check multiple children
        for component in labeled_data['children']:
            height, width, channels = img.shape
            scaling = height / png_height
            x_min, y_min, x_max, y_max = [int(v * scaling) for v in component['bounds']]

            img_crop = img[y_min:y_max, x_min:x_max]
            ret_img = tf.image.resize_with_pad(img_crop, png_height, png_width)

            label = component['componentLabel']
            return ret_img, label
    print(data_path)

def parse_into_data_sets(annot_dir, img_dir, num_files, num_threads):
    b = time.time()
    images = tf.io.gfile.glob(os.path.join(annot_dir, '*.png'))[:num_files]
    print('Number of images', len(images))
    print("got glob", time.time() - b)

    b = time.time()
    tf.random.shuffle(images)
    # p = Pool(processes=num_threads)
    # ret = p.map(load_label_pair, images)
    ret = [load_label_pair(image, img_dir) for image in images]
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
    eval_pts = data_points[int(0.8*len(data_points)):]
    print("split dataset", time.time() - b)

    b = time.time()
    train_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in train_pts), (tf.uint8, tf.uint8))
    test_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in test_pts), (tf.uint8, tf.uint8))
    eval_ds = tf.data.Dataset.from_generator(lambda: (pair for pair in eval_pts), (tf.uint8, tf.uint8))
    # sample one point to make sure the ds is loaded properly
    for image, label in train_ds.take(5):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())
        # plt.figure()
        # plt.imshow(image)
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

    print("from dataset generator", time.time() - b)

    return train_ds, eval_ds, test_ds

