import tensorflow as tf
import os
import itertools
# from multiprocessing import Pool
import json
import time
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('using tensorflow', tf.__version__)
tf.random.set_seed(0)

label_mapping = {
        "Advertisement": 0,
        "Background Image": 1,
        "Bottom Navigation": 2,
        "Button Bar": 3,
        "Card": 4,
        "Checkbox": 5,
        "Date Picker": 6,
        "Drawer": 7,
        "Icon": 8,
        "Image": 9,
        "Input": 10,
        "List Item": 11,
        "Map View": 12,
        "Modal": 13,
        "Multi-Tab": 14,
        "Number Stepper": 15,
        "On/Off Switch": 16,
        "Pager Indicator": 17,
        "Radio Button": 18,
        "Slider": 19,
        "Text": 20,
        "Text Button": 21,
        "Text Buttonn": 22,
        "Toolbar": 23,
        "Video": 24,
        "Web View": 25
        }


def get_pairs(file_path, img):
    f = tf.io.read_file(img_path)
    annots = json.load(f)

    png_height = 2560
    png_width = 1440

    pairs = []

    # TODO recursively grab nested children
    for component in labeled_data['children']:
        height, width, channels = img.shape
        scaling = height / png_height
        x_min, y_min, x_max, y_max = [int(v * scaling) for v in component['bounds']]

        img_crop = img[y_min:y_max, x_min:x_max]
        try:
            ret_img = tf.image.resize_with_pad(img_crop, int(png_height/5), int(png_width/5))
            label = component['componentLabel']
            pairs.append(ret_img, label)
        except Exception as e:
            print(e)
    return pairs


def get_img(img):
    if tf.io.gfile.exists(img_path):
        img = tf.io.read_file(img_path)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
        # # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        # img = tf.image.convert_image_dtype(img, tf.float32)
        # # resize the image to the desired size.
        # return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
        return img


def process_path(file_path, img_dir):
    root, fname = os.path.split(file_path)
    num = os.path.splitext(fname)[0]
    img_path = os.path.join(img_dir, num + '.jpg')

    # parts = tf.strings.split(file_path, os.sep)
    # fileparts = tf.strings.split(parts[-1], '.')
    # img_path = tf.strings.join([img_dir, fileparts[0], fileparts[1]])

    img = get_img(img_path)
    return get_pairs(file_path, img)


AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_data(annot_dir, img_dir, num_files, num_threads=AUTOTUNE):
    disable_eager_execution()

    list_ds = tf.data.Dataset.list_files(os.path.join(annot_dir, '*json'), seed=0)
    labeled_ds = list_ds.map(lambda file_path: tf.py_function(func=process_path,
        inp=[file_path, img_dir], Tout=[tf.float32,tf.int8]), num_parallel_calls=num_threads)
    # flattens "pairs"
    dataset = labeled_ds.unbatch()

    # train_ds = dataset.drop(num)
    # test_ds = dataset.drop(10000).take(10000)
    # val_ds = dataset.take(10000)

    for image, label in dataset.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    return train_ds, val_ds, test_ds, label_mapping

