import tensorflow as tf
import os
import itertools
# from multiprocessing import Pool
import json
import time
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
    if tf.io.gfile.exists(file_name) is False:
        return None

    with tf.io.gfile.GFile(file_name, 'rb') as f:
        img = f.read()
        img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
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
    if img is None or tf.io.gfile.exists(data_path) is False:
        return None

    with tf.io.gfile.GFile(data_path, 'r') as f:
        labeled_data = json.load(f)
        # TODO check multiple children
        for component in labeled_data['children']:
            height, width, channels = img.shape
            scaling = height / png_height
            x_min, y_min, x_max, y_max = [int(v * scaling) for v in component['bounds']]

            img_crop = img[y_min:y_max, x_min:x_max]
            # only method="nearest" preserves the dtype of uint8...
            # https://www.tensorflow.org/api_docs/python/tf/image/resize
            # ret_img = tf.image.resize_with_pad(img_crop, int(png_height/5), int(png_width/5), method='nearest')

            # alternatively, decode images into float32 format
            try:
                ret_img = tf.image.resize_with_pad(img_crop, int(png_height/5), int(png_width/5))
                label = component['componentLabel']
                return ret_img, label
            except Exception as e:
                print(e)
    print(data_path)



# other data loading techniques: https://www.tensorflow.org/tutorials/load_data/images#load_using_tfdata
def load_data(annot_dir, img_dir, num_files, num_threads):
    b = time.time()
    images = tf.io.gfile.glob(os.path.join(annot_dir, '*.png'))[:num_files]
    print("got glob", time.time() - b)

    b = time.time()
    # p = Pool(processes=num_threads)
    # ret = p.map(load_label_pair, images)
    ret = [load_label_pair(image, img_dir) for image in images]
    pairs = [x for x in ret if x is not None]
    print("finish load", time.time() - b)

    # TODO make fixed mapping
    labels = set([x[1] for x in pairs])
    label_mapping = {}
    for idx, key in enumerate(labels):
        label_mapping[key] = idx

    # split 60/20/20
    x_data = [x[0] for x in pairs]
    y_data = [label_mapping[x[1]] for x in pairs]
    split1 = int(0.6*len(pairs))
    split2 = int(0.8*len(pairs))

    train_ds = tf.data.Dataset.from_tensor_slices((x_data[0:split1], y_data[0:split1]))
    val_ds = tf.data.Dataset.from_tensor_slices((x_data[split1:split2], y_data[split1:split2]))
    test_ds = tf.data.Dataset.from_tensor_slices((x_data[split2:], y_data[split2:]))

    # sample one point to make sure the ds is loaded properly
    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())
        # plt.figure()
        # plt.imshow(image.numpy())
        # plt.colorbar()
        # plt.grid(False)
        # plt.show()

    return train_ds, val_ds, test_ds, label_mapping

