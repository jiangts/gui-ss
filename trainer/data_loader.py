import tensorflow as tf
import matplotlib.pyplot as plt


print('using tensorflow', tf.__version__)
tf.random.set_seed(0)

AUTOTUNE = tf.data.experimental.AUTOTUNE

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



def get_img(img_path):
    if tf.io.gfile.exists(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
        return img

def process_img(img_path, bounds):
    # img = get_img(img_path)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    png_height = 2560
    png_width = 1440
    out_height = 512 # int(png_height/5)
    out_width = 288 # int(png_width/5)

    img = tf.image.resize_with_pad(img, png_height, png_width)

    xmin, ymin, xmax, ymax = bounds
    img = img[ymin:ymax, xmin:xmax]

    img = tf.image.resize_with_pad(img, out_height, out_width)
    return img


def process_path(img_path,xmin,ymin,xmax,ymax,label):
    # label = label_mapping[label]
    img = process_img(img_path, [xmin,ymin,xmax,ymax])
    return img, label


def show(image, label):
  plt.figure()
  plt.imshow(image)
  plt.title(label.numpy().decode('utf-8'))
  plt.axis('off')
  plt.show()



def load_data(registry_file, num_threads=AUTOTUNE):
    box_types  = [tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.string]
    registry = tf.data.experimental.CsvDataset(registry_file, box_types , header=False)
    dataset = registry.map(process_path, num_parallel_calls=num_threads)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())


    for img, label in dataset.shuffle(buffer_size=1024).take(50):
        show(img, label)


load_data('/Users/jiangts/Documents/stanford/cs231n/final_project/frcnn_annot.txt')

#    list_ds = tf.data.Dataset.list_files(os.path.join(annot_dir, '*json'), seed=0)
#    labeled_ds = list_ds.map(lambda file_path: tf.py_function(func=process_path,
#        inp=[file_path, img_dir], Tout=[tf.float32,tf.int8]), num_parallel_calls=num_threads)
#
#    train_ds = dataset.drop(num)
#    test_ds = dataset.drop(10000).take(10000)
#    val_ds = dataset.take(10000)
#
#    for image, label in dataset.take(1):
#        print("Image shape: ", image.numpy().shape)
#        print("Label: ", label.numpy())
#
#    return train_ds, val_ds, test_ds, label_mapping
#
