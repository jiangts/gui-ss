import tensorflow as tf
import matplotlib.pyplot as plt
import os


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


def process_img(img_path, bounds):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)

    png_height = 2560
    png_width = 1440
    out_height = 512 # int(png_height/5)
    out_width = 288 # int(png_width/5)

    # img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # img = tf.image.resize(img, [png_height, png_width])

    img = tf.image.resize_with_pad(img, png_height, png_width)

    xmin, ymin, xmax, ymax = bounds
    img = img[ymin:ymax, xmin:xmax]

    img = tf.image.resize_with_pad(img, out_height, out_width)
    return img


def process_path(img_path,xmin,ymin,xmax,ymax,label):
    label = tf.strings.to_number(label, out_type=tf.dtypes.int32)
    img = process_img(img_path, [xmin,ymin,xmax,ymax])
    return img, label


def show(image, label):
    plt.figure()
    plt.imshow(image)
    if isinstance(label.numpy(), str):
        plt.title(label.numpy().decode('utf-8'))
    plt.axis('off')
    plt.show()



def make_dataset(registry_file, num_threads=AUTOTUNE):
    box_types  = [tf.string, tf.int32, tf.int32, tf.int32, tf.int32, tf.string]
    registry = tf.data.experimental.CsvDataset(registry_file, box_types , header=False)
    dataset = registry.map(process_path, num_parallel_calls=num_threads)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    return dataset



# gsplit -l 260830 -d --additional-suffix=.txt classify.txt classify
def load_data(registry_file, num_splits=5, num_threads=AUTOTUNE):
    filename, file_extension = os.path.splitext(registry_file)
    datasets = []
    for i in range(num_splits):
        registry_file = filename + str(i).zfill(2) + file_extension
        datasets.append(make_dataset(registry_file))
    s1, s2, s3, s4, s5 = datasets
    train_ds = s1.concatenate(s2).concatenate(s3)
    val_ds = s4
    test_ds = s5


    # img = tf.io.read_file('/Users/jiangts/Documents/stanford/cs231n/final_project/combined/28861.jpg')
    # img = tf.image.decode_image(img, channels=3, dtype=tf.dtypes.float32)
    # plt.figure()
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # for img, label in dataset.shuffle(buffer_size=1024).take(50):
    #    show(img, label)

    num_examples = 1304147
    num_eval = int(num_examples/5)
    num_train = num_examples - num_eval * 2

    for image, label in train_ds.take(1):
        # show(image, label)
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    input_dim = (512, 288, 3)

    return train_ds, val_ds, test_ds, label_mapping, input_dim, num_train, num_eval


# load_data('/Users/jiangts/Documents/stanford/cs231n/final_project/classify.txt')
# load_data('gs://ui-scene-seg_training/data/classify.txt')

