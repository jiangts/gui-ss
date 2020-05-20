from utils import *
import argparse
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', type=int, default=50)
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--annotations_path', default='/Users/jiangts/Documents/stanford/cs231n/final_project/semantic_annotations')
parser.add_argument('--images_path', default='/Users/jiangts/Documents/stanford/cs231n/final_project/combined')
args = parser.parse_args()


train_ds, val_ds, test_ds = parse_into_data_sets(args.annotations_path,
        args.images_path,
        args.n_samples,
        args.num_threads)


## see example: https://keras.io/examples/vision/image_classification_from_scratch/

## Build and compile model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(int(2560/5), int(1440/5), 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## Train model
my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
        # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]


train_dataset = train_ds.shuffle(buffer_size=1024).batch(64)
val_dataset = train_ds.shuffle(buffer_size=1024).batch(64)
test_dataset = test_ds.batch(64)

# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.
model.fit(train_dataset, epochs=10, callbacks=my_callbacks, validation_data=val_dataset)



## Test model
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)

print('\nTest accuracy:', test_acc)

