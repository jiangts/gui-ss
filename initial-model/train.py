from utils import *
import argparse
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--num_threads', type=int, default=50)
parser.add_argument('--n_samples', type=int, default=1000)
parser.add_argument('--annotations_path', default='/Users/jiangts/Documents/stanford/cs231n/final_project/semantic_annotations')
parser.add_argument('--images_path', default='/Users/jiangts/Documents/stanford/cs231n/final_project/combined')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()


train_ds, val_ds, test_ds, classes = parse_into_data_sets(args.annotations_path,
        args.images_path,
        args.n_samples,
        args.num_threads)


## see example: https://keras.io/examples/vision/image_classification_from_scratch/

## Build and compile model
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(int(2560/5), int(1440/5), 3)),
#     keras.layers.Dense(128, activation='relu'),
#     keras.layers.Dense(10)
#     ])
training_pair = list(train_ds.take(1).as_numpy_iterator())[0]
num_classes = len(classes)
print(training_pair[0].shape, num_classes)

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
    input_shape=training_pair[0].shape))
# model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

# model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# linear baseline
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=training_pair[0].shape),
#     keras.layers.Dense(num_classes)
#     ])

top3_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3acc')
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['categorical_accuracy', top3_acc])

## Train model
my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
        tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]


train_dataset = train_ds.shuffle(buffer_size=1024).batch(args.batch_size)
val_dataset = train_ds.shuffle(buffer_size=1024).batch(args.batch_size)
test_dataset = test_ds.batch(args.batch_size)

# Since the dataset already takes care of batching,
# we don't pass a `batch_size` argument.
model.fit(train_dataset, epochs=args.epochs, callbacks=my_callbacks, validation_data=val_dataset)



## Test model
metrics = model.evaluate(test_dataset, verbose=2)

print('\nTest metrics:', metrics)

