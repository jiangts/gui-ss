# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras


def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.

    Args:
      features: numpy array of features used for training or inference
      labels: numpy array of labels for each example
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      num_epochs: number of epochs to provide the data for
      batch_size: batch size for training

    Returns:
      A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset


def create_keras_model(input_shape, num_classes, learning_rate):
    """Creates Keras Model for Binary Classification.

    The single output node + Sigmoid activation makes this a Logistic
    Regression.

    Args:
      input_dim: How many features the input has
      learning_rate: Learning rate for training

    Returns:
      The compiled Keras model (still needs to be trained)
    """

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
        input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(1, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile Keras model
    top3_acc = keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3acc')
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model
