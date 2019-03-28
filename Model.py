from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import numpy as np
import sys
import os

if len(sys.argv) < 3:
    print("You should specify directory to train on as first argument and "
        "directory to validate on as second one")
    exit()

train_dir = sys.argv[1]
validate_dir = sys.argv[2]

# ResNet model for classifications mushrooms into 3 categories:
# Edible, non-edible, not-a-mushroom
model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=2)

for layer in model.layers:
   layer.trainable = True

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validate_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800,
        verbose=2)

export_dir = 'export'
keras_file = 'keras_model_and_weights.h5'
tflite_file = os.path.join(export_dir, 'tflite_model_and_weigths.tflite')
model.save(keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open(tflite_file, "wb").write(tflite_model)

