from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import sys
import os

# Arguments:
# 1 - programm name
# 2 - directory with train fotos
# 3 - directory with test fotos
if len(sys.argv) < 3:
    print("You should specify directory to train on as first argument and "
        "directory to validate on as second one")
    exit()

train_dir = sys.argv[1]
validate_dir = sys.argv[2]

def create_trainable_model(classes_count):
    model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=classes_count)
    for layer in model.layers:
        layer.trainable = True
    return model

def compile_model(model):
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')
    return

def train_model(model, train_dir, validation_dir, epochs):
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
    
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=800,
            verbose=2)
    return
    
# ResNet model for classifications mushrooms into 4 categories:
# edible, non-edible, partial-edible, not-a-mushroom
model = create_trainable_model(4)
compile_model(model)
train_model(model, train_dir, validate_dir, 50)
