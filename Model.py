#Retrain MobileNetV2 neural network model based on imagenet wheights

#python3 Model.py 4 50 "../mushroom-images/Mushrooms_with_classes/" "../mushroom-images/Mushrooms_with_classes/"

#from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.optimizers import SGD
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras import metrics
from keras.callbacks import CSVLogger
from matplotlib import pyplot
import tensorflow as tf
import numpy as np
import sys
import os

# Arguments:
# 0 - programm name
# 1 - number of classes
# 2 - number of epohs
# 3 - directory with train fotos
# 4 - directory with test fotos
if len(sys.argv) < 5:
    print("You should specify number of output classes as first argument, number of training epochs as second argument, "
    "directory to train on as third argument and directory to validate on as forth argument.")
    exit()

number_of_classes = int(sys.argv[1])
number_of_epochs = int(sys.argv[2])
train_dir = sys.argv[3]
validate_dir = sys.argv[4]

def create_trainable_resnet50(classes_count):
	model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    #do not retrain hole model
	for layer in model.layers:
		layer.trainable = False

    #create new output layer and train it
	x = Flatten()(model.output)
	predictions = Dense(classes_count, activation='softmax', name='fc1000')(x)
	
	return Model(input=model.input, output=predictions)
	
def create_trainable_MobileNetV2(classes_count):
	model = MobileNetV2(alpha=1.0, include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    #do not retrain hole model
	for layer in model.layers:
		layer.trainable = False

    #create new output layer and train it
	x = Flatten()(model.output)
	predictions = Dense(classes_count, activation='softmax', name='fc1000')(x)
	
	return Model(input=model.input, output=predictions)

def compile_model(model):
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['mse', 'mae', 'mape', 'cosine'])
    return

def train_model(model, train_generator, validation_generator, epochs):
    csv_logger = CSVLogger('ResNet50_log.csv', append=True, separator=';')
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=800,
            verbose=2,
            callbacks=[csv_logger])
    return history

def get_data_generators(train_dir, validation_dir):
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
    
    return train_generator, validation_generator

def save_class_labels(label_map, dir_path, file_name):
    sorted_labels = sorted(label_map, key=label_map.__getitem__)
    with open(os.path.join(dir_path, file_name), 'w') as label_file:
        for label in sorted_labels:
            label_file.write("%s\n" % label)
    return

def save_model_to_pb(model, dir_path, file_name, output_node_name):
    K.set_learning_phase(0)
    tf.identity(model.output, name=output_node_name)
    session = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [output_node_name])
    graph_io.write_graph(constant_graph, dir_path, file_name, as_text=False)
    return
    
def save_model_to_h5(model, dir_path, file_name):
    model.save(os.path.join(dir_path, file_name))
    return

#model = create_trainable_resnet50(number_of_classes)
model = create_trainable_MobileNetV2(number_of_classes)
compile_model(model)
train_generator, validation_generator = get_data_generators(train_dir, validate_dir)
history = train_model(model, train_generator, validation_generator, number_of_epochs)
compile_model(model)

print(train_generator.class_indices)

save_model_to_h5(model, '.', 'model_with_weights.h5')
save_model_to_pb(model, 'export', 'model_with_weights.pb', 'output')
save_class_labels(train_generator.class_indices, 'export', 'labels.txt')
