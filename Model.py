from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.optimizers import SGD
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import tensorflow as tf
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

def create_trainable_resnet50(classes_count):
    model = ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=classes_count)
    for layer in model.layers:
        layer.trainable = True
    return model

def compile_model(model):
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')
    return

def train_model(model, train_generator, validation_generator, epochs):
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=800,
            verbose=2)
    return

def save_model_to_pb(model, dir_path, file_name, output_node_name):
    K.set_learning_phase(0)
    tf.identity(model.output, name=output_node_name)
    session = K.get_session()
    constant_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), [output_node_name])
    graph_io.write_graph(constant_graph, dir_path, file_name, as_text=False)
    return

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

model = create_trainable_resnet50(4)
compile_model(model)
train_generator, validation_generator = get_data_generators(train_dir, validate_dir)
train_model(model, train_generator, validation_generator, 50)
compile_model(model)
save_model_to_pb(model, 'export', 'model_with_weights.pb', 'output')
save_class_labels(train_generator.class_indices, 'export', 'labels.txt')
