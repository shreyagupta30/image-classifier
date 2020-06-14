import argparse
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json
from PIL import Image



def main():
    parser = argparse.ArgumentParser(description = 'Image classifier')
    parser.add_argument('--image', type = str, required = True, help = 'Path to image')
    parser.add_argument('--model', type = str, help = 'Load a saved model')
    parser.add_argument('--topk', type = int, help = 'To display top k number of results')
    parser.add_argument('--category_names', type = str, help = 'Map categories to real names')
    args = parser.parse_args()
    
    image_path = args.image
    topk = args.topk
    names = None
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    if args.category_names is not None:
        class_names = args.category_names
        with open(names, 'r') as f:
            class_names = json.load(f)
            
    if args.model:
        model = load_model(args.model)
        execute(image_path, model, topk, class_names)
    else:
        create_model(image_path, topk, class_names)
        model = load_model('./my_model.h5')
        execute(image_path, model, topk, class_names)
        
            
    
    
def create_model(image_path, topk, class_names):
    print("Creating a model.... Please wait...")
    train_split = 60
    test_val_split = 20
    splits = tfds.Split.ALL.subsplit([60,20, 20])
    (training_set, validation_set, test_set), dataset_info = tfds.load('oxford_flowers102', split=splits, as_supervised=True, with_info=True)
    total_examples = dataset_info.splits['train'].num_examples + dataset_info.splits['test'].num_examples
    num_training_examples = (total_examples * train_split) // 100
    num_validation_examples = (total_examples * test_val_split) // 100
    num_test_examples = num_validation_examples
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    batch_size = 102
    image_size = 224
    def format_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        image /= 255
        return image, label
    
    training_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(batch_size).prefetch(1)
    validation_batches = validation_set.map(format_image).batch(batch_size).prefetch(1)
    testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)
    URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(URL, input_shape=(image_size, image_size,3))
    feature_extractor.trainable = False
    layer_neurons = [3000, 2048, 1024]
    model = tf.keras.Sequential()
    model.add(feature_extractor)
    for neurons in layer_neurons:
        model.add(tf.keras.layers.Dense(neurons, activation = 'relu'))
        model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(102, activation = 'softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    save_best = tf.keras.callbacks.ModelCheckpoint('./best_model.h5', monitor='val_loss', save_best_only=True)
    epochs = 30
    history = model.fit(training_batches, epochs = epochs, validation_data=validation_batches, callbacks=[early_stopping])    
    saved_keras_model_filepath = './my_model.h5'
    model.save(saved_keras_model_filepath)
    print("Model creation Completed")
    
def load_model(path):
    model = tf.keras.models.load_model(path, custom_objects={'KerasLayer':hub.KerasLayer})
    return model

def predict(image_path, model, topk, class_names):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    ps = model.predict(image)
    probs, indices0 = tf.nn.top_k(ps, k=topk)    
    indices0 = indices0.numpy()
    indices = []
    for i in indices0:
        indices.append(i+1)
    probs = probs.numpy()
    print("Probs:", probs)
    top_classes = [class_names[str(each)] for each in indices[0]]
    print("Indices:", indices[0])
    return probs, top_classes

def execute(image_path, model, topk, class_names):
    image = Image.open(image_path)
    image = np.asarray(image)
    probs, classes = predict(image_path, model, topk, class_names)

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.reshape(image, (224, 224, 3))
    image /= 255
    image = image.numpy()
    return image
    

if __name__ == '__main__': main()