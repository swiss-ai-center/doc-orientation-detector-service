import numpy as np
import tensorflow as tf
import os

#WEIGHTS = '../models/flip-ml_DSflat_data_origsize_v1.0_IS512_NE20_IE20000_BS64_DR10_EN18_AC0.96_weights.keras'
WEIGHTS = "weights.keras"
N_CLASSES = 2
CLASSES_NAMES = ['0', '180']
IMG_SIZE = 512
NUM_EPOCHS = 20
IMG_PER_EPOCH = 20000
BATCH_SIZE = 64
DROP_RATE_PERCENT = 10

def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(DROP_RATE_PERCENT / 100.0),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROP_RATE_PERCENT / 100.0),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.load_weights(WEIGHTS)
    return model

def predict(model, image_data):
    predicted_scores = model.predict(image_data, verbose=1)
    predicted_labels = np.argmax(predicted_scores, axis=1)
    return int(CLASSES_NAMES[predicted_labels[0]])