import os
import pandas as pd 
import numpy as np 
import random
from tqdm import tqdm
import matplotlib.pyplot as plt 
import cv2
import datetime
import seaborn as sns


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

random.seed(100)
LABEL_INDEX = ["A", "B", "C", "D"]

## Preprocess

def detect_lateral(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    left_half = image[:, :width//2]
    right_half = image[:, width//2:]
    left_avg_intensity = np.mean(left_half)
    right_avg_intensity = np.mean(right_half)
    if left_avg_intensity >= right_avg_intensity:
        return "Left"
    else:
        return "Right"

def find_largest_contour(contours):
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

def find_edges(image, gray):
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    if len(contours)>1:
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        image = cv2.bitwise_and(image, image, mask=mask)
    return image

def crop_breast_area(edge_image, image):
    lateral = detect_lateral(image)
    if lateral=="Right":
        image = cv2.flip(image, 1)
        edge_image = cv2.flip(edge_image, 1)
    width, height, channel = np.where(edge_image!=0)
    max_width = np.max(width)
    max_height = np.max(height)
    image = image[:max_width,:max_height]
    if lateral=="Right":
        image = cv2.flip(image, 1)
    return image

def preprocess(X):
    X = (X - X.min()) / (X.max() - X.min())
    X = X * 255
    X = X[10:-10, 10:-10]
    X = X.astype(np.uint8)
    gray = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
    edge_image = find_edges(X, gray)
    X = crop_breast_area(edge_image, X)
    output = cv2.connectedComponentsWithStats((gray > 20).astype(np.uint8), 8, cv2.CV_32S)
    stats = output[2]
    try:
        idx = stats[1:, 4].argmax() + 1
    except:
        return X, edge_image
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h
    X_fit = X[y1: y2, x1: x2]
    if X_fit.shape[0]==0 or X_fit.shape[1]==0:
        X_fit = X
    return X_fit, edge_image

## Create dataset
df = pd.read_csv("validation.csv")
X = []
y = list(df.Density.values-1)

image_path = list(df["fullPath"].values)
count = 0
for img in tqdm(image_path):
    if "\\" in img:
        img = img.replace("\\","/")
    n_img = cv2.imread(img, cv2.IMREAD_COLOR)
    ori = n_img.copy()
    n_img, edge_image = preprocess(n_img)
    n_img_size = cv2.resize(n_img, (256, 256))
    X.append(n_img_size)


bvtn = pd.read_csv('ThongNhat_labels.csv')
bvtn.columns = ["path", "breast_density"]
bvtn["breast_density"] = bvtn["breast_density"].apply(lambda x: LABEL_INDEX.index(x))
names = list(set(bvtn.path.values))
patients = os.listdir("data")

new_X = []
for index, row in bvtn.iterrows():
    name, label = row
    flag = False
    for p in patients:
        if name in p:
            flag = True
            new_X.append(p)
            break

ori_y = list(bvtn.breast_density.values)
for patient, label in tqdm(zip(new_X, ori_y)):
    files = os.listdir("data/"+patient)
    for f in files:
        n_img = cv2.imread(os.path.join("data/"+patient, f), cv2.IMREAD_COLOR)
        n_img = preprocess(n_img)
        n_img_size = cv2.resize(n_img, (256, 256))
        X.append(n_img_size)
        y.append(label)
        

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.3, random_state=42)

y_train = to_categorical(y_train, 4)  
y_val = to_categorical(y_val, 4)  
y_test = to_categorical(y_test, 4)


print('X_train shape : {}' .format(X_train.shape))
print('X_val shape : {}' .format(X_val.shape))
print('X_test shape : {}' .format(X_test.shape))
print('y_train shape : {}' .format(y_train.shape))
print('y_val shape : {}' .format(y_val.shape))
print('y_test shape : {}' .format(y_test.shape))

## Training
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(strides=2),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((3, 3), strides=2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',          
    patience=5,                  
    min_delta=1e-7,              
    restore_best_weights=True,   
)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',   
    factor=0.2,           
    patience=2,           
    min_delta=1e-7,       
    cooldown=0,           
    verbose=1             
)

history = model.fit(X_train, 
                    y_train, 
                    batch_size=64,
                    epochs=50, 
                    validation_data=(X_val, y_val), 
                    callbacks=[
                    early_stopping,
                    plateau,
                ])
os.makedirs("model", exist_ok=True)
model.save(f"model/density_model.h5")

print("Acc", model.evaluate(X_val, y_val, verbose=0))
print("Acc", model.evaluate(X_test, y_test, verbose=0))
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
