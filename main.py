import os
import cv2
import os
import numpy as np
import cv2
from tqdm import tqdm
import shutil
from collections import Counter
import argparse
from keras.models import load_model

MODEL_PATH = "model"
DENSITY_LABEL = ["A", "B", "C", "D"]
DENSITY_MODEL = load_model(os.path.join(MODEL_PATH, "density_model.h5"))


def find_largest_contour(contours):
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour


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


def detect_orient(image):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h
    return aspect_ratio


def detect_density(image):
    image = cv2.resize(image, (256, 256))
    density = DENSITY_MODEL.predict(np.expand_dims(image, axis=0))
    density = density.argmax()
    return DENSITY_LABEL[density]


def combine_density(density_values):
    counter = Counter(density_values)
    most_common_items = counter.most_common()
    highest_frequency = most_common_items[0][1]
    if highest_frequency == 1:
        return -1
    highest_frequency_items = sorted(
        [item for item, count in most_common_items if count == highest_frequency])
    return highest_frequency_items[-1]


parser = argparse.ArgumentParser()
parser.add_argument('--data', metavar='d', type=str,
                    default="Data",
                    help='data folder')
parser.add_argument('--output', metavar='o', type=str,
                    default="output",
                    help='output folder')

args = parser.parse_args()
output_folder = args.output
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
data_folder = args.data
patients = sorted(os.listdir(data_folder))
for patient in tqdm(patients):
    patient_path = os.path.join(data_folder, patient)
    date = sorted(os.listdir(patient_path))[0]
    date_path = os.path.join(patient_path, date)
    images = sorted(os.listdir(date_path))
    images01 = [i for i in images if "IMG-0001" in i]
    images02 = [i for i in images if "IMG-0002" in i]
    if len(images02) != 0:
        images = images02
    else:
        images = images01

    density_values = []
    orient_values = []
    names = []

    for image in images:
        image_path = os.path.join(patient_path, image)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        density = detect_density(image)
        density_values.append(density)

        orient_values.append(detect_orient(grayscale))

        name = detect_lateral(grayscale)
        names.append(name)

    if len(density_values) != 0:
        density = combine_density(density_values)
        if density == -1:
            continue
        folder_path = os.path.join("output", patient + ' - ' + density)
        os.mkdir(folder_path)
        top2 = sorted(orient_values)[-2]
        for index, image in enumerate(images):
            image_path = os.path.join(patient_path, image)
            if orient_values[index] < top2:
                orient = "CC"
            else:
                orient = "MLO"
            name = names[index] + ' - ' + orient + '.jpg'
            shutil.copy(image_path, os.path.join(folder_path, name))
