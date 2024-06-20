import os
import cv2
import pandas as pd
import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import albumentations as A
import torch
from catboost import CatBoostRegressor
from tqdm import tqdm
import shutil
from collections import Counter
import argparse

MODEL_PATH = "models"
DENSITY_LABEL = ["A", "B", "C", "D"]
folds = 5
unet_models = []
catboost_models = []
for fold in range(folds):
    model_path = os.path.join(MODEL_PATH, "unet_resnet50_model_{fold}.pth")
    model = torch.load(model_path)
    model.eval()
    unet_models.append(model)
    reg = CatBoostRegressor().load_model(
        os.path.join(MODEL_PATH, "catboost_fold_{fold}.cbm"))
    catboost_models.append(reg)


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


def image_tensor(img):
    if type(img) not in [np.ndarray, Image.Image]:
        raise TypeError("Input must be np.ndarray or PIL.Image")
    torch_tensor = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    if type(img) == Image.Image:
        image = torch_tensor(img)
        image = image.unsqueeze(0)
        return image
    elif type(img) == np.ndarray:
        pil_image = Image.fromarray(img).convert("RGB")
        image = torch_tensor(pil_image)
        image = image.unsqueeze(0)
        return image
    else:
        raise TypeError("Input must be np.ndarray or PIL.Image")


def detect_density(image_path):
    image = Image.open(image_path).convert('RGB')
    img = image_tensor(image)
    final_pred1 = 0
    final_pred2 = 0

    for fold in range(5):
        with torch.no_grad():
            pred1, pred2 = unet_models[fold].module.predict(img.cuda())
        pred1 = pred1[0].cpu().numpy().transpose(1, 2, 0)
        pred1 = pred1[:, :, 0]
        pred2 = pred2[0].cpu().numpy().transpose(1, 2, 0)
        pred2 = pred2[:, :, 0]
        final_pred1 += pred1.sum()/folds
        final_pred2 += pred2.sum()/folds
    final_pred3 = final_pred2/final_pred1
    X_solution = [[final_pred1, final_pred2, final_pred3]]
    density = 0
    for fold in range(folds):
        density += catboost_models[fold].predict(X_solution)/5
    density = density[0]
    if density < 1:
        density = 1
    return DENSITY_LABEL[int((density-1)/25)], density


def find_largest_contour(contours):
    max_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour


def combine_density(density_values, values):
    counter = Counter(density_values)
    most_common_items = counter.most_common()
    highest_frequency = most_common_items[0][1]
    if highest_frequency == 1:
        return -1, -1
    highest_frequency_items = sorted(
        [item for item, count in most_common_items if count == highest_frequency])
    density = highest_frequency_items[-1]
    for d, v in zip(density_values, values):
        if d == density:
            break
    return highest_frequency_items[-1], v


def predict_one_image(img, model):
    resized_img = cv2.resize(img, (512, 512))
    X = np.reshape(
        resized_img, (1, resized_img.shape[0], resized_img.shape[1], 1))
    normalized_X = X/255
    normalized_X = np.rollaxis(normalized_X, 3, 1)
    pred_y = model.predict(normalized_X, verbose=0)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y != 1] = 0
    pred_img = np.reshape(pred_y[0]*255, (512, 512))
    match_img = pred_img*resized_img
    return pred_img, match_img


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
    values = []
    orient_values = []
    names = []
    for image in images:
        image_path = os.path.join(date_path, image)
        image = cv2.imread(image_path, 0)
        density, value = detect_density(image_path)
        values.append(value)
        density_values.append(density)
        orient_values.append(detect_orient(image))
        name = detect_lateral(image)
        names.append(name)

    if len(density_values) != 0:
        density, value = combine_density(density_values, values)
        if density == -1:
            continue
        folder_path = os.path.join(
            "output", patient + ' - ' + density + ' ' + str(value))
        os.mkdir(folder_path)
        top2 = sorted(orient_values)[-2]
        for index, image in enumerate(images):
            image_path = os.path.join(date_path, image)
            if orient_values[index] < top2:
                orient = "CC"
            else:
                orient = "MLO"
            name = names[index] + ' - ' + orient + '.jpg'
            shutil.copy(image_path, os.path.join(folder_path, name))