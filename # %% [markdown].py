# %% [markdown]
# #**Download dataset and installations**

# %%
from google.colab import drive
drive.mount('/content/drive')

# Download the dataset from this link, you need to be logged in Kaggle:
# https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/download?datasetVersionNumber=1
# save it in your Google Drive or in a folder,
# for this notebook this is the path /content/drive/MyDrive/Datasets/Brain_MRI_Images_for_Brain_tumor_detection/archive.zip
# make a new directory
!mkdir dataset
%cd dataset

# Copy the dataset from the drive
!cp /content/drive/MyDrive/Datasets/Brain_MRI_Images_for_Brain_tumor_detection/archive.zip .
!unzip -q archive.zip
!mkdir models

# %%
!wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

# %% [markdown]
# #**Import**

# %%
import os
import cv2
import glob
import random
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
import math
import gc
import keras

from sklearn.model_selection import train_test_split, KFold
from scipy.stats import entropy, skew
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.callbacks import LambdaCallback

SEED = 123
WEIGHT_PATH_VGG16 = os.getcwd() + '/' + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def grab_contours(cnts:tuple) -> tuple:
  # if the length the contours tuple returned by cv2.findContours
  # is '2' then we are using either OpenCV v2.4, v4-beta, or
  # v4-official
  if len(cnts) == 2:
      cnts = cnts[0]

  # if the length of the contours tuple is '3' then we are using
  # either OpenCV v3, v4-pre, or v4-alpha
  elif len(cnts) == 3:
      cnts = cnts[1]

  # otherwise OpenCV has changed their cv2.findContours return
  # signature yet again and I have no idea WTH is going on
  else:
      raise Exception(("Contours tuple must have length 2 or 3, "
          "otherwise OpenCV changed their cv2.findContours return "
          "signature yet again. Refer to OpenCV's documentation "
          "in that case"))

  # return the actual contours array
  return cnts

def crop(set_to_crop: np.array, pixel_to_add: int=0, tresh_method: str = 'static') -> np.array:
  #different modality for tresholding:
  # - static, normal tresholding with a fixed tresh
  # - adaptive_mean, The threshold value is the mean of the neighbourhood area minus the constant C
  # - adaptive_gaussian, The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C.
  # - otsu, apply to otsu binarization technique
  # preprocessing, pretty much like this cropping https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
  IMG_SIZE = (224,224)
  new_set = []

  for img_rgb in set_to_crop:
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    if tresh_method == 'static':
      thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]
    elif tresh_method == 'otsu':
      thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    extRight = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
    extTop = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
    extBot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])


    up = extTop[1]-pixel_to_add
    down = extBot[1]+pixel_to_add
    left = extLeft[0]-pixel_to_add
    right = extRight[0]+pixel_to_add

    # Check if up and down are within the range [0, LEN]
    if up < 0 :
      up = 0
    elif up > LEN:
      up = LEN
      print(" WARNING crop: up is outside the valid range.")

    if down > LEN:
      down = LEN
    elif down < 0:
      down = 0
      print(" WARNING crop: down is outside the valid range.")

    if left < 0:
      left = 0
    elif left > WID:
      left = WID
      print(" WARNING crop: left is outside the valid range.")

    if right > WID:
      right = WID
    elif right < 0:
      right = 0
      print(" WARNING crop: right is outside the valid range.")


    new_img = img_rgb[up:down, left:right].copy()

    #interpolation INTER_CUBIC is good for enlarge and INTER_NEAREST for shrink
    if np.shape(img)[0] > IMG_SIZE[0] or np.shape(img)[1] > IMG_SIZE[0]:
      # the image is bigger than IMG_SIZE
      new_img = cv2.resize(img_rgb, dsize=IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    else: #the image is smaller than IMG_SIZE
      new_img = cv2.resize(img_rgb, dsize=IMG_SIZE, interpolation=cv2.INTER_CUBIC)

    new_set.append(new_img)

  return np.array(new_set)


def ratio_imgs(set_name: np.array) -> None:

  #size of the images
  list_sizes = []
  list_ratios = []
  for img in set_name:
    list_sizes.append(img.shape)
    list_ratios.append(img.shape[0]/img.shape[1])

  # plt.hist(list_sizes)
  plt.figure()
  _ = plt.hist(list_ratios, bins=20)
  plt.title('Distribution of ratios')


def plot_samples(set_name: np.array, set_labels: np.array,  num_samples: int = 50) -> None:

  columns = 10
  plt.figure(figsize=(10,5))
  for i in range(num_samples):
    rnd = random.randint(0,len(set_name))

    plt.subplot(int(num_samples/columns),columns,i+1)
    plt.imshow(set_name[rnd-1], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('yes' if set_labels[rnd]==1 else 'no')
    plt.tight_layout()

    #TODO gives an error when it select the last imageù

def save_images(X, y, folder_name) -> None:
    i = 0
    for (img, imclass) in zip(X, y):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1

def plot_cm(cm: np.array, classes: list, normalize: bool = False, title: str = 'Confusion matrix') -> None:
  if normalize:
    plt.figure(figsize=(8, 6))
    s = sns.heatmap(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis],
                    annot=True,
                    # annot_kws={"size": 16},
                    fmt='.1%',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

  else:
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)  # Adjust font size if needed
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
                )

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)


def plot_performances(history: object) -> None:
  # plot model performance
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(1, len(history.epoch) + 1)

  plt.figure(figsize=(15,5))

  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Train Set')
  plt.plot(epochs_range, val_acc, label='Val Set')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Model Accuracy')
  plt.ylim([0,1.1])
  plt.grid()


  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Train Set')
  plt.plot(epochs_range, val_loss, label='Val Set')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Model Loss')
  plt.grid()

  plt.tight_layout()
  plt.show()


def create_vgg16model(lr: float, optimizer: object, vgg16_weight_path: str = WEIGHT_PATH_VGG16) -> Model:


  base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False,
    input_shape=IMG_SIZE + (3,)
    )

  NUM_CLASSES = 1

  model = Sequential()
  model.add(base_model)
  model.add(layers.Flatten())
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

  model.layers[0].trainable = False

  model.compile(
      loss='binary_crossentropy',
      optimizer = optimizer(learning_rate = lr),
      metrics=['accuracy']
  )

  # model.summary()
  return model

def create_generators(TRAIN_DIR: str, VAL_DIR : str) -> tuple:
  train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
    )

  val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    )

  train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=SEED
    )

  validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=SEED
    )

  return train_generator, validation_generator


# %% [markdown]
# #**Loading dataset**

# %%
path = "/content/dataset/brain_tumor_dataset"
X, y = load_data(path)

# %% [markdown]
# #**Train-Test-Val Splitting**

# %%
# Train Test Validation splitting
X_train, X_remain, y_train, y_remain = train_test_split(X, y, train_size=int(np.ceil(len(X)*0.76)), random_state=SEED)
X_test, X_val, y_test, y_val = train_test_split(X_remain, y_remain, train_size=0.18, random_state=SEED)

del X_remain, y_remain

# %% [markdown]
# #**EDA**

# %%
# Distribution of ratios
ratio_imgs(X)

# %%
#distribution of classes initial dataset
plot_classes_distribution(y)

# %%
#plot examples from original dataset

plt.figure(figsize=(10,5))
for i in range(10):
  rnd = random.randint(0,len(X))
  plt.subplot(2,5,i+1)
  plt.imshow(X[rnd])
  plt.xticks([])
  plt.yticks([])
  plt.title('yes' if y[rnd]==1 else 'no')
  # plt.tight_layout()

# %%
#distribution of classes train test and validation
plot_classes_distribution([y_train,y_test,y_val],['Train','Test','Validation'])

# %% [markdown]
# #**Interpolation**

# %%

IMG_SIZE = (224,224)
img = X[0]

#different interpolations
dict_interpolations = {
  'INTER_NEAREST': cv2.INTER_NEAREST,  # fastest, blockiest
  'INTER_LINEAR': cv2.INTER_LINEAR,   # good compromise
  'INTER_AREA': cv2.INTER_AREA,    # slower, smoother than INTER_LINEAR
  'INTER_CUBIC': cv2.INTER_CUBIC,   # slowest, smoothest
  'INTER_LANCZOS4': cv2.INTER_LANCZOS4,  # even slower, even smoother
}

plt.figure(figsize=(12,4))
plt.suptitle('Different interpolations')
plt.subplot(1,6,1)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.xlabel(f'{entropy(img.ravel())}')

for i,interp in enumerate(dict_interpolations):
  plt.subplot(1,6,i+2)
  imginterp = cv2.resize(img, dsize=IMG_SIZE, interpolation=dict_interpolations[interp])
  plt.imshow(imginterp)
  plt.title( str(interp))
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(f'{entropy(imginterp.ravel())}')
plt.tight_layout()

# INTER_NEAREST: fastest, blockiest, good for shrinking
# INTER_LINEAR: good compromise, good for both shrinking and enlarging
# INTER_AREA: slower, smoother than INTER_LINEAR, good for shrinking
# INTER_CUBIC: slowest, smoothest, good for enlarging
# INTER_LANCZOS4: even slower, even smoother, good for enlarging

# For resizing an image to 224x224 pixels, INTER_LINEAR or INTER_CUBIC are good choices.
# INTER_NEAREST is a good choice for shrinking images, while INTER_CUBIC is a good choice for enlarging images.

# INTER_CUBIC is ok

#TODO control if it is possible to cut the black before interpolation, maybe there is less loss of information

# %% [markdown]
# #**Preprocessing demo**

# %%
# preprocessing, one image, cropping https://pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
IMG_SIZE = (224,224)
rnd = random.randint(1, len(X))
print('image num. ', rnd )
img_rgb = X[106]
LEN, WID, _ = img_rgb.shape

# from RGB to gray scale

#before converting it we have to check if we are losing information with the gray scale conversion
# a simple way is to calculate the entropy after and before
print(f'original entropy {entropy(img_rgb.ravel())}')
plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.title('original')
# plt.xticks([])
# plt.yticks([])

img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
plt.subplot(2,3,2)
plt.imshow(img,cmap='gray')
plt.title('gray')
plt.xticks([])
plt.yticks([])
print(f'entropy after conversion {entropy(img.ravel())}')


# gaussian filter
img = cv2.GaussianBlur(img, (5,5), 0)
plt.subplot(2,3,3)
plt.imshow(img,cmap='gray')
plt.title('filtered')
plt.xticks([])
plt.yticks([])

thresh = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY)[1]

plt.subplot(2,3,4)
plt.imshow(thresh,cmap='gray')
plt.title('thresh')
plt.xticks([])
plt.yticks([])

thresh = cv2.erode(thresh, None, iterations=2)
plt.subplot(2,3,5)
plt.imshow(thresh,cmap='gray')
plt.title('erosion')
plt.xticks([])
plt.yticks([])

thresh = cv2.dilate(thresh, None, iterations=2)
plt.subplot(2,3,6)
plt.imshow(thresh,cmap='gray')
plt.title('dilatation')
plt.xticks([])
plt.yticks([])

# find contours in thresholded image, then grab the largest
# one
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# first one is the retrieval mode,
# RETR_EXTERNAL retrieves only the extreme outer contours
# RETR_LIST retrieves all of the contours and organizes them into a two-level hierarchy. At the top level, there are external boundaries of the components.
#           At the second level, there are boundaries of the holes. If there is another contour inside a hole of a connected component, it is still put at the top level.
# RETR_TREE  retrieves all of the contours and reconstructs a full hierarchy of nested contours.
# RETR_FLOODFILL  not specified in the documentation

# second one is the aproximation method
# CHAIN_APPROX_NONE stores absolutely all the contour points.
# CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
#                     For example, an up-right rectangular contour is encoded with 4 points.
# CHAIN_APPROX_TC89_L1, CHAIN_APPROX_TC89_KCOS, Teh-Chin chain approximation algorithm


# cnts = imutils.grab_contours(cnts) #basically this is doing cnts[0] when you don't specify hierarchy in the output of findContours
# cnts = cnts[0] #this is the same for openCV version 2.4

# Contours tuple must have length 2 or 3, otherwise OpenCV changed their cv2.findContours return
for i in range(len(cnts)):
    n_points = len(cnts[i])
    print(f"Contour {i + 1}: {n_points} points")


num_contours = len(cnts)
plt.figure(figsize=(20, 5))
plt.suptitle("Contours")
for i in range(len(cnts)):
    img_ = img_rgb.copy()
    plt.subplot(1,num_contours,i+1)
    color = np.random.randint(0, 256, 3) #random select a color
    color = (int(color[0]), int(color[1]), int(color[2])) #conversion from int64 to int because opencv does not like int64
    cv2.drawContours(img_, cnts[i], -1, tuple(color), 10)
    plt.imshow(img_,cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()


max_contour = max(cnts, key=cv2.contourArea)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title("Max contour")
img_ = img_rgb.copy()
cv2.drawContours(img_, max_contour, -1, (153, 102, 204), 5)
plt.imshow(img_,cmap='gray')
plt.xticks([])
plt.yticks([])

# find the extreme points
extLeft = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
extRight = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
extTop = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
extBot = tuple(max_contour[max_contour[:, :, 1].argmax()][0])

ADD_PIXELS = 4
up = extTop[1]-ADD_PIXELS
down = extBot[1]+ADD_PIXELS
left = extLeft[0]-ADD_PIXELS
right = extRight[0]+ADD_PIXELS

# Check if up and down are within the range [0, LEN]
if up < 0 :
  up = 0
elif up > LEN:
  print(" WARNING crop: up is outside the valid range.")

if down > LEN:
  down = LEN
elif down < 0:
  print(" WARNING crop: down is outside the valid range.")

if left < 0:
  left = 0
elif left > WID:
  print(" WARNING crop: left is outside the valid range.")

if right > WID:
  right = WID
elif right < 0:
  print(" WARNING crop: right is outside the valid range.")

# add extreme points
img_pnt = cv2.circle(img_.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

plt.subplot(1,3,2)
plt.imshow(img_pnt,cmap='gray')
plt.title("Points")
plt.xticks([])
plt.yticks([])

# crop
new_img = img[up:down, left:right].copy()

plt.subplot(1,3,3)
plt.imshow(new_img,cmap='gray')
plt.title("Cropped")
plt.xticks([])
plt.yticks([])

#TODO some images have some writings in the upper left and in the lower left
# implement in the preprocessing ocr to recognize writings and delete it
# is important to avoid the CNN to learn the writings and make decision based on that

# %% [markdown]
# #**Crop and Preprocessing for VGG16**

# %%
X_train_crop = crop (X_train)
X_val_crop = crop (X_val)
X_test_crop = crop (X_test)

# %%
#plot examples from cropped dataset
# plot_samples(X_train_crop, y_train)

# %%
!mkdir CROP CROP/TRAIN_CROP CROP/TEST_CROP CROP/VAL_CROP CROP/TRAIN_CROP/YES CROP/TRAIN_CROP/NO CROP/TEST_CROP/YES CROP/TEST_CROP/NO CROP/VAL_CROP/YES CROP/VAL_CROP/NO

save_images(X_train_crop, y_train, folder_name='CROP/TRAIN_CROP/')
save_images(X_val_crop, y_val, folder_name='CROP/VAL_CROP/')
save_images(X_test_crop, y_test, folder_name='CROP/TEST_CROP/')

# %%
#preprocessing for the validation and test

# Create an empty array to store preprocessed images
pre_X_test = np.empty_like(X_test_crop)
pre_X_val = np.empty_like(X_val_crop)

# Preprocess each image in X_test using a loop
pre_X_test = np.array([preprocess_input(img) for img in X_test_crop])
pre_X_val = np.array([preprocess_input(img) for img in X_val_crop])

# %% [markdown]
# #**Demo Image Generator**

# %%

#demo of imagegenerator

demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

# Load an image
image = X_train_crop[random.randint(1,len(X_train_crop)-1)]

# the rank of the image must be 4
# image = image.reshape((1,224,224,1))
image = image.reshape((1,224,224,3))

# Generate augmented images
augmented_images = demo_datagen.flow(
    x=image,
    batch_size=1,
    shuffle=False
)

num_augmented_images = 10

# Plot the original image
plt.figure(figsize=(8, 4))
plt.imshow(image[0, :, :, 0], cmap='gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

# Plot the augmented images
plt.figure(figsize=(10,5))
for i, augmented_image in enumerate(augmented_images):
    columns = 5
    plt.subplot(int(num_augmented_images/columns),columns,i+1)
    plt.imshow(augmented_image[0, :, :, 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])

    if i >= num_augmented_images - 1:
        break


# %% [markdown]
# #**Image Generators**

# %%
TRAIN_DIR = 'CROP/TRAIN_CROP/'
VAL_DIR = 'CROP/VAL_CROP/'

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=SEED
)


validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=SEED
)

# %% [markdown]
# #**Hyperparameters searching (Learning rate range test + Grid search)**

# %%
!mkdir lr_tests_adam, lr_tests_rms

# %%
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    verbose = 1
)

EPOCHS = 30
lr_list = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# %%
# For loop to test the learning rate with adam

histories_adam_range = []
optimizer = tf.keras.optimizers.Adam

for lr in lr_list:

  print(f'---------- lr {lr:.0e} ----------')

  with tf.device('gpu'):
    gc.collect()

  save_weights_at = os.getcwd() +'/lr_tests_adam/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M")+'.h5'

  checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                  filepath=save_weights_at,
                                  monitor='val_loss',
                                  mode='min',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False
                                  )

  model = create_vgg16model(lr, optimizer, WEIGHT_PATH_VGG16)

  #training
  history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator), #num samples 193, size batch 32, steps per epoch = 193/32 = 6
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(validation_generator), #num samples 50, size batch 32, steps per epoch = 50/32 = 1
    callbacks = [es, checkpoint]
  )

  histories_adam_range.append(history)

  del model
  tf.keras.backend.clear_session()

# %%
fig, axs = plt.subplots(2,figsize=(20,10))

for i,history in enumerate(histories_adam_range):
  # plot model performance
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(1, len(history.epoch) + 1)

  print(f'Train lr = {lr_list[i]:.0e} ----> Acc = {max(acc):.2f}')
  print(f'Val   lr = {lr_list[i]:.0e} ----> Acc = {max(val_acc):.2f}\n')

  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label=f'Train Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Model Accuracy')
  plt.ylim([0.35,1])

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, val_acc, label=f'Val Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Model Accuracy')
  plt.ylim([0.35,1])

  # for ax in axs.flat:
  #     ax.grid(True)


# %%
plt.figure(figsize=(20,10))

for i,history in enumerate(histories_adam_range):

  # plot model performance
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(1, len(history.epoch) + 1)

  print(f'Train lr = {lr_list[i]:.0e} ----> Loss = {min(loss):.2f}')
  print(f'Val   lr = {lr_list[i]:.0e} ----> Loss = {min(val_loss):.2f}\n')

  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, loss, label=f'Train Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Model Loss')
  plt.ylim([0,5])

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, val_loss, label=f'Val Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Model Loss')
  plt.ylim([0,5])



# %%
folder = '/content/dataset/lr_tests_adam'
for i,model_path in enumerate(os.listdir(folder)):
  #load the best model
  model = tf.keras.models.load_model(os.path.join(folder, model_path))

  # validate on test set
  predictions = model.predict(pre_X_test)

  predictions = [1 if x>0.5 else 0 for x in predictions]

  accuracy = accuracy_score(y_test, predictions)
  print(f'Test Accuracy with lr {lr_list[i]:.0e} = {accuracy:.2f}')


# %%
# For loop to test the learning rate with adam

histories_rms_range = []
optimizer = tf.keras.optimizers.RMSprop

for lr in lr_list:

  print(f'---------- lr {lr:.0e} ----------')

  with tf.device('gpu'):
    gc.collect()

  save_weights_at = os.getcwd() +'/lr_tests_adam/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M")+'.h5'

  checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                  filepath=save_weights_at,
                                  monitor='val_loss',
                                  mode='min',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False
                                  )

  model = create_vgg16model(lr, optimizer, WEIGHT_PATH_VGG16)

  #training
  history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator), #num samples 193, size batch 32, steps per epoch = 193/32 = 6
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(validation_generator), #num samples 50, size batch 32, steps per epoch = 50/32 = 1
    callbacks = [es, checkpoint]
  )

  histories_rms_range.append(history)

  del model
  tf.keras.backend.clear_session()

# %%
fig, axs = plt.subplots(2,figsize=(20,10))

for i,history in enumerate(histories_rms_range):
  # plot model performance
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(1, len(history.epoch) + 1)

  print(f'Train lr = {lr_list[i]:.0e} ----> Acc = {max(acc):.2f}')
  print(f'Val   lr = {lr_list[i]:.0e} ----> Acc = {max(val_acc):.2f}\n')

  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label=f'Train Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Model Accuracy')
  plt.ylim([0.35,1])

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, val_acc, label=f'Val Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Model Accuracy')
  plt.ylim([0.35,1])

  # for ax in axs.flat:
  #     ax.grid(True)


# %%
plt.figure(figsize=(20,10))

for i,history in enumerate(histories_rms_range):

  # plot model performance
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(1, len(history.epoch) + 1)

  print(f'Train lr = {lr_list[i]:.0e} ----> Loss = {min(loss):.2f}')
  print(f'Val   lr = {lr_list[i]:.0e} ----> Loss = {min(val_loss):.2f}\n')

  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, loss, label=f'Train Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Model Loss')
  plt.ylim([0,5])

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, val_loss, label=f'Val Set {lr_list[i]:.0e}')
  plt.legend(loc="best")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Model Loss')
  plt.ylim([0,5])



# %%
folder = '/content/dataset/lr_tests_rms'
for i,model_path in enumerate(os.listdir(folder)):
  #load the best model
  model = tf.keras.models.load_model(os.path.join(folder, model_path))

  # validate on test set
  predictions = model.predict(pre_X_test)

  predictions = [1 if x>0.5 else 0 for x in predictions]

  accuracy = accuracy_score(y_test, predictions)
  print(f'Test Accuracy with lr {lr_list[i]:.0e} = {accuracy:.2f}')

# %% [markdown]
# #**Model Building**

# %%
#Instantiate base model and load pre-trained weights into it

vgg16_weight_path = '/content/dataset/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

# %%
NUM_CLASSES = 1
LR = 1e-4

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss = 'binary_crossentropy',
    optimizer = tf.keras.optimizers.Adam(learning_rate = LR),
    metrics = ['accuracy']
)

model.summary()

# %% [markdown]
# #**Training**

# %%
EPOCHS = 100

#callbacks for training

es = EarlyStopping(
    # monitor='val_accuracy',
    monitor='val_loss',
    # mode='max',
    mode='min',
    patience=15,
    verbose = 1
)

save_weights_at = os.getcwd() +'/models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M")+'.h5'

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights_at,
                                          monitor='val_loss',
                                          mode='min',
                                          verbose = 0,
                                          save_best_only=True,
                                          save_weights_only=False)

#fit

history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator), #num samples 193, size batch 32, steps per epoch = 193/32 = 6
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(validation_generator), #num samples 50, size batch 32, steps per epoch = 50/32 = 1
    callbacks = [es,checkpoint]
)

plot_performances(history)

# %%
#preprocessing for the validation and test

# Create an empty array to store preprocessed images
pre_X_test = np.empty_like(X_test_crop)
pre_X_val = np.empty_like(X_val_crop)

# Preprocess each image in X_test using a loop
pre_X_test = np.array([preprocess_input(img) for img in X_test_crop])
pre_X_val = np.array([preprocess_input(img) for img in X_val_crop])

# Define class labels (assuming 0, not tumor, and 1, yes tumor)
class_names = ['Class 0, No', 'Class 1, Yes']

#load the best model
model = tf.keras.models.load_model(save_weights_at)

# validate on validation set
predictions = model.predict(pre_X_val)

#plot predictions
# plt.figure()
# plt.plot(predictions,'bo')
# plt.axhline(y=0.5, color='r', linestyle='-')


predictions = [1 if x>0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_val, predictions)
print('Val Accuracy = %.2f' % accuracy)

cm = confusion_matrix(y_val, predictions)

plot_cm(cm, class_names, title = 'Confusion matrix validation')
plot_cm(cm, class_names, normalize = True, title = 'Confusion matrix validation percentage')

# validate on test set
predictions = model.predict(pre_X_test)

#plot predictions
# plt.figure()
# plt.plot(predictions,'bo')
# plt.axhline(y=0.5, color='r', linestyle='-')


predictions = [1 if x>0.5 else 0 for x in predictions]

accuracy = accuracy_score(y_test, predictions)
print('Test Accuracy = %.2f' % accuracy)

cm = confusion_matrix(y_test, predictions)

plot_cm(cm, class_names, title = 'Confusion matrix test')
plot_cm(cm, class_names, normalize = True, title = 'Confusion matrix test percentage')

# %% [markdown]
# #**Validation with KFold**

# %%
!mkdir FOLD FOLD/TRAIN_CROP FOLD/VAL_CROP FOLD/TRAIN_CROP/YES FOLD/TRAIN_CROP/NO FOLD/VAL_CROP/YES FOLD/VAL_CROP/NO FOLD/models

folder_names = ['FOLD/TRAIN_CROP/YES', 'FOLD/TRAIN_CROP/NO', 'FOLD/VAL_CROP/YES', 'FOLD/VAL_CROP/NO']

X_fold = np.concatenate((X_train, X_val))
y_fold = np.concatenate((y_train, y_val))

# %%
EPOCHS = 100
n_split = 10
i = 0
accuracy_fold = []
cm_fold = []
histories = []


for train_fold_index, val_fold_index in KFold(n_split).split(X_fold):
  print('Fold n.', i)

  #splitting fold
  x_train_fold, x_val_fold = X_fold[train_fold_index], X_fold[val_fold_index]
  y_train_fold , y_val_fold = y_fold[train_fold_index], y_fold[val_fold_index]

  #crop
  X_train_crop = crop(x_train_fold)
  X_val_crop = crop(x_val_fold)

  #saving imgs
  save_images(X_train_crop, y_train, folder_name='FOLD/TRAIN_CROP/')
  save_images(X_val_crop, y_val, folder_name='FOLD/VAL_CROP/')

  #generators
  train_generator, validation_generator = create_generators('FOLD/TRAIN_CROP/', 'FOLD/VAL_CROP/')

  with tf.device('gpu'):
    gc.collect()

  #model
  model = create_vgg16model(1e-4, WEIGHT_PATH_VGG16)

  # print('Initial weights:')
  # print(model.layers[3].get_weights()[0])

  #callbacks for training

  es = EarlyStopping(
      # monitor='val_accuracy',
      monitor='val_loss',
      # mode='max',
      mode='min',
      patience=10,
      verbose = 1
  )

  save_weights_at = os.getcwd() +'/FOLD/models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M")+'.h5'

  checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights_at,
                                            monitor='val_loss',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False)

  #training
  history = model.fit(
    train_generator,
    steps_per_epoch = len(train_generator), #num samples 193, size batch 32, steps per epoch = 193/32 = 6
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = len(validation_generator), #num samples 50, size batch 32, steps per epoch = 50/32 = 1
    callbacks = [es, checkpoint]
  )
  histories.append(history)
  # print('Final weights:')
  # print(model.layers[3].get_weights()[0])

  del model

  # #load the best model
  # model = tf.keras.models.load_model(save_weights_at)

  # # validate on test set
  # predictions = model.predict(X_test_crop)

  # predictions = [1 if x>0.5 else 0 for x in predictions]

  # accuracy = accuracy_score(y_test, predictions)
  # print('Test Accuracy = %.2f' % accuracy)

  # cm = confusion_matrix(y_test, predictions)

  # accuracy_fold.append(accuracy)
  # cm_fold.append(cm)

  #cleaning the folder
  for folder in folder_names:
    for file in os.listdir(folder):
      os.remove(os.path.join(folder, file))

  tf.keras.backend.clear_session()
  # del model

  i += 1


# %%
folder = '/content/dataset/FOLD/models'
for model_path in os.listdir(folder):
  #load the best model
  model = tf.keras.models.load_model(os.path.join(folder, model_path))

  # validate on test set
  predictions = model.predict(pre_X_test)

  predictions = [1 if x>0.5 else 0 for x in predictions]

  accuracy = accuracy_score(y_test, predictions)
  print('Test Accuracy = %.2f' % accuracy)

# %%
for history in histories:
  plot_performances(history)

# %%
# Create an empty array to store preprocessed images
preprocessed_X_test = np.empty_like(X_test)

# Preprocess each image in X_test using a loop
for i in range(len(X_test)):
    preprocessed_X_test[i] = preprocess_input(X_test[i])

# preprocessed_X_test_tf = tf.convert_to_tensor(preprocessed_X_test, dtype=tf.float32)


