# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:27:16 2022

@author: Ashton
"""

import tensorflow as tf
from tensorflow import data
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import nibabel
import pandas as pd
import numpy as np
import math
from scipy.ndimage import zoom

from modelTeam11v1 import make_model as model_v1
from modelTeam11v2 import make_model as model_v2
from modelTeam11v3 import make_model as model_v3

from modelTeam11v1 import MRIData, add_dim_to_img

dataset = MRIData('./BET_BSE_DATA/Label_file.csv')
(train_labels, train_imgs, test_labels, test_imgs) = dataset.get_data()
train_imgs = np.asarray(train_imgs).astype(np.float32)
test_imgs = np.asarray(test_imgs).astype(np.float32)
print(f'train_labels shape: {train_labels.shape}')
print(f'train_imgs shape: {train_imgs[0].shape}')
print(f'train_imgs shape: {train_imgs.shape}')

print(f'test_labels shape: {test_labels.shape}')
print(f'test_imgs shape: {test_imgs[0].shape}')
print(f'test_imgs shape: {test_imgs.shape}')    

train_dl = data.Dataset.from_tensor_slices((train_imgs, train_labels))
test_dl = data.Dataset.from_tensor_slices((test_imgs, test_labels))

test_dataset = (
    test_dl.shuffle(len(test_dl))
    .map(add_dim_to_img)
    .batch(2)
)

model_1 = model_v1(128, 128, 75)
model_2 = model_v2(128, 128, 75)
model_3 = model_v3(128, 128, 75)

model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()]
              )
model_2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()]
              )
model_3.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()]
              )

model_1.load_weights('./team11_model_save_v1')
model_2.load_weights('./team11_model_save_v2')
model_3.load_weights('./team11_model_save_v3')

first_results = model_1.evaluate(test_dataset)
second_results = model_2.evaluate(test_dataset)
third_results = model_3.evaluate(test_dataset)

print('first model result')
for result in first_results:
    print(result)

for result in second_results:
    print(result)

for result in third_results:
    print(result)