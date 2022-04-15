# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:30:42 2022

@author: Ashton Sobeck
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
import os

class MRIData():
    def __init__(self, file):
        self.file = file
        # 1310 data samples
        # going to use a 85/15 training/testing split
        # 1311 * 0.85 ~= 1114 training samples
        # 1311 * 0.15 ~= 197 testing samples
        self.df = pd.read_csv(self.file)
        self.facial_feature_yes_brain_loss_no = self.df.loc[
                (self.df['Recognizable-Facial-Feature'] == 'Yes') &
                (self.df['Brain-Feature-Loss'] == 'No')
            ]
        self.facial_feature_no_brain_loss_yes = self.df.loc[
                (self.df['Recognizable-Facial-Feature'] == 'No') &
                (self.df['Brain-Feature-Loss'] == 'Yes')
            ]
        self.facial_feature_no_brain_loss_no = self.df.loc[
                (self.df['Recognizable-Facial-Feature'] == 'No') &
                (self.df['Brain-Feature-Loss'] == 'No')
            ]
        
       
        
    def get_img(self, filename):
        img = nibabel.load(f'./BET_BSE_DATA/files/{filename}').get_fdata()
        img = np.array(img)
        new_depth = 75
        # need this because not every depth is the same, but we want them 
        # all to have same depth
        # every width and height is the same within the data 
        depth_resizing_factor = new_depth / img.shape[2]
        img = zoom(img, (.5,.5,depth_resizing_factor))
        return img
    
    def get_data(self):
       
        
        # pass in filename from dataframe into the get_img fn
       
        print('yes no')
        yes_no_images = np.array([self.get_img(self.facial_feature_yes_brain_loss_no['Filename'][i]) for i in self.facial_feature_yes_brain_loss_no.index.tolist()], dtype=object)
        # filter out None values
        yes_no_images = [img for img in yes_no_images if img is not None]
        print('no yes')
        no_yes_images = np.array([self.get_img(self.facial_feature_no_brain_loss_yes['Filename'][i]) for i in self.facial_feature_no_brain_loss_yes.index.tolist()], dtype=object)
        no_yes_images = [img for img in no_yes_images if img is not None]
        print('no no ')
        no_no_images = np.array([self.get_img(self.facial_feature_no_brain_loss_no['Filename'][i]) for i in self.facial_feature_no_brain_loss_no.index.tolist()], dtype=object)
        no_no_images = [img for img in no_no_images if img is not None]
        
        yes_no_labels = np.array([0 for _ in range(len(yes_no_images))])
        no_yes_labels = np.array([1 for _ in range(len(no_yes_images))])
        no_no_labels = np.array([2 for _ in range(len(no_no_images))])
        
        split_yes_no_idx = math.floor(.85 * len(yes_no_images))
        split_no_yes_idx = math.floor(.85 * len(no_yes_images))
        split_no_no_idx = math.floor(.85 * len(no_no_images))
        
        train_labels = np.concatenate((yes_no_labels[:split_yes_no_idx],
                                       no_yes_labels[:split_no_yes_idx],
                                       no_no_labels[:split_no_no_idx]),
                                       axis=0)
        train_imgs = np.concatenate((yes_no_images[:split_yes_no_idx],
                                     no_yes_images[:split_no_yes_idx],
                                     no_no_images[:split_no_no_idx]),
                                     axis=0)
        
        test_labels = np.concatenate((yes_no_labels[split_yes_no_idx:],
                                      no_yes_labels[split_no_yes_idx:],
                                      no_no_labels[split_no_no_idx:]),
                                      axis=0)
        test_imgs = np.concatenate((yes_no_images[split_yes_no_idx:],
                                    no_yes_images[split_no_yes_idx:],
                                    no_no_images[split_no_no_idx:]),
                                    axis=0)
        print(train_labels.shape)
        return (train_labels, train_imgs, test_labels, test_imgs)

def add_dim_to_img(img, label):
    img = tf.expand_dims(img, axis=3)
    return img, label

def make_model(w, h, d):
    # going to be using a 3d convolutional NN
    imgs = keras.Input((w, h, d, 1))
    model = layers.Conv3D(filters=128, kernel_size=3, activation='relu')(imgs)
    model = layers.MaxPooling3D()(model)
   
    model = layers.Conv3D(filters=256, kernel_size=3, activation='relu')(model)
    model = layers.MaxPooling3D()(model)
    model = layers.SpatialDropout3D(.4)(model)
    
    model = layers.GlobalAveragePooling3D()(model)
    model = layers.Dense(256, activation='relu')(model)
    model = layers.Dropout(.5)(model)
    # 3 is the number of classes that we have 
    # yes-no; no-yes; no-no
    outputs = layers.Dense(3, activation='softmax')(model)
    full_model = keras.Model(imgs, outputs, name='cnn')
    return full_model
def main():
    
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
    
    
    batch_size = 1
    epochs = 100
    
    train_dataset = (
        train_dl.shuffle(len(train_dl))
        .map(add_dim_to_img)
        .batch(batch_size)
    )
    print(train_dataset)
    print(train_dataset.take(1))
    test_dataset = (
        test_dl.shuffle(len(test_dl))
        .map(add_dim_to_img)
        .batch(batch_size)
    )
    
    model = make_model(w=128, h=128, d=75)
    model.summary()
    learning_rate = 1e-3
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()]
                  )
    train = model.fit(train_dataset, 
              epochs=epochs, 
              shuffle=True,
              callbacks=[
                  keras.callbacks.ModelCheckpoint(filepath='./team11_model_save_v2', 
                                                  monitor="val_loss",
                                                  mode="max",
                                                  save_freq='epoch',
                                                  save_best_only=True)  
                ],
              validation_data=(test_dataset)
              )
    
    
    model.load_weights('./team11_model_save_v2')
    
    classes =["identifiable-no-brain-damage", 
              "not-identifiable-with-brain-damage",
              "not-identifiable-no-brain-damage"
              ]
    for img, label in test_dataset:
        pred = model.predict(img)
        print(f'predicted class: {classes[tf.math.argmax(pred, axis=1)[0]]}')
        print(f'actual label: {classes[label[0]]}')
            
    results = model.evaluate(test_dataset)
    for result in results:
        print(result)
    plt.plot(train.history['sparse_categorical_accuracy'])
    plt.plot(train.history['val_sparse_categorical_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train','test'])
    plt.savefig('./acc_chart_v2.png')
    plt.show()
    plt.clf()
    
    plt.plot(train.history['loss'])
    plt.plot(train.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train','test'])
    plt.savefig('./loss_chart_v2.png')
    plt.show()
if __name__ == "__main__":
    main()
            
            
        
        