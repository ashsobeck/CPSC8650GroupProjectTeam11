# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:30:42 2022

@author: Ashton Sobeck
"""

import tensorflow as tf
from tensorflow import data
import matplotlib.pyplot as plt
import nibabel
import pandas as pd
import numpy as np
import math

a = nibabel.load('./BET_BSE_DATA/files/IXI002-Guys-0828-T1_bet_09.nii').get_fdata()
a = np.array(a)
print(a.shape)
plt.imshow(np.squeeze(a[:,:,30]), cmap="gray")

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
        return img if img.shape == (256, 256, 150) else None
    
    def get_data(self):
       
        
        # pass in filename from dataframe into the get_img fn
       
        print('yes no')
        yes_no_images = np.array([self.get_img(self.facial_feature_yes_brain_loss_no['Filename'][i]) for i in self.facial_feature_yes_brain_loss_no.index.tolist()[:100]], dtype=object)
        # filter out None values
        yes_no_images = [img for img in yes_no_images if img is not None]
        print('no yes')
        no_yes_images = np.array([self.get_img(self.facial_feature_no_brain_loss_yes['Filename'][i]) for i in self.facial_feature_no_brain_loss_yes.index.tolist()[:100]], dtype=object)
        no_yes_images = [img for img in no_yes_images if img is not None]
        print('no no ')
        no_no_images = np.array([self.get_img(self.facial_feature_no_brain_loss_no['Filename'][i]) for i in self.facial_feature_no_brain_loss_no.index.tolist()], dtype=object)
        no_no_images = [img for img in no_no_images if img is not None]
        
        yes_no_labels = np.array([0 for _ in range(len(yes_no_images))])
        no_yes_labels = np.array([1 for _ in range(len(no_yes_images))])
        # less than 100 in no no
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
   
        return (train_labels, train_imgs, test_labels, test_imgs)

def add_dim_to_img(img, label):
    img = tf.expand_dims(img, axis=3)
    return img, label

def main():
    dataset = MRIData('./BET_BSE_DATA/Label_file.csv')
    (train_labels, train_imgs, test_labels, test_imgs) = dataset.get_data()
    
    print(f'train_labels shape: {train_labels.shape}')
    print(f'train_imgs shape: {train_imgs[0].shape}')
    print(f'train_imgs shape: {train_imgs.shape}')
    
    print(f'test_labels shape: {test_labels.shape}')
    print(f'test_imgs shape: {test_imgs[0].shape}')
    print(f'test_imgs shape: {test_imgs.shape}')    
    
    train_dl = data.Dataset.from_tensor_slices((train_imgs, train_labels))
    test_dl = data.Dataset.from_tensor_slices((test_imgs, test_labels))
    
    batch_size = 8
    
    train_dataset = (
        train_dl.shuffle(len(train_imgs))
        .map(add_dim_to_img)
        .batch(batch_size)
        .prefetch(1)
    )
    
    test_dataset = (
        test_dl.shuffle(len(test_imgs))
        .map(add_dim_to_img)
        .batch(batch_size)
        .prefetch(1)
    )
    
if __name__ == "__main__":
    main()
            
            
        
        