# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:30:42 2022

@author: Ashton
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel
import numpy as np

a = nibabel.load('./BET_BSE_DATA/files/IXI002-Guys-0828-T1_bet_09.nii').get_fdata()
a = np.array(a)
print(a.shape)
plt.imshow(np.squeeze(a[:,:,30]), cmap="gray")