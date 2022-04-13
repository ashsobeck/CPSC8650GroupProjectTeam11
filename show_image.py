# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:05:44 2022

@author: Ashton
"""

import nibabel
import numpy as np
import matplotlib.pyplot as plt

a = nibabel.load('./BET_BSE_DATA/files/IXI002-Guys-0828-T1_bet_09.nii').get_fdata()
a = np.array(a)
print(a.shape)
plt.imshow(np.squeeze(a[:,:,30]), cmap="gray")