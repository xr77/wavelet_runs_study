#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from mvpa2.suite import *
import os.path as op
import sklearn
import seaborn as sns
from pywt import wavedecn
from scipy import stats
from mpl_toolkits import mplot3d
from scipy.io import loadmat
from tqdm import tqdm
from scipy.ndimage.interpolation import shift
from matplotlib.pylab import *
import dtcwt


# In[3]:


subj_lst = ["s1","s2","s3","s5","s6","s7","s8","s10"]

for subject in tqdm(subj_lst):

    print "subject=", subject
    
    run_lst = np.loadtxt(subject +'/'+'short_run_list.txt',dtype = str)
    
    bold_fname = []

    for i in run_lst:
        
        all_runs = (subject+'/'+subject+'.nii/'+i+'.nii')
        
        bold_fname.append(all_runs)
    
    vt = fmri_dataset(subject+'/vt.nii') #load mask
    #vt.shape

    conditions=loadmat(subject+'/conds_short_tlrc.mat')
    conditions = conditions['conds_short_tlrc']
    conditions_sh2 = shift(conditions,[0,2], cval=0) #shift by 2 TRs
    
    def convert_binary_to_multiclass(binary_conditions):
        """Convert binary representation into multiclass reprentation:
        For example: convert [[1 1 1 1 0 0 0 0]
                              [0 0 0 0 1 1 1 1]]
        to [1 1 1 1 2 2 2 2]"""
        x,y = np.where(binary_conditions)
        conditions=np.zeros(binary_conditions.shape[1])
        conditions[y]=x+1
        return conditions

    conditions_multi = convert_binary_to_multiclass(conditions_sh2)
    
    runs = np.arange(0,512)/32
    
    ds = fmri_dataset (bold_fname, mask = subject+'/vt.nii', targets = conditions_multi, chunks = runs)#mask = vt doesn't work here, because the error message 'array must be sequence' therefore, it has to be load with the file name
    #print ds.summary()
    
    orig_data = ds.a.mapper.reverse(ds.samples)
    orig_vt = vt.a.mapper.reverse(vt.samples)
    #orig_vt.shape
    new_data = orig_vt * orig_data
    
    var_L1=[]
    var_L2=[]
    var_L3=[]
    var_L4=[]
    var_L5=[]

    trans = dtcwt.Transform3d()

    #animals = [insect1,insect2,insect3,insect4,bird1,bird2,bird3,bird4,monkey1,monkey2,monkey3,monkey4]
    #for i in range(5):
    print "starting wavelet transform"

    for i in tqdm(range(new_data.shape[0])): #transform all 1230 trs
        TR=new_data[i,:,:,:]
        wvt = trans.forward(TR,nlevels=5)

        # level 1:
        highpasses = wvt.highpasses[0]
        highpasses = np.ma.log(np.abs(highpasses))
        var_per_orientation = highpasses.var(axis=(0,1,2)) #0,1,2 means take var across x,y,z
        #total_var = np.mean(var_per_orientation)  
        var_L1.append(var_per_orientation)

        # level 2:
        highpasses = wvt.highpasses[1]
        highpasses = np.ma.log(np.abs(highpasses))
        var_per_orientation = highpasses.var(axis=(0,1,2))
        #total_var = np.mean(var_per_orientation)
        var_L2.append(var_per_orientation)

        # level 3:
        highpasses = wvt.highpasses[2]
        highpasses = np.ma.log(np.abs(highpasses))
        var_per_orientation = highpasses.var(axis=(0,1,2))
        #total_var = np.mean(var_per_orientation) 
        var_L3.append(var_per_orientation)

        # level 4:
        highpasses = wvt.highpasses[3]
        highpasses = np.ma.log(np.abs(highpasses))
        var_per_orientation = highpasses.var(axis=(0,1,2))
        #total_var = np.mean(var_per_orientation)  
        var_L4.append(var_per_orientation)

        # level 5:
        highpasses = wvt.highpasses[4]
        highpasses = np.ma.log(np.abs(highpasses))
        var_per_orientation = highpasses.var(axis=(0,1,2))
        #total_var = np.mean(var_per_orientation)  
        var_L5.append(var_per_orientation)
        
    L1_ALLTR=np.vstack( var_L1 )
    L2_ALLTR=np.vstack( var_L2 )
    L3_ALLTR=np.vstack( var_L3 )
    L4_ALLTR=np.vstack( var_L4 )
    L5_ALLTR=np.vstack( var_L5 )



    ALL_Level_TR=np.stack([L1_ALLTR,L2_ALLTR,L3_ALLTR,L4_ALLTR,L5_ALLTR], axis=1)#vstack can only take a list



    #ALL_Level_TR.shape #1230 TR, 5 levels, each level 28 orientations. 
    #np.save("all_subjs_TRs_levels_log_BA17"+subjects,ALL_Level_TR.data)
    np.save("runs_study_vt_"+subject,ALL_Level_TR.data)


# In[ ]:




