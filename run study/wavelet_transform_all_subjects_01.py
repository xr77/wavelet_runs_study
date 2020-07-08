#!/usr/bin/env python
# coding: utf-8

# ## Spatial frequency study

# In[2]:


# load all the packages and libraries going to be used
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


# In[3]:


# First, let's load fMRI data, 10 runs
sub_list=["s1","s2","s3","s5","s6","s7","s8","s10"]
for subjects in tqdm(sub_list):

    print "subject=", subjects

    bold_fname = ('all_subjects_18'+'/'+subjects+'.nii/01.nii','all_subjects_18'+'/'+subjects+'.nii/02.nii','all_subjects_18'+'/'+subjects+'.nii/03.nii','all_subjects_18'+'/'+subjects+'.nii/04.nii','all_subjects_18'+'/'+subjects+'.nii/05.nii','all_subjects_18'+'/'+subjects+'.nii/06.nii','all_subjects_18'+'/'+subjects+'.nii/07.nii','all_subjects_18'+'/'+subjects+'.nii/08.nii','all_subjects_18'+'/'+subjects+'.nii/09.nii','all_subjects_18'+'/'+subjects+'.nii/10.nii')

    ds= fmri_dataset(bold_fname)



    VT = loadmat ("all_subjects_18/vt.mat") #load mat.file into python.
    VT = VT['VT'] #get the value out of dictionary
    #BA17 = loadmat ("all_subjects_18/BA17.mat")
    #BA17 =BA17['BA17']



    # Load in the mask of the ROI
    #ds = fmri_dataset (bold_fname, mask=BA17)
    ds = fmri_dataset (bold_fname, mask=VT)




    # Load in the condition label file
    conditions=loadmat('all_subjects_18/regressors_sh3.mat')
    conditions = conditions['regressors_sh3']



    def convert_binary_to_multiclass(binary_conditions):
        """Convert binary representation into multiclass reprentation:
        For example: convert [[1 1 1 1 0 0 0 0]
                              [0 0 0 0 1 1 1 1]]
        to [1 1 1 1 2 2 2 2]"""
        x,y = np.where(binary_conditions)
        conditions=np.zeros(binary_conditions.shape[1])
        conditions[y]=x+1
        return conditions

    conditions_multi = convert_binary_to_multiclass(conditions)
    conditions_multi[:123]#first run



    # Load in runs
    runs=loadmat('all_subjects_18/runs.mat')
    runs = runs['runs']
    runs = runs[0,:] #get rid of the rows. 



    #ds = fmri_dataset (bold_fname, mask=BA17, targets= conditions_multi, chunks=runs)
    ds = fmri_dataset (bold_fname, mask=VT, targets= conditions_multi, chunks=runs)
    print ds.summary()



    insect1 = ds.targets == 1
    insect2 = ds.targets == 2
    insect3 = ds.targets == 3
    insect4 = ds.targets == 4
    bird1 = ds.targets == 5
    bird2 = ds.targets == 6
    bird3 = ds.targets == 7
    bird4 = ds.targets == 8
    monkey1 = ds.targets == 9
    monkey2 = ds.targets == 10
    monkey3 = ds.targets == 11
    monkey4 = ds.targets == 12


    # In[16]:


    insect1_allTR = ds[insect1,:]
    insect2_allTR = ds[insect2,:]
    insect3_allTR = ds[insect3,:]
    insect4_allTR = ds[insect4,:]
    insect1_mean = np.mean(insect1_allTR,axis=0)
    insect2_mean = np.mean(insect2_allTR,axis=0)
    insect3_mean = np.mean(insect3_allTR,axis=0)
    insect4_mean = np.mean(insect4_allTR,axis=0)

    bird1_allTR = ds[bird1,:]
    bird2_allTR = ds[bird2,:]
    bird3_allTR = ds[bird3,:]
    bird4_allTR = ds[bird4,:]
    bird1_mean = np.mean(bird1_allTR,axis=0)
    bird2_mean = np.mean(bird2_allTR,axis=0)
    bird3_mean = np.mean(bird3_allTR,axis=0)
    bird4_mean = np.mean(bird4_allTR,axis=0)

    monkey1_allTR = ds[monkey1,:]
    monkey2_allTR = ds[monkey2,:]
    monkey3_allTR = ds[monkey3,:]
    monkey4_allTR = ds[monkey4,:]
    monkey1_mean = np.mean(monkey1_allTR,axis=0)
    monkey2_mean = np.mean(monkey2_allTR,axis=0)
    monkey3_mean = np.mean(monkey3_allTR,axis=0)
    monkey4_mean = np.mean(monkey4_allTR,axis=0)



    orig_data = ds.a.mapper.reverse(ds.samples) #data plus spatial information 3d matrix


    #new_data = BA17 * orig_data
    new_data = VT * orig_data


    from matplotlib.pylab import *
    import dtcwt



    # for any monkeys, it's a logical or 
    #monkeys_12 = np.logical_or(monkey1,monkey2)
    #monkeys_34 = np.logical_or(monkey3,monkey4)
    #any_monkeys = np.logical_or(monkeys_12,monkeys_34)


    # In[22]:


    # look at monkey only
    #new_data_monkey = new_data[any_monkeys,:,:,:]
    #new_data_monkey.shape
    #monkeys_mean = new_data_monkey.mean(axis=0)
    #monkeys_mean.shape


    # In[27]:


    var_L1=[]
    var_L2=[]
    var_L3=[]
    var_L4=[]
    var_L5=[]

    trans = dtcwt.Transform3d()

    #animals = [insect1,insect2,insect3,insect4,bird1,bird2,bird3,bird4,monkey1,monkey2,monkey3,monkey4]
    #for i in range(5):
    print "starting wavelet transform"

    for i in range(new_data.shape[0]): #transform all 1230 trs
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



    ALL_Level_TR.shape #1230 TR, 5 levels, each level 28 orientations. 
    #np.save("all_subjs_TRs_levels_log_BA17"+subjects,ALL_Level_TR.data)
    np.save("all_subjs_TRs_levels_log_VT"+subjects,ALL_Level_TR.data)

