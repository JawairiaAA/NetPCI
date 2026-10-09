# WA CI model training

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
from u_interpolate_small import regrid_irregular_quick
import u_interpolate_small as uint
import glob
import calendar
import os
import math
import time
from netncc_functions import fill_nan_nearest, FSS_loss_gpu, create_mean_filter, get_previous_date
import warnings
import rasterio
import rioxarray



device = torch.device('cpu')
image_height= 1024 #lat
image_width= image_height #lon
out_channels = 1

# grid
resolution = 0.05
end_lat = 15.50  #25
start_lon = 3.0 #
reg_lat = np.arange(end_lat-resolution*image_height, end_lat, resolution)
reg_lon = np.arange(start_lon, start_lon+resolution*(image_height), resolution) 



with open('weights/sma_weights_0p05','rb') as file:
    inds_sm, weights_sm, shape_sm= pickle.load(file)

with open('weights/sm_clim_weights_0p05','rb') as file:
    inds_sm_clim, weights_sm_clim, shape_sm_clim= pickle.load(file)

# lsta
with open('weights/lsta_weights_0p05','rb') as file:
    inds_lst, weights_lst, shape_lst= pickle.load(file)

# wind
with open('weights/wind100m_weights_0p05','rb') as file:
    inds_wind_100, weights_wind_100, shape_wind_100= pickle.load(file)

with open('weights/wind700hPa_weights_0p05','rb') as file:
    inds_wind_650, weights_wind_650, shape_wind_650= pickle.load(file)

# cape- same as 100 wind
reg_elev = xr.open_dataset("elevation_EA_0p05_"+str(image_height)+".nc")
elevation = reg_elev.elevation.values
elevation[np.where(elevation<0)]=0

#sm clim
dir_clim = '/prj/swift/ASCAT_H122/ancil/SSA_grid_01_vn2H/clim/'

nx, ny, nt = 730, 602, 12
sm_clim_tmp = 0.0001*np.fromfile(dir_clim+'clim_2007_2024_am.gra', dtype=np.int16)
sm_clim_tmp = sm_clim_tmp.reshape((2,nt,ny,nx))
sm_clim_tmp = sm_clim_tmp[0,:,:,:].squeeze()


# Dataset class using xarray
def RealtimeDatasetSmLSTA(current_date, filenames,in_channels):
    
           
    x_train = np.zeros((in_channels,image_height,image_width),dtype=float) 

    # sm
    da = rioxarray.open_rasterio(filenames[0])
    da = da.where(da >= -998)
    # Reproject to lat/lon
    da_ll = da.rio.reproject("EPSG:4326")

    sm_orig = np.flip(0.01*da_ll.squeeze().values,axis=0)
    ind_nan = np.where(np.isnan(sm_orig))
    sm_orig[ind_nan] = 0
    sm_mask = sm_orig*0
    sm_mask[np.where(np.abs(sm_orig)>0)]=1
    
    regridded_sm = uint.interpolate_data(sm_orig, inds_sm, weights_sm, shape_sm)  
    regridded_sm_mask = uint.interpolate_data(sm_mask, inds_sm, weights_sm, shape_sm)  

    regridded_sm[np.isnan(regridded_sm)]=0
    regridded_sm_mask[np.where(regridded_sm_mask<=0.9)]=0
    regridded_sm_mask[np.where(regridded_sm_mask>0.9)]=1
    regridded_sm_mask[np.isnan(regridded_sm_mask)]=0


    sm_clim_am = sm_clim_tmp[[int(current_date[4:6])-1],:,:].squeeze()
    regridded_sm_clim_am = uint.interpolate_data(sm_clim_am, 
                                                 inds_sm_clim, weights_sm_clim, shape_sm_clim)  # 
    regridded_sm_clim_am[np.where(regridded_sm_clim_am<0)]=0
    regridded_sm_clim_am[np.isnan(regridded_sm_clim_am)]=0
    sm = (regridded_sm+regridded_sm_clim_am)*regridded_sm_mask
    sm[np.where(regridded_sm_mask<=0.9)]=0 
    sm[np.where(sm<0)]=0
    sm[np.where(sm>1)]=0
    
    x_train[0,:,:] = sm
    x_train[1,:,:] = regridded_sm_mask

    # lsta
    lsta_file = filenames[3]
    ds = xr.open_dataset(lsta_file)
    lsta = ds.lst_anom_dailymean.values  #/100
    lsta_mask = lsta*0
    lsta_mask[np.where(lsta>-20)]= 1 # includes -9999
    lsta[np.where(lsta<-20)]= 0 #
    lsta_mask[np.isnan(lsta)]=0
    lsta[np.isnan(lsta)]=0

    regridded_lsta = uint.interpolate_data(lsta, inds_lst, weights_lst, shape_lst)  
    regridded_lsta_mask = uint.interpolate_data(lsta_mask, inds_lst, weights_lst, shape_lst)  
    regridded_lsta_mask[np.where(regridded_lsta_mask<=0.9)]=0
    regridded_lsta_mask[np.where(regridded_lsta_mask>0.9)]=1
    regridded_lsta_mask[np.isnan(regridded_lsta_mask)]=0
    regridded_lsta[np.isnan(regridded_lsta)]=0

    # else:
    #     regridded_lsta = regridded_sm*0 
    #     regridded_lsta_mask = regridded_sm*0 
        
    x_train[2,:,:] = regridded_lsta/20
    x_train[3,:,:] = regridded_lsta_mask
    
    # wind u 100
    wind_100 = xr.open_dataset(filenames[1]) 
    regridded_wind_u100 = uint.interpolate_data(wind_100.u100.values[2,:,:], 
                                                inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_wind_v100 = uint.interpolate_data(wind_100.v100.values[2,:,:], 
                                                inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_wind_u100 = fill_nan_nearest(regridded_wind_u100)
    regridded_wind_v100 = fill_nan_nearest(regridded_wind_v100)
    #regridded_wind_v100[np.where(reg_lat>25)] = 0

    # u 650
    wind_650 = xr.open_dataset(filenames[2])
    regridded_wind_u650 = uint.interpolate_data(wind_650.u.values[2,:,:], 
                                                inds_wind_650, weights_wind_650, shape_wind_650)  
    regridded_wind_v650 = uint.interpolate_data(wind_650.v.values[2,:,:], 
                                                inds_wind_650, weights_wind_650, shape_wind_650)  
    regridded_wind_u650 = fill_nan_nearest(regridded_wind_u650)
    #regridded_wind_u650[np.where(reg_lat>25)] = 0
    regridded_wind_v650 = fill_nan_nearest(regridded_wind_v650)

    x_train[4,:,:] = regridded_wind_u100/40
    x_train[5,:,:] = regridded_wind_v100/40
    x_train[6,:,:] = regridded_wind_u650/50
    x_train[7,:,:] = regridded_wind_v650/50

    # cape
    cape = xr.open_dataset(filenames[1])
    regridded_cape = uint.interpolate_data(wind_100.mucape.values[2,:,:], 
                                           inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_cape[np.isnan(regridded_cape)] = 0
    x_train[8,:,:] = regridded_cape/12000

    # elevation
    elevation[np.isnan(elevation)] = 0
    x_train[9,:,:] = elevation/5000 # threshold 4000
    
    # # open and read files- cores output
    # target_file = self.mask_files[idx]
    # #print(target_file)
    
    # target_CI = xr.open_dataset(target_file).squeeze() 
    # ci_temp = target_CI['CI_min_temp'].values
    # ci_temp[np.where(np.abs(ci_temp)>0)]=1
    # ci_temp[np.isnan(ci_temp)] = 0
    
    # regridded_ci = uint.interpolate_data(ci_temp, inds_CI, weights_CI, shape_CI)  #
    # regridded_ci[np.where(regridded_ci>0)] = 1
    # regridded_ci[np.isnan(regridded_ci)] = 0
    # y_train=np.expand_dims(regridded_ci, axis=0)
    #print(y_train.shape)

    return torch.tensor(x_train.astype(np.float32))


# Dataset class using xarray
def RealtimeDatasetSm(current_date, filenames,in_channels):
    
           
    x_train = np.zeros((in_channels,image_height,image_width),dtype=float) 

    # sm
    da = rioxarray.open_rasterio(filenames[0])
    da = da.where(da >= -998)
    # Reproject to lat/lon
    da_ll = da.rio.reproject("EPSG:4326")

    sm_orig = np.flip(0.01*da_ll.squeeze().values,axis=0)
    ind_nan = np.where(np.isnan(sm_orig))
    sm_orig[ind_nan] = 0
    sm_mask = sm_orig*0
    sm_mask[np.where(np.abs(sm_orig)>0)]=1
    
    regridded_sm = uint.interpolate_data(sm_orig, inds_sm, weights_sm, shape_sm)  
    regridded_sm_mask = uint.interpolate_data(sm_mask, inds_sm, weights_sm, shape_sm)  

    regridded_sm[np.isnan(regridded_sm)]=0
    regridded_sm_mask[np.where(regridded_sm_mask<=0.9)]=0
    regridded_sm_mask[np.where(regridded_sm_mask>0.9)]=1
    regridded_sm_mask[np.isnan(regridded_sm_mask)]=0


    sm_clim_am = sm_clim_tmp[[int(current_date[4:6])-1],:,:].squeeze()
    regridded_sm_clim_am = uint.interpolate_data(sm_clim_am, 
                                                 inds_sm_clim, weights_sm_clim, shape_sm_clim)  # 
    regridded_sm_clim_am[np.where(regridded_sm_clim_am<0)]=0
    regridded_sm_clim_am[np.isnan(regridded_sm_clim_am)]=0
    sm = (regridded_sm+regridded_sm_clim_am)*regridded_sm_mask
    sm[np.where(regridded_sm_mask<=0.9)]=0 
    sm[np.where(sm<0)]=0
    sm[np.where(sm>1)]=0
    
    x_train[0,:,:] = sm
    x_train[1,:,:] = regridded_sm_mask
    
    # wind u 100
    wind_100 = xr.open_dataset(filenames[1]) 
    regridded_wind_u100 = uint.interpolate_data(wind_100.u100.values[2,:,:], 
                                                inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_wind_v100 = uint.interpolate_data(wind_100.v100.values[2,:,:], 
                                                inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_wind_u100 = fill_nan_nearest(regridded_wind_u100)
    regridded_wind_v100 = fill_nan_nearest(regridded_wind_v100)
    #regridded_wind_v100[np.where(reg_lat>25)] = 0

    # u 650
    wind_650 = xr.open_dataset(filenames[2])
    regridded_wind_u650 = uint.interpolate_data(wind_650.u.values[2,:,:], 
                                                inds_wind_650, weights_wind_650, shape_wind_650)  
    regridded_wind_v650 = uint.interpolate_data(wind_650.v.values[2,:,:], 
                                                inds_wind_650, weights_wind_650, shape_wind_650)  
    regridded_wind_u650 = fill_nan_nearest(regridded_wind_u650)
    #regridded_wind_u650[np.where(reg_lat>25)] = 0
    regridded_wind_v650 = fill_nan_nearest(regridded_wind_v650)

    x_train[2,:,:] = regridded_wind_u100/40
    x_train[3,:,:] = regridded_wind_v100/40
    x_train[4,:,:] = regridded_wind_u650/50
    x_train[5,:,:] = regridded_wind_v650/50

    # cape
    cape = xr.open_dataset(filenames[1])
    regridded_cape = uint.interpolate_data(wind_100.mucape.values[2,:,:], 
                                           inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_cape[np.isnan(regridded_cape)] = 0
    x_train[6,:,:] = regridded_cape/12000

    # elevation
    elevation[np.isnan(elevation)] = 0
    x_train[7,:,:] = elevation/5000 # threshold 4000
 
    return torch.tensor(x_train.astype(np.float32))

# Dataset class using xarray
def RealtimeDatasetNoLand(current_date, filenames,in_channels):
    
           
    x_train = np.zeros((in_channels,image_height,image_width),dtype=float) 
    
    # wind u 100
    wind_100 = xr.open_dataset(filenames[1]) 
    regridded_wind_u100 = uint.interpolate_data(wind_100.u100.values[2,:,:], 
                                                inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_wind_v100 = uint.interpolate_data(wind_100.v100.values[2,:,:], 
                                                inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_wind_u100 = fill_nan_nearest(regridded_wind_u100)
    regridded_wind_v100 = fill_nan_nearest(regridded_wind_v100)
    #regridded_wind_v100[np.where(reg_lat>25)] = 0

    # u 650
    wind_650 = xr.open_dataset(filenames[2])
    regridded_wind_u650 = uint.interpolate_data(wind_650.u.values[2,:,:], 
                                                inds_wind_650, weights_wind_650, shape_wind_650)  
    regridded_wind_v650 = uint.interpolate_data(wind_650.v.values[2,:,:], 
                                                inds_wind_650, weights_wind_650, shape_wind_650)  
    regridded_wind_u650 = fill_nan_nearest(regridded_wind_u650)
    #regridded_wind_u650[np.where(reg_lat>25)] = 0
    regridded_wind_v650 = fill_nan_nearest(regridded_wind_v650)

    x_train[0,:,:] = regridded_wind_u100/40
    x_train[1,:,:] = regridded_wind_v100/40
    x_train[2,:,:] = regridded_wind_u650/50
    x_train[3,:,:] = regridded_wind_v650/50

    # cape
    cape = xr.open_dataset(filenames[1])
    regridded_cape = uint.interpolate_data(wind_100.mucape.values[2,:,:], 
                                           inds_wind_100, weights_wind_100, shape_wind_100)  
    regridded_cape[np.isnan(regridded_cape)] = 0
    x_train[4,:,:] = regridded_cape/12000

    # elevation
    elevation[np.isnan(elevation)] = 0
    x_train[5,:,:] = elevation/5000 # threshold 4000
 
    return torch.tensor(x_train.astype(np.float32))