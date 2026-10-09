# code to predict daily convective initiation 
# Jawairia Ahmad
# 8/10/2026
# for CEH nowcasting portal

# EA CI model 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
from u_interpolate_small import regrid_irregular_quick
import u_interpolate_small as uint
from pathlib import Path
import glob
import calendar
import os
import math
import time
from netncc_functions import fill_nan_nearest, FSS_loss_gpu, create_mean_filter, get_previous_date
import warnings
import rasterio
import rioxarray
from portal_netpci import netpci
from portal_EA_realtime_dataset import RealtimeDatasetSmLSTA, RealtimeDatasetSm
from portal_EA_realtime_dataset import RealtimeDatasetNoLand
from read_CI_file import read_CI


### user input#########
current_date = '20261007'   ##########read current date 


####### Define domain 
image_height= 1024 #lat
image_width= image_height #lon
out_channels = 1
runtype= 'sm_lsta' #'sm_lsta'  'sm', 'nl'

if runtype== 'sm_lsta': 
    in_channels =10
elif runtype== 'sm':
    in_channels =8
elif runtype== 'nl':
    in_channels =6

# EA grid
resolution = 0.05
end_lat = 15.50  #
start_lon = 3.0 #

###### coords
reg_lat = np.arange(end_lat-resolution*image_height, end_lat, resolution)
reg_lon = np.arange(start_lon, start_lon+resolution*(image_height), resolution) 


#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model 
model = netpci(in_channels)# 
model = nn.DataParallel(model)
model = model.to(device)


# load model
if runtype== 'sm_lsta': 
    modelFile = 'trained_models/EA_all_year_0p05deg_1024_LST_18km_LT_CI_resconv_unet.pth'
    model.load_state_dict(torch.load(modelFile,map_location=torch.device("cpu")))
    #model.eval()

elif runtype== 'sm':
    model.load_state_dict(torch.load(
    'trained_models/EA_all_year_0p05deg_1024_18km_LT_CI_resconv_unet.pth',
        map_location=torch.device("cpu")))
    
elif runtype== 'nl':
    model.load_state_dict(torch.load(
    'trained_models/EA_all_year_0p05deg_1024_noLandInfo_18km_LT_CI_resconv_unet.pth',
        map_location=torch.device("cpu")))


# model input files
# use soil moisture as primary filename
sm_dir = '/mnt/HYDROLOGY/geotiff/ssa_soil_moisture_anomaly/'
filename_sm =  f'{sm_dir}{current_date}/ASCAT_dsm_{current_date}_am_6k_3857.tif'
ifs_dir =  '/prj/swift/ecmwf_windfc/'
filename_wind_100_cape= (ifs_dir+'/'+current_date[0:4]+'/'+current_date[4:6]+'/'+
                         'uv10tcwv_'+current_date+'0000.nc')
filename_wind_700=(ifs_dir+'/'+current_date[0:4]+'/'+current_date[4:6]+'/'+
                         'uv700_'+current_date+'0000.nc')

LST_dir = '/prj/swift/SEVIRI_LST/lsta_ssa/nrt/'
filename_lst=(
    LST_dir+current_date[0:6]+'/'+
    'LSASAF_lst_anom_Daymean_withmask_withHistClim_'+current_date+'_1700.nc')
 

filenames = [
    filename_sm,
    filename_wind_100_cape,
    filename_wind_700,
    filename_lst
]


# read input data
if runtype== 'sm_lsta': 
    inputs = RealtimeDatasetSmLSTA(current_date, filenames, in_channels)
elif runtype== 'sm':
    inputs = RealtimeDatasetSm(current_date, filenames, in_channels)
elif runtype== 'nl':
    inputs = RealtimeDatasetNoLand(current_date, filenames, in_channels)


inputs = inputs.unsqueeze(0)   # Add batch dimension
inputs.to(device)

with torch.no_grad():
    predicted_ci = model(inputs)

predicted_ci=predicted_ci.squeeze()#.cpu()

regridded_ci = read_CI(current_date)


##### write predicted ci to geotiff for display
##### write regridded_ci ci to geotiff for display
# coords are the variables reg_lat, reg_lon


