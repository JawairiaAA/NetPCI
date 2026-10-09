# create regridded CI from files stored in dir_conv

import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
from u_interpolate_small import regrid_irregular_quick
import u_interpolate_small as uint
import glob
import os
import time
import warnings
import rasterio
import rioxarray
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from pathlib import Path
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point

# CI
with open('weights/CI_weights_0p05','rb') as file:
    inds_CI, weights_CI, shape_CI= pickle.load(file)

# CI saved as database
dir_conv = '/users/global/cmt/lsta_SSA/initiations/txt_op/NRT/'
var_name = 'CI_min_temp'

# coordinates
res = 0.1 #0.05  # degrees
xmin= -20
ymin= -35
xmax= 55
ymax= 30

nx = int(np.ceil((xmax - xmin) / res))
ny = int(np.ceil((ymax - ymin) / res))

lons_CI = xmin + res/2 + np.arange(nx) * res
lats_CI = ymax - res/2 - np.arange(ny) * res  # raster row order

tf = TimezoneFinder()


def convert_row(row):
    tz = tf.timezone_at(lat=row["Latitude"], lng=row["Longitude"])

    if not tz:
        return pd.NaT

    utc_dt = pd.to_datetime(row["utc_time"], utc=True)

    # Convert to local timezone and then remove timezone info
    return utc_dt.tz_convert(ZoneInfo(tz)).tz_localize(None)

            
def read_CI(current_date):
    filename = 'init_'+current_date+'_T40_vn4_18km.txt'
    #print(filename)

    filtered_df = pd.read_csv(dir_conv+filename, sep=r'\s+')
    
    data = {
        'Longitude':  filtered_df['lonE_c'], 
        'Latitude': filtered_df['latN_c'], 
        'mo_of_conv': filtered_df['mo'],
        'dy_of_conv': filtered_df['dy'], 'Hr_of_conv': filtered_df['hr'],
        'min_of_conv': filtered_df['mn'],'min_temp' : filtered_df['minT']}

    geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=geometry)
    
    # Create datetime column
    gdf["date"] = pd.to_datetime({
        "year": current_date[0:4],  # replace if year column exists
        "month": gdf["mo_of_conv"],
        "day": gdf["dy_of_conv"],
    })
    
    gdf["utc_time"] = pd.to_datetime({
        "year": current_date[0:4],  # replace if year column exists
        "month": gdf["mo_of_conv"],
        "day": gdf["dy_of_conv"],
        "hour": gdf["Hr_of_conv"],
        "minute": gdf["min_of_conv"]
    })
    
    gdf["local_time"] = gdf.apply(convert_row, axis=1)
          
    filtered_gdf = gdf[gdf["local_time"].dt.hour >= 12]
    #filtered_gdf = filtered_gdf[filtered_gdf["local_time"].dt.hour < 18]

    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    
    # Unique timesteps
    times = np.sort(filtered_gdf["date"].unique())
            
    # Rasterize each timestep
    for t_idx, t in enumerate(times):

        # Rasterize each timestep
        subset = filtered_gdf[filtered_gdf["date"] == t]
    
        data = rasterize(
            (
                (geom, value)
                for geom, value in zip(
                    subset.geometry,
                    subset["min_temp"]
                )
            ),
            out_shape=(ny, nx),
            transform=transform,
            fill=np.nan,
            dtype="float32"
        )
    ci_temp = data
    ci_temp[np.where(np.abs(data)>0)]=1
    ci_temp[np.isnan(ci_temp)] = 0
    
    regridded_ci = uint.interpolate_data(ci_temp, inds_CI, weights_CI, shape_CI)  #
    regridded_ci[np.where(regridded_ci>0)] = 1
    regridded_ci[np.isnan(regridded_ci)] = 0

    return regridded_ci
