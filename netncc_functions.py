import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle 
from datetime import date, datetime, timedelta
import xarray as xr
import netCDF4 as nc
import glob
import calendar
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import math
import time
from pathlib import Path
import re
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import uniform_filter

#############################
#if torch.cuda.is_available():
#    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
#else:
#    print("No GPU available. Training will run on CPU.")

device = torch.device('cpu')
############################


###

def calc_local_FSS(forecast, observed, window_size=5):
    """
    Local FSS from probability fields.

    Parameters
    ----------
    forecast : ndarray (time, lat, lon)
    observed : ndarray (time, lat, lon)
    window_size : int
        Neighbourhood size in grid cells.

    Returns
    -------
    local_fss : ndarray (lat, lon)
    """

    # FSS numerator and denominator for each sample
    mse = (forecast - observed)**2
    ref = forecast**2 + observed**2

    # Average over time first
    mse_mean = np.mean(mse, axis=0)
    ref_mean = np.mean(ref, axis=0)

    # Local neighbourhood averages
    mse_local = uniform_filter(
        mse_mean,
        size=window_size,
        mode='nearest'
    )

    ref_local = uniform_filter(
        ref_mean,
        size=window_size,
        mode='nearest'
    )

    local_fss = np.where(
        ref_local > 0,
        1 - mse_local / ref_local,
        np.nan
    )

    return local_fss
    
def reliability_diagram(pred_probs, obs, n_bins=10):
    """
    Plot a reliability diagram from 3D probabilistic prediction and observation arrays.

    Parameters
    ----------
    pred_probs : np.ndarray
        3D array of probabilistic predictions (values between 0 and 1)
    obs : np.ndarray
        3D array of observed outcomes (0/1), same shape as pred_probs
    n_bins : int
        Number of probability bins
    ax : matplotlib axis (optional)
        Axis to draw on. If None, a new figure is created.

    Returns
    -------
    fig, ax : matplotlib Figure and Axis
    """

    # Flatten arrays (shape -> 1D vectors)
    p = pred_probs.ravel()
    o = obs.ravel()

    # Remove NaNs just in case
    valid = ~np.isnan(p) & ~np.isnan(o)
    p = p[valid]
    o = o[valid]

    # Probability bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Store results
    mean_forecast = np.zeros(n_bins)
    mean_observed = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    # Compute stats per bin
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        if np.sum(mask) > 0:
            mean_forecast[i] = np.mean(p[mask])  #mean
            mean_observed[i] = np.mean(o[mask])  #mean
            counts[i] = np.sum(mask)
        else:
            mean_forecast[i] = np.nan
            mean_observed[i] = np.nan

    return mean_forecast, mean_observed, bin_centers, counts

# prob of detetction
def pod_per_gridpoint(truth, detection,threshold):
    hits = np.sum((truth >0.9) & (detection > threshold), axis=0)
    misses = np.sum((truth >0.9) & (detection < threshold), axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        pod = hits / (hits + misses)
        
    return pod # shape = (lat, lon)


def fill_nan_nearest(arr):
    """
    Replace NaNs with the value of the nearest non-NaN pixel.
    """
    
    arr = np.asarray(arr)
    # Mask of NaNs
    nan_mask = np.isnan(arr)
    
    # Get indices of nearest non-NaN cells
    _, indices = distance_transform_edt(
        nan_mask,
        return_distances=True,
        return_indices=True
    )
    # Fill NaNs using nearest valid value
    filled = arr[tuple(indices)]
    return filled


def extract_time_vis(filename):
    """
    Extract datetime from filenames such as:
    VIS_006_rad_20240507_1345.nc
    VIS_006_rad_20240507_1345_eumdat.nc
    """

    pattern = r'_(\d{8})_(\d{4})(?:_eumdat)?\.nc$'
    match = re.search(pattern, Path(filename).name)
    if not match:
        raise ValueError(f"Could not parse timestamp from: {filename}")

    date_str, time_str = match.groups()
    return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")


def select_files(folder, start=1500, end=1600, ext="*.nc",recursive=False):
    folder = Path(folder)
    
    # Match exactly: tile_#### at the end of the filename (before extension)
    rx = re.compile(r"^(?P<date>\d{8})(?P<hhmm>\d{4})$")
    
    #iterator = folder.rglob("*") if recursive else folder.iterdir()

    selected = []
    
    for p in sorted(folder.glob(ext)):
        if not p.is_file():
            continue
        
        match = rx.match(p.stem)   # match without extension
        if not match:
            continue

        hhmm = int(match.group("hhmm"))      

        if start <= hhmm <= end:
            selected.append(p)
    
    return selected
###

# Define FSS loss function
def create_mean_filter(half_num_rows, half_num_columns, num_channels):
    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)
    return torch.full((num_channels, num_channels, num_rows, num_columns), weight, dtype=torch.float32)


def FSS_grid_res_gpu(predicted_tensor, target_tensor):
    actual_mse = torch.mean((torch.from_numpy(target_tensor) - torch.from_numpy(predicted_tensor)) ** 2)
    reference_mse = torch.mean(torch.from_numpy(target_tensor) ** 2 + torch.from_numpy(predicted_tensor) ** 2)
    
    return 1. - actual_mse / reference_mse


def FSS_loss_custom_training_filter(predicted_tensor, target_tensor, half_window_size_px, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse
    
def FSS_loss(predicted_tensor, target_tensor, half_window_size_px=2, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse
    
def FSS_loss_gpu(predicted_tensor, target_tensor, half_window_size_px=2, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse

def FSS_loss_gpu_hws(predicted_tensor, target_tensor, half_window_size_px, use_as_loss_function=True):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse
    
def calc_FSS_pixelwise(forecast, observed):
    actual_mse = np.mean((forecast - observed) ** 2, axis=0)
    reference_mse = np.mean(forecast ** 2 + observed ** 2, axis=0)
    return 1 - (actual_mse / reference_mse)
 

def FSS_accuracy_metric(predicted_tensor, target_tensor, half_window_size_px):
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)    
    actual_mse = torch.mean((smoothed_target_tensor - smoothed_prediction_tensor) ** 2)
    reference_mse = torch.mean(smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2)
    return 1. - actual_mse / reference_mse

def create_mean_filter_window(num_rows, num_columns, num_channels):
    weight = 1.0 / (num_rows * num_columns)
    return torch.full(
        (num_channels, num_channels, num_rows, num_columns),
        weight,
        dtype=torch.float32
    )

def FSS_accuracy_metric_gpu(predicted_tensor, target_tensor, window_size_px):

    # Ensure float32
    predicted_tensor = predicted_tensor.float().to(device)
    target_tensor = target_tensor.float().to(device)

    # Ensure shape is [N, C, H, W]
    if predicted_tensor.ndim == 2:
        predicted_tensor = predicted_tensor.unsqueeze(0).unsqueeze(0)
    elif predicted_tensor.ndim == 3:
        predicted_tensor = predicted_tensor.unsqueeze(1)

    if target_tensor.ndim == 2:
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)
    elif target_tensor.ndim == 3:
        target_tensor = target_tensor.unsqueeze(1)

    weight_matrix = create_mean_filter_window(
        window_size_px,
        window_size_px,
        1
    ).to(device)

    half_window_size_px = window_size_px // 2

    smoothed_target_tensor = F.conv2d(
        target_tensor,
        weight_matrix,
        padding=half_window_size_px
    )

    smoothed_prediction_tensor = F.conv2d(
        predicted_tensor,
        weight_matrix,
        padding=half_window_size_px
    )

    actual_mse = torch.mean(
        (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
    )

    reference_mse = torch.mean(
        smoothed_target_tensor ** 2 +
        smoothed_prediction_tensor ** 2
    )

    return (1.0 - actual_mse / reference_mse).item()
    

def calc_FSS_for_filter_sizes(predicted_tensor,target_tensor):
    # compute FSS at different scales
    half_window_size=[1, 2, 5, 7, 10, 13, 20, 30, 40, 49]
    ffs_array = []
    FSS_eval_kernel=[]
    for FFS_kernal_size in half_window_size:
        FSS_Score = FSS_accuracy_metric_gpu(predicted_tensor.to(device), target_tensor.to(device),FFS_kernal_size)
        ffs_array.append(FSS_Score.to('cpu')) 
        FSS_eval_kernel.append(FFS_kernal_size*2+1)
    return ffs_array, FSS_eval_kernel

def calc_FSS_for_filter_sizes_HOD(predicted_tensor,target_tensor):
    half_window_size=[2, 10, 20, 40]
    ffs_array = []
    FSS_eval_kernel=[]
    for FFS_kernal_size in half_window_size:
        FSS_Score = FSS_accuracy_metric_gpu(predicted_tensor.to(device), target_tensor.to(device),FFS_kernal_size)
        ffs_array.append(FSS_Score.to('cpu')) 
        FSS_eval_kernel.append(FFS_kernal_size*2+1)
    return ffs_array, FSS_eval_kernel

def calc_FSS_for_filter_sizes_roa(predicted_tensor,target_tensor):
    # compute FSS at different scales
    half_window_size=[1, 2, 5, 10]
    ffs_array = []
    FSS_eval_kernel=[]
    for FFS_kernal_size in half_window_size:
        FSS_Score = FSS_accuracy_metric(predicted_tensor, target_tensor,FFS_kernal_size)
        ffs_array.append(FSS_Score.to('cpu')) 
        FSS_eval_kernel.append(FFS_kernal_size*2+1)
    return ffs_array, FSS_eval_kernel


def fss_map_per_batch(pred, target, aux, window, std_threshold, eps=1e-8):
    """
    Compute a local FSS map (center-anchored) between pred and target using a window x window kernel,
    but only for windows where std(aux) > std_threshold.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted event field, shape [N, C, H, W]. Values should be 0/1 (event indicator) or in [0,1].
    target : torch.Tensor
        Target/observed event field, shape [N, C, H, W]. Values should be 0/1 (event indicator) or in [0,1].
    aux : torch.Tensor
        Auxiliary variable used for masking by windowed std, shape [N, C, H, W].
    window : int
        Window size (k x k), e.g., 20.
    std_threshold : float
        Only compute FSS where windowed std(aux) > std_threshold; elsewhere output NaN.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    fss_hw : torch.Tensor
        FSS map of shape [H, W] (e.g., 1024 x 1024), center-anchored.
        NaN where masked out or near borders where the window cannot be centered.
    """

    #if pred.ndim != 2 or target.ndim != 2 or aux.ndim != 2:
    #    raise ValueError("pred, target, aux must be 2D tensors with shape [H, W].")
    if pred.shape != target.shape or pred.shape != aux.shape:
        raise ValueError("pred, target, aux must have the same shape.")

    pred = pred.float()
    target = target.float()
    aux = aux.float()

    N, C, H, W = pred.shape
    k = int(window)

    # Box kernel for windowed mean
    kernel = torch.ones((1, 1, k, k), device=pred.device, dtype=pred.dtype) / (k * k)
    
    # Windowed std(aux): sqrt(E[x^2] - (E[x])^2)
    m1 = F.conv2d(aux, kernel, padding=0) 
    m2 = F.conv2d(aux * aux, kernel, padding=0) 
    std = torch.sqrt((m2 - m1 * m1).clamp_min(0.0) + eps)

    # for batch calculation- 
    f_pred = F.conv2d(pred, kernel, padding=0) 
    f_targ = F.conv2d(target, kernel, padding=0) 

    actual_mse = (f_pred - f_targ) ** 2
    reference_mse = f_pred ** 2 + f_targ ** 2
    fss_valid = 1.0 - (actual_mse / reference_mse)
    
    fss_valid = torch.where(std > std_threshold, fss_valid, torch.full_like(fss_valid, float("nan")))
    fss_valid=torch.nanmean(fss_valid,dim=(0, 1)).squeeze()
    
    
    # ---- Center-anchored padding to [H, W] ----
    # For even k, choose a deterministic "center" convention:
    # anchor window score at offset = k//2 - 1 (upper-left of the central 2x2)
    if k % 2 == 1:
        offset = k // 2
    else:
        offset = k // 2 - 1

    pad_left = offset
    pad_top = offset
    pad_right = (k - 1) - offset
    pad_bottom = (k - 1) - offset

    fss_hw = F.pad(
        fss_valid,
        (pad_left, pad_right, pad_top, pad_bottom),
        value=float("nan")
    )

    # fss_hw should now match original [H, W]
    return fss_hw


def fss_map_per_image(pred, target, aux, window, std_threshold, eps=1e-8):

    if pred.shape != target.shape or pred.shape != aux.shape:
        raise ValueError("pred, target, aux must have the same shape.")

    pred = pred.float()
    target = target.float()
    aux = aux.float()

    N, C, H, W = pred.shape
    k = int(window)

    # Box kernel for windowed mean
    kernel = torch.ones((1, 1, k, k), device=pred.device, dtype=pred.dtype) / (k * k)

    # Windowed std(aux): sqrt(E[x^2] - (E[x])^2)
    m1 = F.conv2d(aux, kernel, padding=0)[0, 0]
    m2 = F.conv2d(aux * aux, kernel, padding=0)[0, 0]
    std = torch.sqrt((m2 - m1 * m1).clamp_min(0.0) + eps)

    # Windowed fractions (valid convolution) -> [H-k+1, W-k+1]
    f_pred = F.conv2d(pred, kernel, padding=0)[0, 0]
    f_targ = F.conv2d(target, kernel, padding=0)[0, 0]

    # Local per-window FSS (valid grid) 
    num = (f_pred - f_targ) ** 2
    den = (f_pred ** 2 + f_targ ** 2 + eps)
    fss_valid = 1.0 - (num / den)

    # Mask: only keep windows with std(aux) > threshold
    fss_valid = torch.where(std > std_threshold, fss_valid, torch.full_like(fss_valid, float("nan")))

    # ---- Center-anchored padding to [H, W] ----
    # For even k, choose a deterministic "center" convention:
    # anchor window score at offset = k//2 - 1 (upper-left of the central 2x2)
    if k % 2 == 1:
        offset = k // 2
    else:
        offset = k // 2 - 1

    pad_left = offset
    pad_top = offset
    pad_right = (k - 1) - offset
    pad_bottom = (k - 1) - offset

    fss_hw = F.pad(
        fss_valid,
        (pad_left, pad_right, pad_top, pad_bottom),
        value=float("nan")
    )

    # fss_hw should now match original [H, W]
    return fss_hw

    
def brier_skill_score(forecast, reference, observed):
    #Calculate Brier skill score (BSS) between forecast and reference.
    bs_forecast = np.round(metrics.brier_score_loss(observed,forecast),3)
    bs_reference = np.round(metrics.brier_score_loss(observed,reference),3)
    if bs_reference<=0.001:
        return 0
    else:
        return 1 - (bs_forecast / bs_reference)
    
def brier_skill_score_mse(forecast, reference, observed):
    #Calculate Brier skill score (BSS) between forecast and reference.
    bs_forecast = np.mean((forecast - observed) ** 2, axis=0)
    bs_reference = np.mean((forecast - reference) ** 2, axis=0)
    return 1 - (bs_forecast / bs_reference)

def get_previous_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Subtract one day using timedelta
    previous_date_obj = date_obj - timedelta(days=1)
    
    # Convert the datetime object back to a string
    previous_date_str = datetime.strftime(previous_date_obj, '%Y-%m-%d')
    
    return previous_date_str

def get_current_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Convert the datetime object back to a string
    date_str = datetime.strftime(date_obj, '%Y-%m-%d')
    
    return previous_date_str

def get_next_date(date_str):
    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    # Subtract one day using timedelta
    next_date_obj = date_obj + timedelta(days=1)
    
    # Convert the datetime object back to a string
    next_date_str = datetime.strftime(next_date_obj, '%Y-%m-%d')
    
    return next_date_str
    
def spatial_filter_conv(predicted_image,half_window_size_px):
    
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix=weight_matrix.to(device)
    #half_window_size_px=half_window_size_px #2
    predicted_image=predicted_image.to(device)
    smoothed_predicted_image = F.conv2d(predicted_image, weight_matrix,padding=half_window_size_px)
    return smoothed_predicted_image

def spatial_filter_conv_cpu(predicted_image,half_window_size_px):
    
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    #weight_matrix=weight_matrix.to(device)
    #half_window_size_px=half_window_size_px #2
    #predicted_image=predicted_image.to(device)
    smoothed_predicted_image = F.conv2d(predicted_image, weight_matrix,padding=half_window_size_px)
    return smoothed_predicted_image


def FSS_loss_gpu_weighted(predicted_tensor, target_tensor, half_window_size_px=2, 
                 use_as_loss_function=True, class1_weight=2.0):
    
    weight_matrix = create_mean_filter(half_window_size_px, half_window_size_px, 1)
    weight_matrix = weight_matrix.to(predicted_tensor.device)

    smoothed_target_tensor = F.conv2d(target_tensor, weight_matrix, padding=half_window_size_px)
    smoothed_prediction_tensor = F.conv2d(predicted_tensor, weight_matrix, padding=half_window_size_px)

    # ---- Class weighting ----
    # weight = 1 for class 0, class1_weight for class 1
    pixel_weights = 1.0 + (class1_weight - 1.0) * target_tensor

    # Apply weights to squared error
    squared_error = (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
    actual_mse = torch.sum(pixel_weights * squared_error) / torch.sum(pixel_weights)

    # Also weight reference term consistently
    reference_term = smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
    reference_mse = torch.sum(pixel_weights * reference_term) / torch.sum(pixel_weights)

    if use_as_loss_function:
        return actual_mse / reference_mse
    return 1. - actual_mse / reference_mse
   