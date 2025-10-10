# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:48:00 2024

@author: mkoolani
"""

#%% Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import cv2
import math
from scipy.optimize import fsolve, brentq
from shapely.geometry import Polygon
from bimvee.importAe import importAe
from bimvee.split import cropTime
from bimvee.plot import plot
from bimvee.player import Player
from bimvee.plot import plotEventRate
from bimvee.plot import plotDvsContrast
from bimvee.importHdf5 import importHdf5
from sklearn.neighbors import KernelDensity
from bimvee.split import selectByBool

#%% ---- Function Definitions ----

"""
Accumulate all events from a DVS camera into a single frame.

Parameters:
- events: dict, contains DVS event data with keys:
    - 'x': np.array, x-coordinates of events
    - 'y': np.array, y-coordinates of events
- width: int, width of the frame (in pixels).
- height: int, height of the frame (in pixels).

Returns:
- image: np.array, 2D array representing the accumulated frame, where each pixel is set to 
         255 if it contains one or more events, and 0 otherwise.
"""

def accumulate_events(events, width, height):
    image = np.zeros((height, width), dtype=np.uint8)
    x = events['x'].astype(int)
    y = events['y'].astype(int)
    valid_indices = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid_indices]
    y = y[valid_indices]
    image[y, x] = 255  
    return image

#%% 
"""
Create frames from DVS (Dynamic Vision Sensor) event data by binning events over time.

Parameters:
- events: dict, contains DVS event data with keys 'ts' (timestamps), 'x' (x-coordinates), 
          'y' (y-coordinates), and 'pol' (polarity).
- stimulus_duration: float, total duration of stimulus in seconds. Only events within this 
                     time frame are used.
- polarity_filtering: bool, whether to filter events based on polarity. If True, only positive 
                      polarity events are considered.
- dt_bin: float, time bin size in seconds for grouping events into frames.

Returns:
- binned_frames: np.array, 3D array of size (num_bins, height, width) for polarity filtering, 
                 or (num_bins, 2, height, width) if polarity filtering is False.
                 Each frame contains the accumulated event counts for the corresponding time bin.
"""
def create_frames(events, stimulus_duration, polarity_filtering, dt_bin):
    e_data = events
    mask = e_data['ts'] <= stimulus_duration
 
    e_ts = e_data['ts'][mask]
    e_x = e_data['x'][mask]
    e_y = e_data['y'][mask]
    e_pol = e_data['pol'][mask]
 
    if polarity_filtering:
        pol_mask = e_pol == True
        e_ts = e_ts[pol_mask]
        e_x = e_x[pol_mask]
        e_y = e_y[pol_mask]
 
    img_size = (e_y.max() + 1, e_x.max() + 1)
    t_min, t_max = e_ts.min(), e_ts.max()
    num_bins = int((t_max - t_min) / dt_bin)
 
    if polarity_filtering:
        binned_frames = np.zeros((num_bins,) + img_size, dtype=int)
        idx_t = ((e_ts - t_min) / dt_bin).astype(int)
        idx_t = np.clip(idx_t, 0, num_bins - 1)
        np.add.at(binned_frames, (idx_t, e_y, e_x), 1)
    else:
        binned_frames = np.zeros((num_bins, 2) + img_size, dtype=int)
        idx_t = ((e_ts - t_min) / dt_bin).astype(int)
        idx_t = np.clip(idx_t, 0, num_bins - 1)
        np.add.at(binned_frames, (idx_t, e_pol, e_y, e_x), 1)
 
    return binned_frames
    

#%%
"""
Undistort event points directly from a DVS camera container, 
verify the results, and visualize them.

Parameters:
- container: dict, event data from the DVS camera
- camera_matrix: np.array, intrinsic camera matrix
- dist_coeffs: np.array, distortion coefficients
- width: int, width of the sensor
- height: int, height of the sensor

Returns:
- x_undistorted: np.array, undistorted x coordinates
- y_undistorted: np.array, undistorted y coordinates
"""
def undistort_event_points_standard(container, camera_matrix, dist_coeffs, width, height):

    # Step 1: Extract x and y coordinates from the container
    x = container['x']
    y = container['y']

    # Step 2: Apply undistortion
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError("The number of x and y coordinates must be the same.")
    
    points = np.vstack((x, y)).T.reshape(-1, 1, 2)
    
    undistorted_points = cv2.undistortPoints(
        points,
        camera_matrix,
        dist_coeffs,
        P=camera_matrix
    )
    
    x_undistorted = undistorted_points[:, 0, 0]
    y_undistorted = undistorted_points[:, 0, 1]
    
    x_undistorted = x_undistorted.astype(np.int64)
    y_undistorted = y_undistorted.astype(np.int64)
    
    # Step 3: Filter valid undistorted coordinates within the sensor bounds
    valid_indices = (
        (x_undistorted >= 0) & (x_undistorted < width) &
        (y_undistorted >= 0) & (y_undistorted < height)
    )
    x_undistorted = x_undistorted[valid_indices]
    y_undistorted = y_undistorted[valid_indices]

    # Step 4: Accumulate and visualize events before and after undistortion
    plt.figure()
    image_before = accumulate_events(container, width, height)
    plt.imshow(image_before, cmap='gray')
    plt.title('Before Undistortion')

    # Prepare undistorted event container for visualization
    undistorted_events = {
        'x': x_undistorted,
        'y': y_undistorted,
        'ts': container['ts'][valid_indices],
        'pol': container['pol'][valid_indices],
    }

    plt.figure()
    image_after = accumulate_events(undistorted_events, width, height)
    plt.imshow(image_after, cmap='gray')
    plt.title('After Undistortion')

    plt.show()
    
    container_undistorted = undistorted_events
    return container_undistorted

#%% Feild of view calculation
"""
Calculate the new field of view (FoV) in both x and y directions.

Parameters:
- camera_matrix: np.array, intrinsic camera matrix
- sensor_width: int, width of the sensor in pixels
- sensor_height: int, height of the sensor in pixels

Returns:
- fov_x_new: int, new horizontal FoV in degrees
- fov_y_new: int, new vertical FoV in degrees
"""
def calculate_new_fov(camera_matrix, sensor_width, sensor_height):

    # Extract focal lengths from the camera matrix
    f_x = camera_matrix[0, 0]  # Focal length in x-direction
    f_y = camera_matrix[1, 1]  # Focal length in y-direction

    # Calculate new FoV
    fov_x_new = int(2 * np.degrees(np.arctan(sensor_width / (2 * f_x))))
    fov_y_new = int(2 * np.degrees(np.arctan(sensor_height / (2 * f_y))))

    return fov_x_new, fov_y_new


#%%
"""
Calculate the start and end times for each press during multiple repetitions.

Parameters:
- first_press_time: float, the time (in seconds) when the first press occurs.
- num_repetitions: int, the total number of repetitions.
- num_presses_per_repetition: int, the number of presses in each repetition.
- press_duration: float, the duration of each press (in seconds).
- delay_between_repetitions: float, the time gap (in seconds) between successive repetitions.

Returns:
- press_timings: list of lists, where each sublist contains the start times of all presses 
                 for a particular repetition.
"""
def calculate_press_timings(first_press_time, num_repetitions, num_presses_per_repetition, press_duration, delay_between_repetitions):

    press_timings = []
    for repetition in range(num_repetitions):
        repetition_press_timings = []
        # Calculate the start time of the current repetition
        repetition_start_time = first_press_time + repetition * (num_presses_per_repetition * press_duration + delay_between_repetitions)
        
        for press_index in range(num_presses_per_repetition):
            # Use an exact formula for each press timing
            press_time = repetition_start_time + press_index * press_duration
            repetition_press_timings.append(press_time)
        
        press_timings.append(repetition_press_timings)
    return press_timings
#%% Function to Split Data by Specific Times for Multiple Repetitions
"""
Split the event data into individual presses based on specified timings for multiple repetitions.

Parameters:
- container: dict, the main data container with events (e.g., timestamps, x/y coordinates).
- press_timings_per_repetition: list of lists, each sublist contains the start times of presses 
                                 for a specific repetition.
- press_duration: float, the duration of each press in seconds.

Returns:
- containers_by_press: dict, where each key is a repetition (e.g., 'repetition_1'), and its value 
                       is a list of cropped data containers for each press in that repetition.
"""
def split_data_by_specific_times_multiple_repetitions_aligned(container, press_timings_per_repetition, press_duration):

    containers_by_press = {}
    # timingdic = {}
    half_duration = press_duration / 2  # To center the split data around the press

    for repetition_index, press_timings in enumerate(press_timings_per_repetition):
        repetition_key = f'repetition_{repetition_index + 1}'
        containers_by_press[repetition_key] = []
        
        for press_time in press_timings:
            # Center the split data on the press time
            start_time = press_time - half_duration
            end_time = press_time + half_duration
            # print(start_time)
            # timingdic[repetition_key].append(timingtest)
            press_container = cropTime(container, startTime=start_time, stopTime=end_time)
            containers_by_press[repetition_key].append(press_container)
    
    return containers_by_press

#%% Function to crop data for each press 
"""
Function to crop the data for each press within a specified time window.

Parameters:
    containers_by_press_list (dict): A dictionary where each key is a repetition identifier 
                                     and the value is a list of containers for each press.
    crop_start (float): The start time (in seconds) relative to the press timestamp to begin cropping.
    crop_end (float): The end time (in seconds) relative to the press timestamp to end cropping.

Returns:
    dict: A dictionary with the same structure as `containers_by_press_list`, where each press
          container has been cropped based on the provided time window (crop_start to crop_end).
          
The function works by iterating through each repetition and its associated press containers. 
For each press, it calculates the start and end times for cropping, using the press timestamp 
as a reference. The `cropTime` function is then called to extract the relevant data within the 
specified time window for each press.
"""
def crop_presses_for_specific_time(containers_by_press_list, crop_start, crop_end):
    cropped_containers = {}

    for repetition_key, containers in containers_by_press_list.items():
        cropped_containers[repetition_key] = []
        
        for press_container in containers:
            press_time = press_container['ts'][0]  # Get the timestamp of the first event (assuming it's for the press)
            
            # Calculate the start and end times for the desired crop window
            start_time = press_time + crop_start
            end_time = press_time + crop_end

            # Use cropTime to crop the data
            cropped_container = cropTime(press_container, startTime=start_time, stopTime=end_time)
            cropped_containers[repetition_key].append(cropped_container)

    return cropped_containers

#%%
def crop_specific_presses(kept_presses, containers_by_press_list, crop_start, crop_end):

    cropped_containers = {}

    for repetition_key, containers in containers_by_press_list.items():
        cropped_containers[repetition_key] = []

        for press_idx, press_container in enumerate(containers):
            if press_idx not in kept_presses:
                cropped_containers[repetition_key].append(None)  # Skip this press
                continue

            # Get the timestamp of the first event
            press_time = press_container['ts'][0]

            # Calculate start and end times for cropping
            start_time = press_time + crop_start
            end_time = press_time + crop_end

            # Use cropTime to crop the data
            cropped_container = cropTime(press_container, startTime=start_time, stopTime=end_time)
            cropped_containers[repetition_key].append(cropped_container)

    return cropped_containers

#%%
from bimvee.plot import plotEventRate

"""
Plots the event rate for all presses in a given repetition with colored lines.

Args:
    containers (dict): A dictionary containing event data split by repetition.
    repetition_key (str): The key for the desired repetition (e.g., 'repetition_10').
    cmap_name (str): The name of the colormap to use for line colors. Default is 'winter'.
    ylim (tuple): The y-axis limits for the plot. Default is (0, 100000).
    title (str): The title for the plot. Default is "Event Rate Plot".

Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
"""
def plot_event_rate_by_repetition(containers, repetition_key, cmap_name='winter', ylim=(0, 100000), title="Event Rate Plot"):

    # Get colormap and initialize plot
    cmap = plt.get_cmap(cmap_name)
    num_colours = cmap.N
    plt.close('all')
    fig, ax = plt.subplots()

    # Plot event rate for each press in the given repetition
    for cont in containers[repetition_key]:
        plotEventRate(cont, axes=ax, periods=[0.1])
        plt.ylim(*ylim)

    # Adjust line colors
    lines = ax.get_lines()
    num_lines = len(lines)
    for line_idx, line in enumerate(lines):
        line.set_color(cmap(int(line_idx / (num_lines - 1) * (num_colours - 1))))

    # Set title
    plt.title(title, fontsize=40)

    return fig


#%% Function to iteratively reduce events by a given factor
"""
Function to randomly reduce the number of events in each press container based on a given reduction factor.

Parameters:
    containers_by_press (dict): A dictionary where each key is a repetition identifier and the value is a list of press containers.
    reduction_factor (int): The factor by which to reduce the number of events. 
    For example, a reduction factor of 2 will keep half of the events, 
    while a factor of 3 will keep one-third of the events.

Returns:
    dict: A dictionary with the same structure as `containers_by_press`, but with the events reduced based on the specified reduction factor.

The function works by iterating through each repetition and its associated press containers. 
For each press container, it calculates the number of events to keep based on the reduction factor. 
It then randomly selects a subset of the events to keep, using a boolean mask to identify the selected events. 
The `selectByBool` function is used to apply the mask and reduce the events. If a press container has no events, it is appended without modification.
"""
def reduce_events(containers_by_press, reduction_factor):
    reduced_containers_by_press = {}

    for repetition_key, presses in containers_by_press.items():
        reduced_containers_by_press[repetition_key] = []

        for press_container in presses:
            if isinstance(press_container, dict):
                num_events = len(press_container['ts'])
                if num_events == 0:
                    reduced_containers_by_press[repetition_key].append(press_container)
                    continue

                # Calculate the number of events to keep based on the reduction factor
                num_events_to_keep = max(1, num_events // reduction_factor)

                # Randomly select indices to keep
                boolean_mask = np.zeros(num_events, dtype=bool)
                true_indices = np.random.choice(num_events, size=num_events_to_keep, replace=False)
                boolean_mask[true_indices] = True

                # Reduce the events
                reduced_container = selectByBool(press_container, boolean_mask)
                reduced_containers_by_press[repetition_key].append(reduced_container)
            else:
                reduced_containers_by_press[repetition_key].append(press_container)

    return reduced_containers_by_press


#%%

def reduce_events_seed(containers_by_press, reduction_factor):
    reduced_containers_by_press = {}
    
    # Ensure different randomness each time
    rng = np.random.default_rng()  # Uses a new random generator instance

    for repetition_key, presses in containers_by_press.items():
        reduced_containers_by_press[repetition_key] = []

        for press_container in presses:
            if isinstance(press_container, dict):
                num_events = len(press_container['ts'])
                if num_events == 0:
                    reduced_containers_by_press[repetition_key].append(press_container)
                    continue

                # Calculate the number of events to keep based on the reduction factor
                num_events_to_keep = max(1, num_events // reduction_factor)

                # Randomly select indices to keep
                boolean_mask = np.zeros(num_events, dtype=bool)
                true_indices = rng.choice(num_events, size=num_events_to_keep, replace=False)  # Uses new generator
                boolean_mask[true_indices] = True

                # Reduce the events
                reduced_container = selectByBool(press_container, boolean_mask)
                reduced_containers_by_press[repetition_key].append(reduced_container)
            else:
                reduced_containers_by_press[repetition_key].append(press_container)

    return reduced_containers_by_press

#%%
"""
Function to calculate the histogram of x-coordinates from the container data, binned by specified column intervals.

Parameters:
    container (dict): A dictionary containing the 'x' coordinate data for which the histogram is calculated.
    params (dict): A dictionary containing parameters for the histogram calculation:
        - 'num_cols' (int): The total number of columns (bins) in the x-axis range.
        - 'cols_per_bin' (int): The number of columns per bin in the histogram.

Returns:
    hist (ndarray): An array containing the histogram counts for each bin.
    
The function works by extracting the 'x' coordinate data from the container and then calculates 
a histogram based on the bin edges defined by `cols_per_bin` and `num_cols`. The `np.histogram` 
function is used to bin the x-data into the specified number of bins, and it returns the histogram 
count for each bin.
"""
def hist_y_rect(container, params):
    num_cols = params['num_cols']
    cols_per_bin = params['cols_per_bin']
    x_data = container['x']
    hist, bins = np.histogram(x_data, bins=np.arange(-0.5, num_cols, cols_per_bin))
    return hist

#%%
"""
Function to compute histograms for all presses in all repetitions and store them in a dictionary.

Parameters:
    containers_by_press (dict): A dictionary where each key is a repetition identifier, and the value is a list of press containers.
    params (dict): A dictionary containing parameters for the histogram calculation:
        - 'num_cols' (int): The total number of columns (bins) in the x-axis range.
        - 'cols_per_bin' (int): The number of columns per bin in the histogram.

Returns:
    histograms_by_repetition (dict): A dictionary where each key is a repetition identifier, and the value is a numpy array 
                                     containing histograms for each press in that repetition.
    
The function works by iterating over each repetition and each press within that repetition. 
For each press container, it calls the `hist_y_rect` function to compute the histogram of the 'x' data. 
If the press has no events (i.e., an empty 'ts' array), a zero-filled histogram is added instead. 
The computed histograms for each press are stored in an array, which is then stored in the dictionary 
`histograms_by_repetition`, with each repetition's histograms grouped together.
"""
def compute_histograms_all_repetitions_dict(containers_by_press, params):

    histograms_by_repetition = {}  # Dictionary to store histograms for all repetitions
    
    for repetition_key, presses in containers_by_press.items():
        histograms = []
        for press in presses:
            if press['ts'].size == 0:  # Skip if 'ts' is empty (removed press)
                histograms.append(np.zeros(params['num_cols'] // params['cols_per_bin']))  # Add a zero-filled histogram
                continue
            histograms.append(hist_y_rect(press, params))
        histograms_by_repetition[repetition_key] = np.array(histograms)  # Store histograms for the repetition as an array
    
    return histograms_by_repetition

#%%

"""
Function to compute histograms for specific presses in all repetitions, skipping invalid presses.

Parameters:
    containers_by_press (dict): A dictionary where each key is a repetition identifier, and the value is a list of press containers.
    params (dict): A dictionary containing parameters for the histogram calculation:
        - 'num_cols' (int): The total number of columns (bins) in the x-axis range.
        - 'cols_per_bin' (int): The number of columns per bin in the histogram.

Returns:
    histograms_by_repetition (dict): A dictionary where each key is a repetition identifier, and the value is a numpy array 
                                     containing histograms for each valid press in that repetition.

The function works by iterating over each repetition and each press within that repetition. 
For each press container, it checks if the press is valid (i.e., it is not `None`, contains a 'ts' field, 
and has events). If the press is invalid or has no events, a zero-filled histogram is added instead. 
The valid histograms are computed using the `hist_y_rect` function and stored in an array for each repetition. 
The histograms for each repetition are then grouped together and stored in the dictionary `histograms_by_repetition`.
"""
def compute_histograms_all_repetitions_dict_with_specific_presses(containers_by_press, params):

    histograms_by_repetition = {}  # Dictionary to store histograms for all repetitions

    for repetition_key, presses in containers_by_press.items():
        histograms = []
        for press in presses:
            if press is None or 'ts' not in press or press['ts'].size == 0:
                # Skip if the press is None or has no events
                histograms.append(np.zeros(params['num_cols'] // params['cols_per_bin']))
                continue
            histograms.append(hist_y_rect(press, params))
        histograms_by_repetition[repetition_key] = np.array(histograms)  # Store histograms for the repetition as an array

    return histograms_by_repetition

#%%
"""
Function to normalize histograms across columns and presses.

Parameters:
    hists_by_press (ndarray): A 2D array where each row represents a histogram for a specific press, 
                               and each column represents the count of events in a specific bin.

Returns:
    hists_by_press_normalised_by_col_and_press (ndarray): A 2D array of the same shape as `hists_by_press`, 
                                                            where the histograms have been normalized first 
                                                            across columns and then across presses.

The function works in two steps:
1. Normalization across columns: Each value in a column is divided by the total sum of the values in that column. 
   This step ensures that the values in each column (bin) are proportionally scaled relative to the total events in each press.
   
2. Normalization across presses: Each value is divided by the total sum of values across all columns (bins) for that press. 
   This step ensures that the histograms are scaled across presses to provide a comparable distribution of events.

To prevent division by zero, a check is made to replace zeros with ones in the totals before division.
"""
def normalize_histograms(hists_by_press):
    # Cap the data at the threshold
    #hists_by_press = np.clip(hists_by_press, None, threshold)
    
    # Normalize across columns
    totals_by_col = np.sum(hists_by_press, axis=1)[:, np.newaxis]
    totals_by_col = np.where(totals_by_col == 0, 1, totals_by_col)  
    hists_by_press_normalised_by_col = hists_by_press / totals_by_col
    
    # Normalize across presses
    totals_by_press = np.sum(hists_by_press_normalised_by_col, axis=0)[np.newaxis, :]
    totals_by_press = np.where(totals_by_press == 0, 1, totals_by_press)  
    hists_by_press_normalised_by_col_and_press = hists_by_press_normalised_by_col / totals_by_press
    
    return hists_by_press_normalised_by_col_and_press
#%%
"""
Function to normalize histograms across columns and presses for all repetitions.

Parameters:
    histograms_by_repetition (dict): A dictionary where each key is a repetition identifier, and the value is a 2D array 
                                      representing histograms for all presses in that repetition. Each row represents 
                                      a histogram for a specific press, and each column represents the count of events 
                                      in a specific bin.

Returns:
    normalized_histograms_by_repetition (dict): A dictionary where each key is a repetition identifier, and the value 
                                                is a 2D array of normalized histograms for that repetition, where 
                                                the histograms are normalized first across columns and then across presses.

The function works in two steps for each repetition:
1. Normalization across columns: Each value in a column is divided by the total sum of the values in that column for that repetition. 
   This step ensures that the values in each column (bin) are proportionally scaled relative to the total events in each press.
   
2. Normalization across presses: Each value in a press histogram is divided by the total sum of values across all columns (bins) 
   for that press. This step ensures that the histograms are scaled across presses to provide a comparable distribution of events.

To prevent division by zero, a check is made to replace zeros with ones in the totals before division.
"""
def normalize_histograms_all_repetitions_dict(histograms_by_repetition):
    normalized_histograms_by_repetition = {}
    for repetition_key, histograms in histograms_by_repetition.items():
        totals_by_col = np.sum(histograms, axis=1)[:, np.newaxis]
        totals_by_col = np.where(totals_by_col == 0, 1, totals_by_col)
        hists_normalized_by_col = histograms / totals_by_col
        
        totals_by_press = np.sum(hists_normalized_by_col, axis=0)[np.newaxis, :]
        totals_by_press = np.where(totals_by_press == 0, 1, totals_by_press)
        hists_normalized_by_col_and_press = hists_normalized_by_col / totals_by_press
        normalized_histograms_by_repetition[repetition_key] = hists_normalized_by_col_and_press
    
    return normalized_histograms_by_repetition

#%%
"""
Function to compute the average of normalized histograms across repetitions for each press.

Parameters:
    normalized_histograms_by_repetition (dict): A dictionary where each key is a repetition identifier, 
                                                 and the value is a 2D array of normalized histograms for that repetition. 
                                                 Each row represents a histogram for a specific press, and each column 
                                                 represents the count of events in a specific bin.

Returns:
    average_histograms (ndarray): A 2D array where each row represents the averaged normalized histogram 
                                   for a specific press across all repetitions, and each column represents 
                                   the count of events in a specific bin.

The function works as follows:
1. For each press (indexed by `press_idx`), it sums the corresponding histograms from all repetitions.
2. The summed histograms are then averaged by dividing by the total number of repetitions.
3. The averaged histogram for each press is added to a list, which is returned as a 2D NumPy array.

This function assumes that all repetitions have the same number of presses (324 in this case) and normalizes across columns for each press.
"""
def average_normalized_histograms(normalized_histograms_by_repetition):
    # Initialize a list to store the average histograms for each press
    average_histograms = []

    # Assuming that all repetitions have the same number of presses (324 in this case)
    num_presses = len(normalized_histograms_by_repetition[list(normalized_histograms_by_repetition.keys())[0]])

    for press_idx in range(num_presses):
        # Sum up the normalized histograms across all repetitions for this press
        sum_histograms = np.zeros_like(normalized_histograms_by_repetition[list(normalized_histograms_by_repetition.keys())[0]][press_idx])
        
        for repetition_key, histograms in normalized_histograms_by_repetition.items():
            sum_histograms += histograms[press_idx]
        
        # Average the histograms by dividing by the number of repetitions (10)
        average_histogram = sum_histograms / len(normalized_histograms_by_repetition)
        
        # Add the averaged histogram to the list
        average_histograms.append(average_histogram)
    
    return np.array(average_histograms)

#%% Plotting Histogram for a specific press using KernelDensity
"""
Function to plot a histogram with a smooth curve (Kernel Density Estimation) for a specific press.

Parameters:
    hist (ndarray): A 2D array where each row represents a histogram for a specific press. Each column represents the count of events in a specific bin.
    title_prefix (str): Prefix for the title of the plot (not currently used in the function, but can be used to set a title).
    press_number (int): The press number (1-based index) for which the histogram and smooth curve will be plotted.
    bandwidth (float): The bandwidth parameter for the Kernel Density Estimation (KDE) used to smooth the curve.

Returns:
    None: Displays the histogram with the smoothed curve for the specified press.

The function works as follows:
1. It selects the histogram for the specified press (`press_number`), adjusting for 0-based indexing.
2. The histogram is plotted as bars with black edges and a transparent fill.
3. A smooth curve is generated using Kernel Density Estimation (KDE) based on the histogram data, with the specified `bandwidth` parameter controlling the smoothness.
4. The smooth curve is plotted on top of the histogram.
5. The function sets appropriate labels for the axes and displays the plot.

This function is useful for visualizing the distribution of events for a specific press along with a smooth estimate of the density.
"""
def plot_histogram_with_smooth_curve(hist, title_prefix, press_number, bandwidth):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Select the histogram for the specified press
    histogram = hist[press_number - 1]  # press_number - 1 for 0-based indexing
    bins = np.arange(len(histogram))  # Create bins based on the histogram length

    # Plot the histogram
    ax.bar(bins, histogram, width=1, edgecolor='black', alpha=0.6, label='Histogram')

    # Prepare data for KDE (smooth curve)
    bin_centers = np.arange(len(histogram))  # Use bin centers corresponding to histogram length
    kde = KernelDensity(bandwidth=bandwidth)  # Set bandwidth for smoothing
    kde.fit(bin_centers[:, np.newaxis], sample_weight=histogram)  # Fit KDE with histogram data

    # Generate smooth curve using KDE
    
    x_d = np.linspace(bins.min(), bins.max(), 1000)
    log_density = kde.score_samples(x_d[:, np.newaxis])
    smooth_curve = np.exp(log_density)

    # Plot the smooth curve
    ax.plot(x_d, smooth_curve, 'r-', linewidth=2, label='Smooth Curve')

    # Set titles and labels
    ax.set_title(f" Press {press_number} smoothed")
    ax.set_xlabel('Bins')
    ax.set_ylabel('Events')
    # ax.set_ylim(0, 0.1)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#%% Plotting Histogram for a specific press with gaussian_kde
from scipy.stats import gaussian_kde
"""
Function to plot a histogram with a sharper smooth curve using Gaussian Kernel Density Estimation (KDE).

Parameters:
    hist (ndarray): A 2D array where each row represents a histogram for a specific press. Each column represents the count of events in a specific bin.
    title_prefix (str): Prefix for the title of the plot, used to customize the plot title.
    press_number (int): The press number (1-based index) for which the histogram and smooth curve will be plotted.
    bandwidth_factor (float): A factor to adjust the sharpness of the smooth curve. A smaller value results in a sharper curve.
    resolution (int): The resolution (number of points) for generating the smooth curve. Higher values provide a smoother curve.

Returns:
    None: Displays the histogram with the smoothed curve using Gaussian KDE for the specified press.

The function works as follows:
1. It selects the histogram for the specified press (`press_number`), adjusting for 0-based indexing.
2. The histogram is plotted as bars with black edges and a transparent fill.
3. A smoother curve is generated using Gaussian Kernel Density Estimation (KDE), with the bin centers adjusted and scaled by the `bandwidth_factor`.
4. The smooth curve is plotted on top of the histogram.
5. The function sets appropriate titles, labels for the axes, and displays the plot.

This function is useful for visualizing the distribution of events for a specific press with a sharper smooth curve using Gaussian KDE.
"""

# Plot histograms with a sharper smooth curve using gaussian_kde
def plot_histogram_with_gaussian_kde(hist, title_prefix, press_number, bandwidth_factor, resolution):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Select the histogram for the specified press
    histogram = hist[press_number - 1]  # press_number - 1 for 0-based indexing
    bins = np.arange(len(histogram))  # Create bins based on the histogram length

    # Plot the histogram
    ax.bar(bins, histogram, width=0.8, edgecolor='black', alpha=0.6, label='Histogram')

    # Adjust bin centers and scale for bandwidth
    bin_centers = bins + 0.5  # Shift to center of bins
    scaled_bins = bin_centers / bandwidth_factor  # Scale for sharper KDE

    # Prepare data for gaussian_kde
    kde = gaussian_kde(scaled_bins, weights=histogram)  # Use histogram as weights for the KDE
    x_d = np.linspace(bins.min(), bins.max(), resolution)  # Higher resolution for smoothness
    scaled_x_d = x_d / bandwidth_factor  # Scale x_d similar to bins
    smooth_curve = kde(scaled_x_d)

    # Plot the smooth curve
    ax.plot(x_d, smooth_curve, 'g-', linewidth=2, label='Smooth Curve (Gaussian KDE)')

    # Set titles and labels
    ax.set_title(f"{title_prefix} Press {press_number} Smoothed with Gaussian KDE (Sharp)")
    ax.set_xlabel('Bins')
    ax.set_ylabel('Events')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
#%% plot the smooth curve using both kernel density and gaussian kde
"""
Function to plot a histogram with two smoothing methods (KernelDensity and Gaussian KDE) for comparison.

Parameters:
    hist (ndarray): A 2D array where each row represents a histogram for a specific press. Each column represents the count of events in a specific bin.
    title_prefix (str): Prefix for the title of the plot, used to customize the plot title.
    press_number (int): The press number (1-based index) for which the histogram and smooth curves will be plotted.
    bandwidth (float): The bandwidth parameter for the KernelDensity method, controlling the smoothness of the KernelDensity curve.
    kde_bandwidth_factor (float): A factor used to adjust the sharpness of the Gaussian KDE curve. A smaller value results in a sharper curve.
    resolution (int): The resolution (number of points) for generating the smooth curves. Higher values provide a smoother curve.

Returns:
    None: Displays the histogram with both smoothed curves (KernelDensity and Gaussian KDE) for the specified press.

The function works as follows:
1. It selects the histogram for the specified press (`press_number`), adjusting for 0-based indexing.
2. The histogram is plotted as bars with black edges and a transparent fill.
3. A smoother curve is generated using KernelDensity from Scikit-Learn, and plotted with the specified `bandwidth`.
4. The maximum value and position of the KernelDensity curve are marked and printed.
5. A second smoother curve is generated using Gaussian KDE from SciPy, with a sharpening effect controlled by the `kde_bandwidth_factor`.
6. The maximum value and position of the Gaussian KDE curve are also marked and printed.
7. The function sets appropriate titles, labels for the axes, and displays the plot.

This function is useful for comparing the smoothness of two different methods (KernelDensity and Gaussian KDE) when applied to the same histogram.
"""
def plot_histogram_with_both_methods(hist, title_prefix, press_number, bandwidth, kde_bandwidth_factor, resolution):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Select the histogram for the specified press
    histogram = hist[press_number - 1]  # press_number - 1 for 0-based indexing
    bins = np.arange(len(histogram))  # Create bins based on the histogram length

    # Plot the histogram
    ax.bar(bins, histogram, width=1, edgecolor='black', alpha=0.6, label='Histogram')

    # Smooth Curve using KernelDensity (Scikit-Learn)
    bin_centers = bins  # Use bin centers corresponding to histogram length
    kde = KernelDensity(bandwidth=bandwidth)  # Set bandwidth for smoothing
    kde.fit(bin_centers[:, np.newaxis], sample_weight=histogram)  # Fit KDE with histogram data
    x_d = np.linspace(bins.min(), bins.max(), resolution)  # Increase resolution
    log_density = kde.score_samples(x_d[:, np.newaxis])
    smooth_curve_kde = np.exp(log_density)
    ax.plot(x_d, smooth_curve_kde, 'r-', linewidth=2, label='Smooth Curve (KernelDensity)')

    # Find and mark maximum point for KernelDensity
    max_idx_kde = np.argmax(smooth_curve_kde)
    max_value_kde = smooth_curve_kde[max_idx_kde]
    max_position_kde = x_d[max_idx_kde]
    ax.plot(max_position_kde, max_value_kde, 'ro', label=f'Max KDE ({max_position_kde:.2f}, {max_value_kde:.2f})')
    print(f"KernelDensity: Max Value = {max_value_kde:.2f} at Position = {max_position_kde:.2f}")

    # Smooth Curve using Gaussian KDE (SciPy) with sharpening
    bin_centers_scaled = bins / kde_bandwidth_factor  # Scale for sharper KDE
    kde_scipy = gaussian_kde(bin_centers_scaled, weights=histogram)  # Use histogram as weights for the KDE
    x_d_scaled = x_d / kde_bandwidth_factor  # Scale x_d similar to bins
    smooth_curve_gaussian_kde = kde_scipy(x_d_scaled)
    ax.plot(x_d, smooth_curve_gaussian_kde, 'g-', linewidth=2, label='Smooth Curve (Gaussian KDE)')

    # Find and mark maximum point for Gaussian KDE
    max_idx_gaussian_kde = np.argmax(smooth_curve_gaussian_kde)
    max_value_gaussian_kde = smooth_curve_gaussian_kde[max_idx_gaussian_kde]
    max_position_gaussian_kde = x_d[max_idx_gaussian_kde]
    ax.plot(max_position_gaussian_kde, max_value_gaussian_kde, 'go', label=f'Max Gaussian KDE ({max_position_gaussian_kde:.2f}, {max_value_gaussian_kde:.2f})')
    print(f"Gaussian KDE: Max Value = {max_value_gaussian_kde:.2f} at Position = {max_position_gaussian_kde:.2f}")

    # Set titles and labels
    ax.set_title(f"{title_prefix} Press {press_number} Smoothed - Comparison")
    ax.set_xlabel('Bins')
    ax.set_ylabel('Events')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
#%% Function to calculate the activity based on the maximum value from the smooth curve
"""
Function to calculate the activity (position) from the smooth curve of a histogram using Kernel Density Estimation (KDE).

Parameters:
    hists (ndarray): A 2D array where each row represents a histogram for a specific press. Each column represents the count of events in a specific bin.
    bin_size (float): The size of each bin in the histogram, used to convert the position from bin index to pixel (or other unit).
    bandwidth (float): The bandwidth parameter for the KernelDensity method, controlling the smoothness of the curve.
    threshold (float): A threshold value used to filter out low activity. If the maximum activity value is below this threshold, the position is returned as NaN.

Returns:
    ndarray: A 1D array of activity positions (in pixels or other unit) corresponding to each press. If the maximum activity value is below the threshold, NaN is returned for that press.

The function works as follows:
1. It iterates over the rows in `hists`, which represent the histograms for each press.
2. For each press:
    - It fits a Kernel Density Estimation (KDE) model to the histogram data using the specified `bandwidth`.
    - A smooth curve is generated based on the KDE.
    - The maximum value and its corresponding position are identified in the smooth curve.
3. If the maximum value of the smooth curve is below the `threshold`, the corresponding position is set to NaN, indicating low activity.
4. The function returns an array of activity positions, where each element corresponds to the position of the maximum activity for each press.

This function is useful for determining the activity locations based on the smoothened histogram, considering a threshold to filter out low activity.
"""
def get_activity_from_smooth_curve(hists, bin_size, bandwidth, threshold):
    num_presses = hists.shape[0]
    activities = []

    for i in range(num_presses):
        histogram = hists[i]
        bins = np.arange(len(histogram))  # Create bin indices

        # Fit KDE using the histogram data
        bin_centers = bins
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(bin_centers[:, np.newaxis], sample_weight=histogram)

        # Generate a smooth curve from KDE
        x_d = np.linspace(bins.min(), bins.max(), 1000)
        log_density = kde.score_samples(x_d[:, np.newaxis])
        smooth_curve = np.exp(log_density)

        # Find the index and value of the maximum in the smooth curve
        max_index = np.argmax(smooth_curve)
        max_activity_value = smooth_curve[max_index]  # Maximum value (height)
        max_activity_position = x_d[max_index] * bin_size  # Position (pixel)

        # Apply threshold to the maximum value
        if max_activity_value < threshold:
            activities.append(np.nan)
        else:
            activities.append(max_activity_position)

    return np.array(activities)
    
#%%calculate the maximum activity for all of the presses using gaussian kde
"""
Function to calculate the activity (position) from the smooth curve of a histogram using Gaussian Kernel Density Estimation (KDE) for multiple repetitions.

Parameters:
    hists_by_repetition (dict): A dictionary where the keys are repetition identifiers (e.g., press repetitions) and the values are 2D arrays. Each array represents a set of histograms for the repetitions, where each row is a histogram for a specific press.
    bin_size (float): The size of each bin in the histogram, used to convert the position from bin index to pixel (or other unit).
    kde_bandwidth_factor (float): A factor to scale the bins for sharper Gaussian KDE.
    threshold (float): A threshold value used to filter out low activity. If the maximum activity value from the smooth curve is below this threshold, the position is returned as NaN.

Returns:
    dict: A dictionary where the keys are the repetition identifiers (from `hists_by_repetition`), and the values are 1D arrays of activity positions (in pixels or other unit) corresponding to each press in the repetition. If the maximum activity value is below the threshold, NaN is returned for that press.

The function works as follows:
1. It iterates over the repetitions in `hists_by_repetition`, processing each repetition's histograms.
2. For each histogram in a repetition:
    - It scales the bins for sharper Gaussian KDE using the `kde_bandwidth_factor`.
    - It fits a Gaussian KDE to the histogram data using the `gaussian_kde` function from `scipy.stats`.
    - A smooth curve is generated based on the Gaussian KDE.
    - The maximum value and its corresponding position are identified in the smooth curve.
3. If the maximum value of the smooth curve is below the `threshold`, the corresponding position is set to NaN.
4. The function returns a dictionary of activity positions for each repetition, with NaN values where the activity was below the threshold.

This function is useful for determining the activity locations across multiple repetitions, considering a threshold to filter out low activity.
"""

def get_activity_from_gaussian_kde_all_repetitions(hists_by_repetition, bin_size, kde_bandwidth_factor, threshold):
    activities_by_repetition = {}

    for repetition_key, hists in hists_by_repetition.items():
        num_presses = hists.shape[0]
        activities = []

        for i in range(num_presses):
            histogram = hists[i]
            bins = np.arange(len(histogram))  # Create bin indices

            # Skip invalid histograms
            if np.all(histogram == 0) or np.any(np.isnan(histogram)):
                activities.append(np.nan)
                continue

            # Scale bins for sharper Gaussian KDE
            bin_centers_scaled = bins / kde_bandwidth_factor

            # Fit Gaussian KDE using histogram as weights
            kde_scipy = gaussian_kde(bin_centers_scaled, weights=histogram)

            # Generate a smooth curve from Gaussian KDE
            x_d = np.linspace(bins.min(), bins.max(), 2000)  # High-resolution sampling
            x_d_scaled = x_d / kde_bandwidth_factor
            smooth_curve = kde_scipy(x_d_scaled)

            # Find the index and value of the maximum in the smooth curve
            max_index = np.argmax(smooth_curve)
            max_activity_value = smooth_curve[max_index]
            max_activity_position = x_d[max_index] * bin_size

            # Apply threshold to the maximum value
            if max_activity_value < threshold:
                activities.append(np.nan)
            else:
                activities.append(max_activity_position)

        activities_by_repetition[repetition_key] = np.array(activities)

    return activities_by_repetition
#%%calculate the maximum activity for all of the presses using gaussian kde

"""
Function to calculate the activity (position) from the smooth curve of a histogram using Gaussian Kernel Density Estimation (KDE) for a single repetition.

Parameters:
    hists (ndarray): A 2D numpy array where each row represents a histogram for a specific press. The columns represent bin counts for each press.
    bin_size (float): The size of each bin in the histogram, used to convert the position from bin index to pixel (or other unit).
    kde_bandwidth_factor (float): A factor to scale the bins for sharper Gaussian KDE.
    threshold (float): A threshold value used to filter out low activity. If the maximum activity value from the smooth curve is below this threshold, the position is returned as NaN.

Returns:
    ndarray: A 1D numpy array containing the activity positions (in pixels or other unit) corresponding to each press. If the maximum activity value is below the threshold, NaN is returned for that press.

The function works as follows:
1. It iterates over the presses in the histograms `hists`, processing each press's histogram.
2. For each histogram:
    - It scales the bins for sharper Gaussian KDE using the `kde_bandwidth_factor`.
    - It fits a Gaussian KDE to the histogram data using the `gaussian_kde` function from `scipy.stats`.
    - A smooth curve is generated based on the Gaussian KDE.
    - The maximum value and its corresponding position are identified in the smooth curve.
3. If the maximum value of the smooth curve is below the `threshold`, the corresponding position is set to NaN.
4. The function returns a 1D numpy array of activity positions for each press, with NaN values where the activity was below the threshold.

This function is useful for determining the activity locations, considering a threshold to filter out low activity.
"""
def get_activity_from_gaussian_kde(hists, bin_size, kde_bandwidth_factor, threshold):
    num_presses = hists.shape[0]
    activities = []

    for i in range(num_presses):
        histogram = hists[i]
        bins = np.arange(len(histogram))  # Create bin indices

        # Scale bins for sharper Gaussian KDE
        bin_centers_scaled = bins / kde_bandwidth_factor

        # Fit Gaussian KDE using histogram as weights
        kde_scipy = gaussian_kde(bin_centers_scaled, weights=histogram)

        # Generate a smooth curve from Gaussian KDE
        x_d = np.linspace(bins.min(), bins.max(), 2000)  # High-resolution sampling
        x_d_scaled = x_d / kde_bandwidth_factor
        smooth_curve = kde_scipy(x_d_scaled)

        # Find the index and value of the maximum in the smooth curve
        max_index = np.argmax(smooth_curve)
        max_activity_value = smooth_curve[max_index]  # Maximum value (height)
        max_activity_position = x_d[max_index] * bin_size  # Position (pixel)

        # Apply threshold to the maximum value
        if max_activity_value < threshold:
            activities.append(np.nan)
        else:
            activities.append(max_activity_position)

    return np.array(activities)

#%%
import gc
from sklearn.cluster import DBSCAN

def compute_activity_centroids_DBSCAN(segments, time_windows=[(0, 4/72)], eps=10, min_samples=1, label=''):
    """
    Compute centroids of clustered activity for given segments using DBSCAN.

    Parameters:
        segments: list of data dictionaries (each with keys 'x', 'y', 'ts')
        time_windows: list of (start_time, end_time) tuples for cropping
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        label: string to label output prints (e.g., 'Camera 1')

    Returns:
        DataFrame with centroid coordinates for each segment
    """
    centroids = []

    for i, segment in enumerate(segments):
        try:
            cropped_data_x = []
            cropped_data_y = []

            # Apply time filtering
            for start_time, end_time in time_windows:
                cropped_segment = cropTime(segment, startTime=start_time, stopTime=end_time)
                cropped_data_x.extend(cropped_segment['x'])
                cropped_data_y.extend(cropped_segment['y'])

            x_values = np.array(cropped_data_x)
            y_values = np.array(cropped_data_y)

            if len(x_values) == 0:
                print(f"No events in Segment {i+1} ({label})")
                centroids.append((np.nan, np.nan))
                continue

            # DBSCAN clustering
            pixel_positions = np.column_stack((x_values, y_values))
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pixel_positions)
            labels = clustering.labels_

            # Filter out noise points
            filtered_positions = pixel_positions[labels != -1]

            if len(filtered_positions) == 0:
                print(f"No cluster found in Segment {i+1} ({label})")
                centroids.append((np.nan, np.nan))
                continue

            # Calculate centroid
            centroid = np.mean(filtered_positions, axis=0)
            centroids.append(centroid)
            print(f"Centroid for Segment {i+1} ({label}): X = {centroid[0]:.2f}, Y = {centroid[1]:.2f}")

        except MemoryError:
            print(f"Memory error at Segment {i+1} ({label}), skipping...")
            centroids.append((np.nan, np.nan))

        finally:
            gc.collect()

    return pd.DataFrame(centroids, columns=["X", "Y"])
#%%

def calculate_theta_with_skew(pixel, pixel_mid, fov_half, skew_angle):
    theta = (pixel - pixel_mid) / pixel_mid * fov_half
    theta_with_skew = theta + skew_angle
    return theta_with_skew


def calculate_theta_exponential(pixel, pixel_mid, fov_half, skew_angle, W):
    # Calculate the base theta (angle from the middle of the FOV)
    theta_base = (pixel - pixel_mid) / pixel_mid * fov_half
    
    # Apply inverted exponential nonlinearity
    nonlinearity_factor = np.exp(-W * np.abs((pixel - pixel_mid) / pixel_mid))
    theta = theta_base * nonlinearity_factor
    
    # Normalize theta to match the full FOV range
    theta_normalized = theta 
    
    # Add the skew angle as an angular offset
    theta_with_skew = theta_normalized + skew_angle

    return theta_with_skew


def point_slope_line(m, x1, y1):
    def line_eq(x):
        return m * (x - x1) + y1
    return line_eq


def find_intersection_fsolve(ly1, ly2, initial_guess):
    def line_diff(x):
        return ly1(x) - ly2(x)
    
    x_intersection = fsolve(line_diff, initial_guess, xtol=1e-6)[0]
    y_intersection = ly1(x_intersection)
    ip = {'xi': x_intersection, 'yi': y_intersection}
    return ip

# Function to calculate the tangent of a degree
def tan(degree):
    radians = math.radians(degree)
    tangent = math.tan(radians)
    return tangent

def calculate_distance(point1, point2):
    x1, y1 = list(point1.values())[:2]
    x2, y2 = list(point2.values())[:2]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def is_inside_skin(x, y, skin_center, skin_width, skin_height):
    x_min = skin_center['xo'] - skin_width / 2
    x_max = skin_center['xo'] + skin_width / 2
    y_min = skin_center['yo'] - skin_height / 2
    y_max = skin_center['yo'] + skin_height / 2
    return x_min <= x <= x_max and y_min <= y <= y_max

#%%
"""
Function to perform triangulation for multiple press positions based on pixel activities from two cameras.

Parameters:
    p1 (dict): The position of the first camera in the format {'x1': x1, 'y1': y1}.
    p2 (dict): The position of the second camera in the format {'x2': x2, 'y2': y2}.
    x_activity1 (ndarray): Array of pixel activities for Camera 1 (pixel positions).
    x_activity2 (ndarray): Array of pixel activities for Camera 2 (pixel positions).
    skew_angle_camera1 (float): The skew angle for Camera 1 (degrees).
    skew_angle_camera2 (float): The skew angle for Camera 2 (degrees).
    po (dict): The position of the object in the format {'xo': xo, 'yo': yo}.
    pixel_mid_camera1 (int): The midpoint pixel of Camera 1.
    pixel_mid_camera2 (int): The midpoint pixel of Camera 2.
    fov_camera1 (float): The field of view (FOV) of Camera 1 (degrees).
    fov_camera2 (float): The field of view (FOV) of Camera 2 (degrees).

Returns:
    all_intersections (list): A list of dictionaries containing the x and y coordinates of the intersections for each press, or 0 if no valid activity is found.

The triangulation process involves:
1. Calculating the distances `d1` and `d2` from the object position `po` to the camera positions `p1` and `p2`.
2. Iterating over each press:
   - For each press, the pixel activities from both cameras (`x_activity1` and `x_activity2`) are used to calculate the corresponding angles (`theta1` and `theta2`) by considering skew angles.
   - Using these angles, the distances `d11` and `d22` are calculated using the tangent of the angles.
   - The slopes `m1` and `m2` are computed from the camera positions and the calculated distances.
   - The intersection of the two lines is found using the `find_intersection_fsolve` function, which solves for the intersection point of the two lines representing the press position in 3D space.
3. The function returns a list of intersection points (x, y) for each press. If a valid intersection is not found (i.e., if pixel activities are `NaN`), the function appends 0 for that press.

This function assumes that the object and cameras are in a 2D plane, and it calculates the intersection of lines originating from the camera positions and passing through the pixel activities for each camera.
"""

def triangulation(p1, p2, x_activity1, x_activity2,
                  skew_angle_camera1, skew_angle_camera2, po, 
                  pixel_mid_camera1, pixel_mid_camera2, fov_camera1, fov_camera2, W_camera1, W_camera2):
    all_intersections = []
    d1 = calculate_distance(po, p1)
    d2 = calculate_distance(po, p2)

    for i in range(len(x_activity1)):
        pixel_activity1 = x_activity1[i]
        pixel_activity2 = x_activity2[i]

        if np.isnan(pixel_activity2) or np.isnan(pixel_activity1):
            all_intersections.append(0)
            continue

        theta1 = calculate_theta_exponential(pixel_activity1, pixel_mid_camera1, fov_camera1 / 2, skew_angle_camera1, W_camera1)
        theta2 = calculate_theta_exponential(pixel_activity2, pixel_mid_camera2, fov_camera2 / 2, skew_angle_camera2, W_camera2)

        d11 = tan(theta1) * d1
        d22 = tan(theta2) * d2

        m1 = ((po['yo'] + d11) - p1['y1']) / (po['xo'] - p1['x1'])
        m2 = (po['yo'] - p2['y2']) / ((po['xo'] - d22) - p2['x2'])

        ly1 = point_slope_line(m1, p1['x1'], p1['y1'])
        ly2 = point_slope_line(m2, p2['x2'], p2['y2'])
        initial_guess = (p1['x1'] + p2['x2']) / 2
        intersection = find_intersection_fsolve(ly1, ly2, initial_guess)

        all_intersections.append(intersection)

    return all_intersections


#%%
"""
The function is similar to the triangulation function, with the difference being that it calculates the 
intersections separately for each repetition when there are multiple repetitions in the data.
"""
def triangulate_for_each_repetition(p1, p2, x_activity1, x_activity2, 
                                      skew_angle_camera1, skew_angle_camera2, po, 
                                      pixel_mid_camera1, pixel_mid_camera2, fov_camera1, fov_camera2):
    all_intersections_by_repetition = {}  # Dictionary to store intersections for each repetition

    for repetition_key in x_activity1.keys():
        all_intersections = []  # List to store intersection points for this repetition
        d1 = calculate_distance(po, p1)
        d2 = calculate_distance(po, p2)
        # Loop through the activities for each press (i.e., for each index from 0 to 323)
        for i in range(len(x_activity1)):  
            # Get the activities for Camera 1 and Camera 2 for this repetition
            pixel_activity1 = x_activity1[repetition_key][i]
            pixel_activity2 = x_activity2[repetition_key][i]
            
            if np.isnan(pixel_activity2) or np.isnan(pixel_activity1):
                print(f"Skipping triangulation for press {i + 1} in repetition {repetition_key} due to no visibility")
                all_intersections.append(0)  # No intersection if activity is missing
                continue
            
            # Calculate theta for Camera 1 and Camera 2
            theta1 = calculate_theta_with_skew(pixel_activity1, pixel_mid_camera1, fov_camera1 / 2, skew_angle_camera1)
            theta2 = calculate_theta_with_skew(pixel_activity2, pixel_mid_camera2, fov_camera2 / 2, skew_angle_camera2)
            
            # Calculate distances based on tangent and theta
            d11 = tan(theta1) * d1
            d22 = tan(theta2) * d2
            
            # Slope calculation for the lines
            m1 = ((po['yo'] + d11) - p1['y1']) / (po['xo'] - p1['x1'])
            m2 = (po['yo'] - p2['y2']) / ((po['xo'] - d22) - p2['x2'])
            
            # Define the lines
            ly1 = point_slope_line(m1, p1['x1'], p1['y1'])
            ly2 = point_slope_line(m2, p2['x2'], p2['y2'])
            
            # Find intersection (press position)
            initial_guess = (p1['x1'] + p2['x2']) / 2
            intersection = find_intersection_fsolve(ly1, ly2, initial_guess)

            all_intersections.append(intersection)  # Store the intersection for this press

        # Store the intersections for this repetition
        all_intersections_by_repetition[repetition_key] = all_intersections

    return all_intersections_by_repetition

#%% Updated objective function for least squares across all repetitions
"""
Function to calculate the error between estimated and actual press positions for least squares optimization.

Parameters:
    params (list or ndarray): A list or array containing the parameters for optimization:
                              [x1, y1, x2, y2, skew_angle_camera1, skew_angle_camera2].
                              These parameters represent the positions of two cameras (x1, y1, x2, y2)
                              and the skew angles for both cameras (skew_angle_camera1, skew_angle_camera2).
    intersections (list): A list of dictionaries containing the estimated intersection points for each press,
                           with keys 'xi' (x-coordinate) and 'yi' (y-coordinate). A value of 0 indicates no valid intersection.
    press_coordinates_df (DataFrame): A DataFrame containing the actual (ground truth) press coordinates, with columns
                                       for x and y positions.

Returns:
    np.ndarray: A NumPy array containing the errors for each press, where the error is the Euclidean distance between
                the estimated intersection point and the actual press position. A large error value (1000) is used for
                missing intersections (i.e., when the intersection value is 0).

The function works as follows:
1. It iterates through each press, using the press ID to access the corresponding actual press position (from `press_coordinates_df`) and the estimated intersection (from `intersections`).
2. For each press:
   - If a valid intersection is found (i.e., `intersection != 0`), the Euclidean distance between the estimated position (`xi`, `yi`) and the actual position (`actual_x`, `actual_y`) is calculated.
   - If no valid intersection is found (i.e., `intersection == 0`), a large error value (1000) is assigned to indicate the missing intersection.
3. The function returns an array of errors, where each error corresponds to a press's position error.

This function can be used to calculate the errors for each press during an optimization process (such as least squares) to minimize the difference between the estimated and actual positions.
"""


# def combined_least_squares(params, intersections, press_coordinates_df):
#     x1, y1, x2, y2, skew_angle_camera1, skew_angle_camera2, W_camera1, W_camera2 = params

#     errors = []
#     for press_id, actual_press in enumerate(press_coordinates_df.iterrows()):
#         actual_x, actual_y = actual_press[1][0], actual_press[1][1]
#         intersection = intersections[press_id]
        
#         # Check for None or 0 values to avoid subscriptable errors
#         if intersection is not None and intersection != 0 and isinstance(intersection, dict) and 'xi' in intersection and 'yi' in intersection:
#             try:
#                 x_estimated = intersection['xi']
#                 y_estimated = intersection['yi']
#                 error = np.sqrt((x_estimated - actual_x)**2 + (y_estimated - actual_y)**2)
#             except KeyError:
#                 print(f"Skipping press {press_id}: Missing 'xi' or 'yi' in estimated intersection.")
#                 error = 1000  # Large penalty for missing values
#         else:
#             error = 1000  # Large error for missing intersection
        
#         errors.append(error)
    
#     return np.array(errors)

def combined_least_squares(params, intersections, press_coordinates_df):
    x1, y1, x2, y2, skew_angle_camera1, skew_angle_camera2, W_camera1, W_camera2 = params

    errors = []
    for press_id, actual_press in enumerate(press_coordinates_df.iterrows()):
        actual_x, actual_y = actual_press[1][0], actual_press[1][1]
        intersection = intersections[press_id]
        
        # Ensure valid intersection
        if intersection is not None and isinstance(intersection, dict) and 'xi' in intersection and 'yi' in intersection:
            x_estimated = intersection['xi']
            y_estimated = intersection['yi']
            error = np.sqrt((x_estimated - actual_x) ** 2 + (y_estimated - actual_y) ** 2)
        else:
            error = np.mean(errors) if errors else 50  # Instead of 1000, use mean error or 50

        errors.append(error)
    
    return np.array(errors)

def combined_least_squares_differential(params, intersections, press_coordinates_df):
    x1, y1, x2, y2, skew_angle_camera1, skew_angle_camera2, W_camera1, W_camera2 = params

    errors = []
    for press_id, actual_press in enumerate(press_coordinates_df.iterrows()):
        actual_x, actual_y = actual_press[1][0], actual_press[1][1]
        intersection = intersections[press_id]
        
        # Ensure valid intersection
        if intersection is not None and isinstance(intersection, dict) and 'xi' in intersection and 'yi' in intersection:
            x_estimated = intersection['xi']
            y_estimated = intersection['yi']
            error = np.sqrt((x_estimated - actual_x) ** 2 + (y_estimated - actual_y) ** 2)
        else:
            error = np.mean(errors) if errors else 50  # Instead of 1000, use mean error or 50

        errors.append(error)
    
    #Instead of returning an array, return a single scalar (sum of squared errors)
    return np.sum(np.square(errors))  # Sum of squared errors (SSE)


#%%
"""
Function to calculate the Root Mean Squared Error (RMSE) and Standard Deviation of Errors for estimated press positions.

Parameters:
    estimated_presses (list or ndarray): A list or array of dictionaries containing the estimated press positions, with
                                          keys 'xi' (estimated x-coordinate) and 'yi' (estimated y-coordinate). A value of 0
                                          indicates an invalid or missing estimated press.
    press_coordinates (list or DataFrame): A list or DataFrame containing the ground truth (actual) press coordinates, with
                                            'x' and 'y' as the actual press positions.

Returns:
    tuple: A tuple containing:
        - rmse (float): The Root Mean Squared Error (RMSE) of the estimated press positions compared to the actual positions.
        - std_dev_error (float): The standard deviation of the errors between estimated and actual positions.

The function works as follows:
1. It iterates through each press, comparing the estimated press position (`estimated_presses`) to the ground truth (`press_coordinates`).
2. For each press:
   - If the estimated press is valid (i.e., not equal to 0), the Euclidean distance between the estimated position (`xi`, `yi`) and the actual position (`actual_x`, `actual_y`) is calculated.
   - If the estimated press is invalid (i.e., equal to 0), the press is skipped.
3. The function computes the RMSE by taking the square root of the mean squared errors of all valid presses.
4. It also calculates the standard deviation of the errors to assess the spread of errors around the mean.
5. The function returns the RMSE and the standard deviation of errors as the output metrics.

This function is useful for evaluating the accuracy of estimated press positions in comparison to the ground truth, providing both a measure of central tendency (RMSE) and the variability (standard deviation) of the estimation errors.
"""

def calculate_rmse_and_error_metrics(estimated_presses, press_coordinates):
    errors = []

    for i, actual_press in enumerate(press_coordinates):
        # Get actual x and y coordinates from the ground truth
        actual_x = actual_press['x']
        actual_y = actual_press['y']

        # Get the corresponding estimated press position
        estimated_press = estimated_presses[i]

        # Skip if the estimated_press is None or 0 (invalid press)
        if estimated_press is None or estimated_press == 0:
            continue

        try:
            # Extract x and y coordinates from the estimated press
            x_estimated = estimated_press['xi']
            y_estimated = estimated_press['yi']

            # Calculate the Euclidean distance (error)
            error = np.sqrt((x_estimated - actual_x) ** 2 + (y_estimated - actual_y) ** 2)
            errors.append(error)

        except KeyError:
            print(f"Skipping index {i}: Missing 'xi' or 'yi' in estimated press.")
            continue

    # Convert errors to a NumPy array for calculations
    errors = np.array(errors)

    if errors.size > 0:
        rmse = np.sqrt(np.mean(errors ** 2))  # Root Mean Squared Error
        std_dev_error = np.std(errors)        # Standard Deviation of Errors
    else:
        rmse = np.nan  # Return NaN to indicate no valid data
        std_dev_error = np.nan

    return rmse, std_dev_error


#%% Finding the threshold
"""
Function to find the best threshold for press position estimation based on the relationship between activity values and distance.

Parameters:
    camera_data (str): A label or identifier for the camera being analyzed.
    press_coordinates (list or DataFrame): A list or DataFrame containing the ground truth press coordinates, with 'x' and 'y' as the actual press positions.
    all_intersections (list): A list of dictionaries containing the estimated intersection points, with 'xi' and 'yi' as the estimated positions.
    activity_curve (list or ndarray): A list or array of pixel activity values for each press, which corresponds to the activity at each position.
    histograms (list): A list of histograms for each press, used for calculating the kernel density estimate (KDE).
    bandwidth (float, optional): The bandwidth parameter for the Kernel Density Estimation (KDE). Default is 1.
    distance_threshold (float, optional): The maximum allowed distance (in mm) between the estimated and actual press positions. Presses exceeding this distance are skipped. Default is 150.

Returns:
    tuple: A tuple containing:
        - max_activity_values (list): A list of the maximum activity values (estimated from the KDE) for each valid press.
        - distances (list): A list of the Euclidean distances between the estimated and actual press positions for each valid press.

The function works as follows:
1. It iterates over each press in the `press_coordinates` list and checks the corresponding intersection in `all_intersections`.
2. If the intersection is invalid (i.e., 0), the press is skipped.
3. It calculates the Euclidean distance between the estimated and actual press positions and skips any presses where the distance exceeds the `distance_threshold`.
4. For each valid press, the function retrieves the pixel activity from `activity_curve` and skips NaN values.
5. It then calculates the maximum activity value using Kernel Density Estimation (KDE) based on the corresponding histogram for that press. The KDE smooths the histogram, and the maximum activity value is obtained from the smoothed curve.
6. The maximum activity value and the distance are stored for valid presses.
7. Finally, the function plots the relationship between the maximum activity values and the distances, helping to visualize how these two factors are related.
8. The function returns the lists of `max_activity_values` and `distances` for further analysis.

This function is useful for determining how activity values, derived from histograms and KDE, correlate with the distance between estimated and actual press positions, which can assist in identifying the optimal threshold for accurate press estimation.
"""

def find_best_threshold(camera_data, press_coordinates, all_intersections, activity_curve, histograms, bandwidth=1, distance_threshold=150):

    max_activity_values = []
    distances = []

    for press_id, actual_press in enumerate(press_coordinates):
        actual_x, actual_y = actual_press['x'], actual_press['y']
        intersection = all_intersections[press_id]

        # Skip invalid intersections
        if intersection == 0:
            continue

        x_estimated, y_estimated = intersection['xi'], intersection['yi']

        # Calculate distance and skip if it exceeds the threshold
        distance = np.sqrt((x_estimated - actual_x) ** 2 + (y_estimated - actual_y) ** 2)
        if distance > distance_threshold:
            continue

        # Get activity value and skip NaNs
        pixel_activity = activity_curve[press_id]
        if np.isnan(pixel_activity):
            continue

        # Perform KDE on histogram to find the maximum activity value
        histogram = histograms[press_id]
        bins = np.arange(len(histogram))
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(bins[:, np.newaxis], sample_weight=histogram)

        x_d = np.linspace(bins.min(), bins.max(), 1000)
        log_density = kde.score_samples(x_d[:, np.newaxis])
        smooth_curve = np.exp(log_density)
        max_activity_value = smooth_curve[np.argmax(smooth_curve)]

        # Store results
        max_activity_values.append(max_activity_value)
        distances.append(distance)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(max_activity_values, distances, alpha=0.6, label=camera_data)
    plt.title(f'{camera_data}: Max Activity Value vs Distance')
    plt.xlabel('Max Activity Value')
    plt.ylabel('Distance (mm)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return max_activity_values, distances

#%% plot the result



def plot_fov_lines(camera, fov_half_angle, color, x_key, y_key, angle_offset=0, ax=None):
    """Plots the field of view lines for a given camera with an optional angle offset."""
    angles = [-fov_half_angle, fov_half_angle]  # angles to left and right edge
    for angle in angles:
        # Apply the angle offset (in degrees) and convert to radians
        angle_rad = math.radians(angle + angle_offset)
        fov_x = camera[x_key] + 100 * math.cos(angle_rad)  # adjust the multiplier for desired line length
        fov_y = camera[y_key] + 100 * math.sin(angle_rad)
        # Plot the FOV line
        ax.plot([camera[x_key], fov_x], [camera[y_key], fov_y], linestyle='--', color=color, alpha=0.5)





def visualize_press_estimation(intersections, press_coordinates, po, p1, p2, fov_camera1, fov_camera2, rmse):
    """Visualizes estimated presses connected to their ground truth positions."""
    
    # Set up the workspace plot
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim([75, -75])  # Adjusted to match typical press coordinate ranges
    ax.set_ylim([85, -75])

    ax.set_xlabel("X (mm)", labelpad=-2)
    ax.set_ylabel("Y (mm)", labelpad=-5)
    ax.set_aspect('equal', 'box')

    # Draw the cameras' positions
    ax.scatter(p1['x1'], p1['y1'], color='red', marker='s', s=50, label='Camera 1 (p1)', zorder=5)
    ax.scatter(p2['x2'], p2['y2'], color='blue', marker='s', s=50, label='Camera 2 (p2)', zorder=5)

    # Draw the rectangular skin area
    rect_width = 100
    rect_height = 100
    rect_x = po['xo'] - rect_width / 2
    rect_y = po['yo'] - rect_height / 2
    rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, edgecolor='black', facecolor='none', linestyle='-', label='Skin Area')
    ax.add_patch(rect)

    skipped_label_used = False
    # Iterate over all presses to plot them and their ground truth connections
    for press_num, (estimated_press_coords, gt) in enumerate(zip(intersections, press_coordinates), start=1):
        x_ground_truth = gt['x']
        y_ground_truth = gt['y']

        if estimated_press_coords is None or estimated_press_coords == 0 or not isinstance(estimated_press_coords, dict):
            # Handle missing intersections
            if not skipped_label_used:
                ax.scatter(x_ground_truth, y_ground_truth, color='blue', marker='x', s=8, label='Skipped Presses')
                skipped_label_used = True
            else:
                ax.scatter(x_ground_truth, y_ground_truth, color='blue', marker='x', s=8)
        else:
            # Extract estimated coordinates safely
            try:
                x_estimated = estimated_press_coords.get('xi', None)
                y_estimated = estimated_press_coords.get('yi', None)

                if x_estimated is not None and y_estimated is not None:
                    ax.scatter(x_ground_truth, y_ground_truth, color='black', marker='o', s=2.5, label='Ground Truth' if press_num == 32 else "")
                    ax.scatter(x_estimated, y_estimated, color='red', s=2.5, marker='o', label='Estimated Press' if press_num == 32 else "")
                    ax.plot([x_ground_truth, x_estimated], [y_ground_truth, y_estimated], color='gray', linestyle='--', alpha=1, linewidth=0.5)
            except KeyError:
                print(f"Skipping press {press_num}: Missing 'xi' or 'yi' in estimated intersection.")

    # Draw the actual path of the ground truth presses
    actual_x_coords = [p['x'] for p in press_coordinates]
    actual_y_coords = [p['y'] for p in press_coordinates]
    ax.plot(actual_x_coords, actual_y_coords, color='green', linestyle='-', linewidth=0.5)

    # Add legend and RMSE text
    ax.legend(loc='upper right', fontsize=5.5)
    plt.text(67, -50, f"RMSE: {rmse:.2f}", fontsize=5.5, color='black')

    # Display the plot
    plt.grid(True, zorder=0)
    plt.show()



#%%Find the activities using DBSCAN

    """
    Applies DBSCAN clustering to find the center of activity for a specified number of event containers.

    Parameters:
    - containers_list: List of event containers (one per press).
    - num_presses: Number of presses to process (default: 250).
    - start_time: Start time of the time window for cropping (default: 0).
    - end_time: End time of the time window for cropping (default: 3).
    - eps: DBSCAN parameter - maximum distance between points to be considered neighbors.
    - min_samples: DBSCAN parameter - minimum points required to form a cluster.

    Returns:
    - centroids_df: DataFrame containing the computed centroids (X, Y).
    """
    
from sklearn.cluster import DBSCAN
import gc  # Garbage collector for memory optimization
import pandas as pd
from bimvee.split import cropTime

def compute_activity_centroids(containers_list, num_presses=250, start_time=0, end_time=3, eps=10, min_samples=10):

    
    centroids = []  # List to store centroids

    for i, press_container in enumerate(containers_list):
        if i >= num_presses:  # Stop processing if we reach the specified number of presses
            print(f"\nReached Press {num_presses}. Stopping processing.")
            break

        try:
            cropped_data_x = []
            cropped_data_y = []

            # Crop each press for the given time window
            cropped_segment = cropTime(press_container, startTime=start_time, stopTime=end_time)
            cropped_data_x.extend(cropped_segment['x'])
            cropped_data_y.extend(cropped_segment['y'])

            # Convert to NumPy arrays
            x_values = np.array(cropped_data_x)
            y_values = np.array(cropped_data_y)

            # If no events, store NaN centroid and continue
            if len(x_values) == 0:
                print(f"No events for Press {i+1}")
                centroids.append((np.nan, np.nan))
                continue

            # Apply DBSCAN Clustering
            pixel_positions = np.column_stack((x_values, y_values))
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pixel_positions)
            labels = clustering.labels_

            # Remove noise (-1 is noise label in DBSCAN)
            filtered_positions = pixel_positions[labels != -1]

            if len(filtered_positions) == 0:
                print(f"No cluster found for Press {i+1}")
                centroids.append((np.nan, np.nan))
                continue

            # Compute centroid
            centroid = np.mean(filtered_positions, axis=0)
            centroids.append(centroid)
            print(f"Centroid for Press {i+1}: X = {centroid[0]:.2f}, Y = {centroid[1]:.2f}")

        except MemoryError:
            print(f"MemoryError encountered at Press {i+1}. Skipping.")
            centroids.append((np.nan, np.nan))

        finally:
            gc.collect()  # Free up memory

    # Convert results to DataFrame
    centroids_df = pd.DataFrame(centroids, columns=['X', 'Y'])
    return centroids_df

#%%

def plot_top_down_view_from_smooth_curve_all_presses(smooth_curve, title_prefix):

    # Press indices: 1 to 289 (since each value corresponds to one press)
    presses = np.arange(1, len(smooth_curve) + 1)
    
    # Plot all press values as separate points, with swapped axes
    plt.figure(figsize=(12, 20))  # Increased height for better resolution
    plt.scatter(smooth_curve, presses, color='black', marker='o', s=10)  # Single color and smaller markers

    # Connect the points with lines
    plt.plot(smooth_curve, presses, linestyle='-', color='black', alpha=0.7, linewidth=1)

    # Swapped labels for the axes
    plt.xlabel('Max Activity Value')
    plt.ylabel('Press Index')
    plt.title(f"{title_prefix}")

    # Invert y-axis to make the first press at the top
    plt.gca().invert_yaxis()

    # Custom y-axis ticks for Press Index: More resolution and space between them
    plt.yticks(np.arange(1, len(smooth_curve) + 1, 10))  # Every 10th press for better spacing
    
    plt.grid(True, linestyle='--', alpha=0.5)  # Lighter grid lines
    
    plt.show()
    
    
#%% Function to calculate average event rate (events per second)
def calculate_average_event_rate(containers_by_press_list, press_duration):
    """
    Calculates the average event rate from DVS camera data across multiple presses.

    Parameters:
    -----------
    containers_by_press_list : dict
        A dictionary where each key (e.g., a repetition ID) maps to a list of press containers.
        Each press container is a dictionary containing DVS event data, and must include a key 'ts'
        which stores the event timestamps for that press.
    
    press_duration : float
        Duration of each press in seconds (assumed to be constant across all presses).

    Returns:
    --------
    avg_event_rate : float
        The average number of DVS events per second (i.e., event rate), averaged across all presses.
    """
    total_events = 0
    total_presses = 0

    # Iterate over all repetitions and their corresponding press containers
    for repetition_key, press_list in containers_by_press_list.items():
        for press_container in press_list:
            total_events += len(press_container['ts'])  # Count event timestamps
            total_presses += 1

    # Compute average event rate: total events per press, divided by duration
    avg_event_rate = (total_events / total_presses) / press_duration if total_presses > 0 else 0
    return avg_event_rate

#%% Function to calculate event rates
def compute_event_rates(event_container, period=0.01):
    """
    Computes the event rate over time from a DVS event container by dividing the event stream
    into fixed-length time intervals (bins) and counting the number of events in each.

    Parameters:
    -----------
    event_container : dict
        A dictionary containing DVS event data. It must include a key 'ts' (timestamps),
        where 'ts' is a list or array of timestamps (in seconds or microseconds).

    period : float, optional (default=0.01)
        The length of each time bin in seconds. Events will be counted within each bin
        to calculate the rate.

    Returns:
    --------
    mid_times : np.ndarray
        An array containing the center time of each bin. Useful for plotting the event rate over time.

    event_rates : np.ndarray
        An array containing the event rate (events per second) in each time bin.
        Calculated as the number of events in the bin divided by the bin duration (`period`).
    
    Notes:
    ------
    - If the input container does not contain any timestamps, or 'ts' is missing or empty,
      the function returns two empty arrays.
    - This function is useful for analyzing the temporal activity pattern of a DVS sensor.
    """
    if 'ts' not in event_container or len(event_container['ts']) == 0:
        return np.array([]), np.array([])

    ts = np.array(event_container['ts'])  
    start_time = np.min(ts)
    end_time = np.max(ts)

    # Create time bins
    end_times = np.arange(start_time, end_time, period) + period
    mid_times = end_times - (period / 2)

    # Find event counts per bin
    end_ids = np.searchsorted(ts, end_times)
    counts = np.diff(np.insert(end_ids, 0, 0))
    event_rates = counts / period

    return mid_times, event_rates



#%% Finding the centroids of the events using DBSCAN

from sklearn.cluster import DBSCAN


def plot_press_centroid(press_container,
                        press_id=None,
                        camera_name="Camera 1",
                        time_windows=[(0, 3)],
                        eps=1,
                        min_samples=1,
                        xlim=(0, 640),
                        ylim=(0, 480)):
    """
    Crop a press to the given time windows, find the densest DBSCAN cluster,
    plot it, and print the centroid.

    Parameters
    ----------
    press_container : dict
        HDF5 press container (must contain 'x', 'y', 'ts').
    press_id : int or None
        Used only for printing / title.  If None, no ID is shown.
    camera_name : str
        Text shown in the plot title.
    time_windows : list of tuple
        Each tuple is (start_time, stop_time) in seconds.
    eps, min_samples : float, int
        DBSCAN parameters.
    xlim, ylim : tuple
        Axis limits for the plot.

    Returns
    -------
    (cx, cy) : tuple
        Centroid coordinates, or (np.nan, np.nan) if no cluster found.
    """
    xs, ys = [], []
    for t0, t1 in time_windows:
        seg = cropTime(press_container, startTime=t0, stopTime=t1)
        xs.extend(seg['x'])
        ys.extend(seg['y'])

    xs, ys = np.asarray(xs), np.asarray(ys)

    if xs.size == 0:
        msg = f"No events in selected window(s)"
        if press_id is not None:
            msg += f" for Press {press_id}"
        print(msg)
        return np.nan, np.nan

    pts    = np.column_stack((xs, ys))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    mask   = labels != -1                       # keep only clustered points

    if not np.any(mask):
        msg = f"No cluster found"
        if press_id is not None:
            msg += f" for Press {press_id}"
        print(msg)
        return np.nan, np.nan

    cx, cy = pts[mask].mean(axis=0)

    # ---------  plotting  ---------
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c='blue', s=2, alpha=0.3, label='Noise')
    plt.scatter(pts[mask, 0], pts[mask, 1],
                c='red', s=5, alpha=0.3, label='Cluster')
    plt.scatter(cx, cy, c='green', s=50, marker='x', label='Centroid')
    title = "Active Pixels and Cluster"
    if press_id is not None:
        title += f" - Press {press_id}"
    title += f" ({camera_name})"
    plt.title(title)
    plt.xlabel("X Pixel Position")
    plt.ylabel("Y Pixel Position")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---------  console output  ---------
    if press_id is not None:
        print(f"Centroid of Activation - Press {press_id}: X = {cx:.2f}, Y = {cy:.2f}")
    else:
        print(f"Centroid of Activation: X = {cx:.2f}, Y = {cy:.2f}")

    return cx, cy



def _dbscan_centroid(xs, ys, eps=10, min_samples=10):
    """
    Apply DBSCAN to filter noise and compute centroid from pixel coordinates.

    Parameters
    ----------
    xs, ys : array-like
        Lists or arrays of x and y pixel values.
    eps : float
        Maximum distance between samples to be considered in the same cluster.
    min_samples : int
        Minimum number of samples to form a dense region.

    Returns
    -------
    (cx, cy) : tuple of float
        Centroid of the largest cluster, or (np.nan, np.nan) if none found.
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    if len(xs) == 0:
        return np.nan, np.nan

    points = np.column_stack((xs, ys))
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    mask = labels != -1

    if not np.any(mask):
        return np.nan, np.nan

    return points[mask].mean(axis=0)



def centroid_from_press(press_container, time_windows=[(0, 3)],
                        eps=10, min_samples=10):
    """
    Crop a single press container to the provided time windows and
    return the centroid of the densest cluster.

    Parameters
    ----------
    press_container : dict
        Single press HDF5 container (with keys 'x', 'y', 'ts', …).
    time_windows : list of tuple
        List of (start, stop) times (seconds) to keep.
    eps, min_samples : DBSCAN parameters.

    Returns
    -------
    cx, cy : float
        Centroid coordinates in pixel space, or (nan, nan) if none.
    """
    xs, ys = [], []
    for t0, t1 in time_windows:
        seg = cropTime(press_container, startTime=t0, stopTime=t1)
        xs.extend(seg['x'])
        ys.extend(seg['y'])

    return _dbscan_centroid(np.asarray(xs), np.asarray(ys),
                            eps=eps, min_samples=min_samples)


#%% Plot for Heat Map Error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from shapely.geometry import Polygon
from scipy.interpolate import griddata  # <-- needed for heatmap interpolation

# Helper – camera field-of-view as a polygon fan
def get_fov_polygon(cam_x, cam_y, fov_deg, fov_len=500,
                    skew_deg=0, num_pts=100):
    half = fov_deg / 2
    angles = np.linspace(-half, half, num_pts)
    angles_rad = np.deg2rad(angles + skew_deg)
    points = [(cam_x, cam_y)] + [
        (cam_x + fov_len * np.cos(a),
         cam_y + fov_len * np.sin(a)) for a in angles_rad
    ]
    return Polygon(points)

# Helper – build sensor polygon with chamfered and rounded corners
def build_sensor_polygon(cx, cy, sensor_w, sensor_h,
                         chamfer=20, round_r=25, n_arc=18):
    left   = cx - sensor_w / 2
    right  = cx + sensor_w / 2
    bottom = cy - sensor_h / 2
    top    = cy + sensor_h / 2

    def quarter_arc(center, start_deg, end_deg):
        ang = np.linspace(start_deg, end_deg, n_arc, endpoint=False)[1:]
        return [(center[0] + round_r*np.cos(np.deg2rad(a)),
                 center[1] + round_r*np.sin(np.deg2rad(a))) for a in ang]

    pts = []
    pts.append((left + round_r, bottom))
    pts.append((right - chamfer, bottom))
    pts.append((right, bottom + chamfer))
    pts.append((right, top - chamfer))
    pts.append((right - chamfer, top))
    pts.append((left + round_r, top))
    pts.extend(quarter_arc((left + round_r, top - round_r), 90, 180))
    pts.append((left, bottom + round_r))
    pts.extend(quarter_arc((left + round_r, bottom + round_r), 180, 270))
    return Polygon(pts)

# Plot camera coverage and overlap + heatmap
def plot_camera_coverage_with_heatmap(po, sensor_width, sensor_height,
                                      p1, p2,
                                      intersections=None,
                                      press_coordinates=None,
                                      fov_camera1=120, fov_camera2=120,
                                      skew_angle1=0, skew_angle2=0,
                                      fov_length=130,
                                      chamfer=20, round_r=25):
    # Sensor polygon
    sensor_poly = build_sensor_polygon(po['xo'], po['yo'],
                                       sensor_width, sensor_height,
                                       chamfer=chamfer,
                                       round_r=round_r)

    # FOV polygons
    fov1 = get_fov_polygon(p1['x1'], p1['y1'], fov_camera1,
                           fov_length, skew_angle1)
    fov2 = get_fov_polygon(p2['x2'], p2['y2'], fov_camera2,
                           fov_length, skew_angle2)

    # Intersections (coverage)
    cov1 = fov1.intersection(sensor_poly)
    cov2 = fov2.intersection(sensor_poly)
    overlap = cov1.intersection(cov2)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.set_xlim([75, -75])
    ax.set_ylim([85, -75])
    ax.set_aspect('equal')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Camera coverage on chamfered & rounded sensor + Error Heatmap")

    # Sensor outline
    ax.add_patch(patches.Polygon(np.array(sensor_poly.exterior.coords),
                                 closed=True, edgecolor='black', facecolor='none',
                                 linewidth=1.4, label='Sensor'))

    # FOV fans
    ax.add_patch(patches.Polygon(np.array(fov1.exterior.coords),
                                 color='red', alpha=0.25, label='Cam 1 FOV'))
    ax.add_patch(patches.Polygon(np.array(fov2.exterior.coords),
                                 color='blue', alpha=0.25, label='Cam 2 FOV'))

    # Overlap
    if not overlap.is_empty:
        ax.add_patch(patches.Polygon(np.array(overlap.exterior.coords),
                                     color='purple', alpha=0.35, label='Overlap'))

    # Camera symbols
    ax.scatter(p1['x1'], p1['y1'], color='red', marker='s', s=60)
    ax.scatter(p2['x2'], p2['y2'], color='blue', marker='s', s=60)

    # ---------- Heatmap inside sensor ------------------------------------
    if intersections is not None and press_coordinates is not None:
        xy_gt = []
        errs = []
        for est, gt in zip(intersections, press_coordinates):
            if est and isinstance(est, dict):
                x_gt, y_gt = gt['x'], gt['y']
                x_est, y_est = est.get('xi'), est.get('yi')
                if x_est is not None and y_est is not None:
                    xy_gt.append([x_gt, y_gt])
                    errs.append(np.hypot(x_gt - x_est, y_gt - y_est))
        if len(xy_gt) > 3:
            xy_gt = np.array(xy_gt)
            errs = np.array(errs)
            rect_x = po['xo'] - sensor_width / 2
            rect_y = po['yo'] - sensor_height / 2
            gx, gy = np.mgrid[rect_x:rect_x + sensor_width:500j,
                              rect_y:rect_y + sensor_height:500j]
            grid = griddata(xy_gt, errs, (gx, gy), method='cubic')
            im = ax.imshow(grid.T,
                           extent=(rect_x, rect_x + sensor_width,
                                   rect_y, rect_y + sensor_height),
                           origin='lower', cmap='RdYlGn_r', alpha=0.85, zorder=5)
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
            cbar.set_label("Localization Error (mm)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
    plt.show()

#%% CMRE Functions

import numpy as np

def build_one_hot_map(gt_x, gt_y, x_edges, y_edges):
    T = np.zeros((len(x_edges)-1, len(y_edges)-1))
    i = np.searchsorted(x_edges, gt_x, side='right') - 1
    j = np.searchsorted(y_edges, gt_y, side='right') - 1
    i = max(0, min(i, T.shape[0]-1))
    j = max(0, min(j, T.shape[1]-1))
    T[i, j] = 1.0
    return T

def build_recon_map(est_x, est_y, x_edges, y_edges, sigma=1.0):
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='ij')
    R = np.exp(-((Xc - est_x)**2 + (Yc - est_y)**2) / (2*sigma**2))
    R /= R.max() if R.max() > 0 else 1
    return R

def compute_CMRE(R, T, SR):
    n_x, n_y = R.shape
    true_i, true_j = np.argwhere(T==1)[0]
    ii, jj = np.indices(R.shape)
    d = np.sqrt((ii-true_i)**2 + (jj-true_j)**2)
    C = np.tanh(3.5 * (R - T)**2)
    W = -1.0 / ((SR/35)**2 + d**2)
    W[true_i, true_j] = +1.0
    return np.sum(W * C)

def global_error(intersections, press_coords, x_edges, y_edges, SR, sigma=1.0):
    cmres = []
    for est, gt in zip(intersections, press_coords):
        if not est or est==0: 
            continue
        R = build_recon_map(est['xi'], est['yi'], x_edges, y_edges, sigma)
        T = build_one_hot_map(gt['x'], gt['y'], x_edges, y_edges)
        cmres.append(compute_CMRE(R, T, SR))
    return np.mean(cmres), np.std(cmres)
