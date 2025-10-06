# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 20:40:18 2025

@author: mkoolani
"""

#%% Preliminaries

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
import math

# Setting paths based on the username
username = os.getlogin()

paths_to_repos = {
    'sim': 'C:/repos',
    'sbamford': 'C:/repos',
    'mkoolani': 'C:/Users/mkoolani/Desktop/Repo',
}

paths_to_data = {
    'sim': 'C:/Users/sim/OneDrive - Fondazione Istituto Italiano Tecnologia/data/Optical_Data_Rectangular/recording_4_9_2024_circular path',
    'sbamford': 'C:/Users/sbamford/OneDrive - Fondazione Istituto Italiano Tecnologia/data/Optical_Data_Rectangular/recording_4_9_2024_circular path',
    'mkoolani': 'C:/Users/mkoolani/Desktop/reording-3-8-241',
}
path_to_data = paths_to_data.get(username)

# Add the current directory to the Python path
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)

sys.path.insert(0, os.path.join(paths_to_repos[username], 'bimvee'))

# Import modules
from bimvee.importAe import importAe
from bimvee.split import cropTime
from bimvee.plot import plot
from bimvee.player import Player
from bimvee.plot import plotEventRate
from bimvee.plot import plotDvsContrast
from bimvee.importHdf5 import importHdf5
from sklearn.neighbors import KernelDensity

from functions import rectangular_skin_functions as rsf

#%%
import importlib
importlib.reload(rsf)
#%% Set global font size for all of the plots as the same as the text in the writing
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.titlesize': 10,
    'figure.dpi': 240
})
#%% Load the dataset

# filePathAndName_camera2 = os.path.join(path_to_data, 'camera2')
container_camera2 = importHdf5(filePathOrName='C:/Users/mkoolani/Desktop/Optical_Skin/Optical-Skin-Rect/data_recording_v2_21.2.25/round2/grid_path2_v2_camera22.hdf5')

# filePathAndName_camera1 = os.path.join(path_to_data, 'camera1')
container_camera1 = importHdf5(filePathOrName='C:/Users/mkoolani/Desktop/Optical_Skin/Optical-Skin-Rect/data_recording_v2_21.2.25/round2/grid_path2_v2_camera1.hdf5')

#%% Cropping
from bimvee.split import cropSpaceTime

events_cropped = {}

minX = 0
minY = 200

maxX = 640
maxY = 355

minX_c1 = 0
minY_c1 = 185

maxX_c1 = 640
maxY_c1 = 360

# Crop events
events_cropped['camera1'] = cropSpaceTime(container_camera1, minX=minX_c1, minY=minY_c1, maxX=maxX_c1, maxY=maxY_c1, zeroSpace=False)
events_cropped['camera2'] = cropSpaceTime(container_camera2, minX=minX, minY=minY, maxX=maxX, maxY=maxY, zeroSpace=False)

#%%
# plt.close('all')
plotEventRate(events_cropped['camera2'])

#%% Play events
plt.close('all')
player = Player(container_camera1)

#%% import press coordinates
press_coordinates_df = pd.read_csv('C:/Users/mkoolani/Desktop/Optical_Skin/Optical-Skin-Rect/data_recording_v2_21.2.25/snake_grid_press_positions.csv')*1000.0
press_coordinates_df = press_coordinates_df.dropna().reset_index(drop=True)
press_coordinates = press_coordinates_df[['x', 'y']].to_dict(orient='records')


#%% Parameters
params_rect = {
    'skin_width': 100,  # mm
    'skin_height': 100,  # mm
    # workspace of the robot
    'workspace_x': 120,   # x direction in mm
    'workspace_y': 180,  # y direction in mm
    #'camera_x': (4, -42), # mm; order is camera2, camera1
    #'camera_y': (55, 55), # mm; order is camera2, camera1
    'camera_x': (45, 45), # mm; order is camera2, camera1  'camera_x': (10, 63)
    'camera_y': (-35, 46), # mm; order is camera2, camera1   'camera_y': (71, 17)
    #'camera_angle': (45, 90), # degs; order is camera2, camera1
    'camera_angle': (90, 0), # degs; order is camera2, camera1
   'camera_fov_camera1': 120,  # degrees
   'camera_fov_camera2': 120 ,  # degrees
    'num_cols': 640,
    'cols_per_bin': 10,
    'skin_center': (10, 2)  # mm (x, y)
}
params_rect['num_bins'] = int(params_rect['num_cols'] / params_rect['cols_per_bin'])


#%% Function to split data by press with specific press times and offsets

# Parameters for Multiple Repetitions
num_repetitions = 1
delay_between_repetitions = 3.0  # seconds
num_presses_per_repetition = 289  # Total presses per repetition
press_duration = 3  # Each press takes 3.0 seconds (0.5 + 1.0 + 1.5)
press_duration += 1/289
gap_duration = 2


def calculate_press_timings_precise(first_press_time, num_repetitions, num_presses_per_repetition, press_duration, delay_between_repetitions):

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


# Initial Times camera1
start_time_camera1 = 9.4
initial_presses_duration = 6.0  # 3 presses at 2.0 seconds each
initial_delay = 3.0  # Delay after initial presses
first_press_time_camera1 = start_time_camera1 + initial_presses_duration + initial_delay

# Calculate Press Timings
press_timings_camera1 = calculate_press_timings_precise(
    first_press_time_camera1,  # Start after initial presses + delay
    num_repetitions,
    num_presses_per_repetition,
    press_duration,
    delay_between_repetitions
)



start_time_camera2 = 7.7
first_press_time_camera2 = start_time_camera2 + 6 + 3   
# Calculate Press Timings
press_timings_camera2 = calculate_press_timings_precise(
    first_press_time_camera2,  # Start after initial presses + delay
    num_repetitions,
    num_presses_per_repetition,
    press_duration,
    delay_between_repetitions
)


# Calculate Press Timings for the gaps

# Initial Times camera1
start_time_camera1 = 9.4
initial_presses_duration = 6.0  # 3 presses at 2.0 seconds each
initial_delay = 3.0  # Delay after initial presses
first_gap_time_camera1 = start_time_camera1 + initial_presses_duration + initial_delay + 1.5

gap_timings_camera1 = calculate_press_timings_precise(
    first_gap_time_camera1,  # Start after initial presses + delay
    num_repetitions,
    num_presses_per_repetition,
    press_duration,
    delay_between_repetitions
)



start_time_camera2 = 7.7
first_gap_time_camera2 = start_time_camera2 + 6 + 3  + 1.5  
# Calculate Press Timings
gap_timings_camera2 = calculate_press_timings_precise(
    first_gap_time_camera2,  # Start after initial presses + delay
    num_repetitions,
    num_presses_per_repetition,
    press_duration,
    delay_between_repetitions
)






#%% Function to Split Data by Specific Times for Multiple Repetitions
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


containers_by_press_camera1_list = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera1'],
    press_timings_camera1,
    press_duration
)



containers_by_press_camera2_list = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera2'],
    press_timings_camera2,
    press_duration
)


#%%
from bimvee.plot import plotEventRate

cmap = plt.get_cmap('winter')
num_colours = cmap.N
plt.close('all')

fig, ax = plt.subplots()
for cont in containers_by_press_camera1_list['repetition_1']:
    plotEventRate(cont, axes=ax, periods=[0.1])
    plt.ylim(0, 300000)

lines = ax.get_lines()
num_lines = len(lines)
for line_idx, line in enumerate(lines):
    line.set_color(cmap(int(line_idx / (num_lines - 1) * (num_colours - 1))))
plt.title("all presses for repetition 1", fontsize=10)

#%%
from bimvee.plot import plotEventRate
# plt.close('all')

plotEventRate(containers_by_press_camera1_list['repetition_1'][75], periods=[0.01])
plt.title("Press number 75 splited- camera1", fontsize=15)
plt.xlabel("Time(s)", fontsize=10)
plt.ylabel("Rate (events/s)", fontsize=10)


#%% Play events
plt.close('all')
player = Player(containers_by_press_camera2_list['repetition_1'][65])

#%%
# Define reduction factors
reduction_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Dictionary to store results
centroids_dict = {'camera1': {}, 'camera2': {}}

# Iterate over reduction factors
for reduction_factor in reduction_factors:
    print(f"\nProcessing reduction factor: {reduction_factor}")

    # Apply event reduction
    containers_by_press_camera1_list_reduced = rsf.reduce_events_seed(containers_by_press_camera1_list, reduction_factor)
    containers_by_press_camera2_list_reduced = rsf.reduce_events_seed(containers_by_press_camera2_list, reduction_factor)

    # Compute centroids
    centroids_df_camera1 = rsf.compute_activity_centroids(
        containers_by_press_camera1_list_reduced['repetition_1'], 
        num_presses=250, 
        start_time=0, 
        end_time=3,
        eps=10, 
        min_samples=1
        
    )

    centroids_df_camera2 = rsf.compute_activity_centroids(
        containers_by_press_camera2_list_reduced['repetition_1'], 
        num_presses=250, 
        start_time=0, 
        end_time=3,
        eps=10, 
        min_samples=1
    )

    # Store results in the dictionary
    centroids_dict['camera1'][reduction_factor] = centroids_df_camera1
    centroids_dict['camera2'][reduction_factor] = centroids_df_camera2

    # Save the centroids to CSV files
    centroids_df_camera1.to_csv(f'centroids_camera1_reduction_{reduction_factor}.csv', index=False)
    centroids_df_camera2.to_csv(f'centroids_camera2_reduction_{reduction_factor}.csv', index=False)

print("\nProcessing completed. Centroid lists are stored in 'centroids_dict'.")




#%% Extract X-values into Lists
# Dictionary to store X-values for each reduction factor
x_activity_dict = {'camera1': {}, 'camera2': {}}

# Indices to set as NaN
nan_indices = []

# Iterate over reduction factors and extract X values
for reduction_factor in reduction_factors:
    print(f"\nExtracting X-values for reduction factor: {reduction_factor}")

    # Extract X values for camera 1
    x_activity1 = centroids_dict['camera1'][reduction_factor]['X'].tolist()
    x_activity2 = centroids_dict['camera2'][reduction_factor]['X'].tolist()

    # Set specified indices to NaN (only if index exists in the list)
    for idx in nan_indices:
        if idx < len(x_activity1):
            x_activity1[idx] = np.nan
        if idx < len(x_activity2):
            x_activity2[idx] = np.nan

    # Store in the dictionary
    x_activity_dict['camera1'][reduction_factor] = x_activity1
    x_activity_dict['camera2'][reduction_factor] = x_activity2

print("\nX-values extraction completed. Data is stored in 'x_activity_dict'.")



#%% reduction for 10 different seeds

import numpy as np

# Define reduction factors
reduction_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
num_seeds = 10  # Number of different runs with different randomness

# Dictionary to store results
centroids_dict = {'camera1': {}, 'camera2': {}}

# Run multiple times with different seeds
for seed in range(num_seeds):
    np.random.seed(seed)  # Set a different seed for each iteration

    print(f"\nProcessing with Seed: {seed}")

    for reduction_factor in reduction_factors:
        print(f"  Processing reduction factor: {reduction_factor}")

        # Apply event reduction with different seeds
        containers_by_press_camera1_list_reduced = rsf.reduce_events_seed(
            containers_by_press_camera1_list, reduction_factor
        )
        containers_by_press_camera2_list_reduced = rsf.reduce_events_seed(
            containers_by_press_camera2_list, reduction_factor
        )

        # Compute centroids
        centroids_df_camera1 = rsf.compute_activity_centroids(
            containers_by_press_camera1_list_reduced['repetition_1'], 
            num_presses=250, 
            start_time=0, 
            end_time=3,
            eps=10, 
            min_samples=1
        )

        centroids_df_camera2 = rsf.compute_activity_centroids(
            containers_by_press_camera2_list_reduced['repetition_1'], 
            num_presses=250, 
            start_time=0, 
            end_time=3,
            eps=10, 
            min_samples=1
        )

        # Initialize nested dictionaries if not present
        if reduction_factor not in centroids_dict['camera1']:
            centroids_dict['camera1'][reduction_factor] = {}
            centroids_dict['camera2'][reduction_factor] = {}

        # Store results per seed
        centroids_dict['camera1'][reduction_factor][seed] = centroids_df_camera1
        centroids_dict['camera2'][reduction_factor][seed] = centroids_df_camera2

        # Save the centroids to CSV files
        centroids_df_camera1.to_csv(f'centroids_camera1_reduction_{reduction_factor}_seed_{seed}.csv', index=False)
        centroids_df_camera2.to_csv(f'centroids_camera2_reduction_{reduction_factor}_seed_{seed}.csv', index=False)

print("\nProcessing completed. Centroid lists are stored in 'centroids_dict'.")


#%% Extract X-values into Lists
# Dictionary to store X-values for each reduction factor and seed
x_activity_dict = {'camera1': {}, 'camera2': {}}

# Indices to set as NaN
nan_indices = []

# Iterate over reduction factors and seeds to extract X values
for reduction_factor in reduction_factors:
    x_activity_dict['camera1'][reduction_factor] = {}
    x_activity_dict['camera2'][reduction_factor] = {}

    for seed in range(num_seeds):
        print(f"\nExtracting X-values for reduction factor {reduction_factor}, seed {seed}")

        # Extract X values for camera 1
        x_activity1 = centroids_dict['camera1'][reduction_factor][seed]['X'].tolist()
        x_activity2 = centroids_dict['camera2'][reduction_factor][seed]['X'].tolist()

        # Set specified indices to NaN (only if index exists in the list)
        for idx in nan_indices:
            if idx < len(x_activity1):
                x_activity1[idx] = np.nan
            if idx < len(x_activity2):
                x_activity2[idx] = np.nan

        # Store in the dictionary with seed as key
        x_activity_dict['camera1'][reduction_factor][seed] = x_activity1
        x_activity_dict['camera2'][reduction_factor][seed] = x_activity2

print("\nX-values extraction completed. Data is stored in 'x_activity_dict'.")




#%% Function to check if a press is inside the skin area

def is_inside_skin(x, y, skin_center, skin_width, skin_height):
    x_min = skin_center['xo'] - skin_width / 2
    x_max = skin_center['xo'] + skin_width / 2
    y_min = skin_center['yo'] - skin_height / 2
    y_max = skin_center['yo'] + skin_height / 2
    return x_min <= x <= x_max and y_min <= y <= y_max

#%% Triangulation

from scipy.optimize import fsolve
from scipy.optimize import brentq


skew_angle_camera1 = -1.92   #-1.98      # Skew angle in degrees (positive for left, negative for right)
skew_angle_camera2 = -1.47 #-2.01    
W_camera1 = -0.17 #-0.37
W_camera2 = 0.074  #0.036


def calculate_theta_exponential(pixel, pixel_mid, fov_half, skew_angle, W):
    theta_base = (pixel - pixel_mid) / pixel_mid * fov_half
    nonlinearity_factor = np.exp(-W * np.abs((pixel - pixel_mid) / pixel_mid))
    theta = theta_base * nonlinearity_factor
    theta_normalized = theta
    theta_with_skew = theta_normalized + skew_angle
    return theta_with_skew


EPSILON = 1e-6  # Small threshold to avoid division by zero

def tan(degree):
    radians = math.radians(degree)
    return math.tan(radians) if abs(degree) < 89.9 else math.tan(math.radians(89.9))


# Function to calculate the sinuse of a degree
def sin(degree):
    radians = math.radians(degree)
    sinuse = math.sin(radians)
    return sinuse

# Define line equation
def point_slope_line(m, x1, y1):
    def line_eq(x):
        return m * (x - x1) + y1
    return line_eq

# Function to calculate the intersection of two lines
def find_intersection_fsolve(ly1, ly2, initial_guess):
    def line_diff(x):
        return ly1(x) - ly2(x)
    x_intersection = fsolve(line_diff, initial_guess, xtol=1e-6)[0]
    y_intersection = ly1(x_intersection)
    return {'xi': x_intersection, 'yi': y_intersection}

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    x1, y1 = list(point1.values())[:2]
    x2, y2 = list(point2.values())[:2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Define the equations to solve
def equations(vars, po, p1, p2, d1, d2, d11, d22, theta1, theta2):
    x11, y11, x22, y22 = vars

    eq1 = (x11 - po['xo'])**2 + (y11 - po['yo'])**2 - d11**2
    eq2 = (x22 - po['xo'])**2 + (y22 - po['yo'])**2 - d22**2
    eq3 = (x11 - p1['x'])**2 + (y11 - p1['y'])**2 - (d11 / sin(theta1))**2
    eq4 = (x22 - p2['x'])**2 + (y22 - p2['y'])**2 - (d22 / sin(theta2))**2

    return [eq1, eq2, eq3, eq4]





# Camera parameters
fov_camera1 = 120        #params_rect['camera_fov_front']  # FOV of Camera 1 in degrees
fov_camera2 = 120        #params_rect['camera_fov_corner']  # FOV of Camera 2 in degrees
sensor_size = 640  # Sensor pixel size (640 pixels)
pixel_mid_camera1 = 320 # Middle pixel of Camera 1
pixel_mid_camera2 = 320  # Middle pixel of Camera 2

# Physical distances
po = {'xo': 10, 'yo': 2}    # Center of the skin in mm
p1 =  {'x': 44.25, 'y': -31.98}    # Position of Camera 1
p2 =  {'x': 49.71, 'y': 39.81}    # Position of Camera2

# Define the skin area
skin_width = 100
skin_height = 100

# Calculate distances from cameras to the center
d1 = calculate_distance(po, p1)
d2 = calculate_distance(po, p2)

#%% run the triangulation
# Dictionary to store triangulation results for each reduction factor
triangulation_results_seed9 = {}

# Iterate over all reduction factors
for reduction_factor in reduction_factors:
    print(f"\nRunning triangulation for reduction factor: {reduction_factor}")

    # List to store intersection points for the current reduction factor
    intersections = []

    # Loop through the presses (up to 250)
    for i in range(250):
        # Get the X activity values for Camera 1 and Camera 2
        pixel_activity1 = x_activity_dict['camera1'][reduction_factor][9][i]
        pixel_activity2 = x_activity_dict['camera2'][reduction_factor][9][i]

        if np.isnan(pixel_activity1) or np.isnan(pixel_activity2):
            print(f"Skipping triangulation for press {i + 1} due to missing visibility")
            intersections.append(None)
            continue
        
        # Calculate theta for Camera 1 and Camera 2
        theta1 = calculate_theta_exponential(pixel_activity1, pixel_mid_camera1, fov_camera1 / 2, skew_angle_camera1, W_camera1)
        theta2 = calculate_theta_exponential(pixel_activity2, pixel_mid_camera2, fov_camera2 / 2, skew_angle_camera2, W_camera2)

        # Calculate distances based on tangent and theta
        d11 = d1 * tan(theta1)
        d22 = d2 * tan(theta2)

        # Initial guess for solving equations
        initial_guess_points = [po['xo'] + d11, po['yo'] + d11, po['xo'] + d22, po['yo'] + d22]

        # Solve for intersection points
        solution = fsolve(equations, initial_guess_points, args=(po, p1, p2, d1, d2, d11, d22, theta1, theta2))

        # Extract values
        x11, y11, x22, y22 = solution

        # Slope calculations
        m1 = (y11 - p1['y']) / (x11 - p1['x'])
        m2 = (y22 - p2['y']) / (x22 - p2['x'])

        # Define line equations
        ly1 = point_slope_line(m1, p1['x'], p1['y'])
        ly2 = point_slope_line(m2, p2['x'], p2['y'])

        # Find intersection (press position)
        initial_guess_lines = (p1['x'] + p2['x']) / 2
        intersection = find_intersection_fsolve(ly1, ly2, initial_guess_lines)

        # Check if the intersection is inside the skin area
        # if is_inside_skin(intersection['xi'], intersection['yi'], po, skin_width, skin_height):
        #     intersections.append(intersection)
        # else:
        #     print(f"Intersection {i + 1} is outside the skin area, ignoring it.")
        #     intersections.append(None)
        intersections.append(intersection)

    # Store results in dictionary
    triangulation_results_seed9[reduction_factor] = intersections

print("\nTriangulation completed for all reduction factors.")




#%% Compute RMSE and STD for each reduction factor
error_metrics9 = {}

for reduction_factor in reduction_factors:
    intersections = triangulation_results_seed9[reduction_factor]
    rmse, std = rsf.calculate_rmse_and_error_metrics(intersections, press_coordinates[:250])
    error_metrics9[reduction_factor] = {'RMSE': rmse, 'STD': std}
    print(f"Reduction Factor {reduction_factor} -> RMSE: {rmse}, STD: {std}")

print("\nError calculations completed.")



#%%



pass_rates9 = {}


fail_distance = 0 

for i, reduction_factor in enumerate(reduction_factors):
    print(f"\nAnalyzing Reduction Factor: {reduction_factor}")

    # Get estimated intersections for this reduction factor
    estimated_presses = triangulation_results_seed9[reduction_factor]

    # Compute distances
    distances = np.full(250, np.nan)  

    for j, actual_press in enumerate(press_coordinates[:250]):
        if estimated_presses[j] is None:
            continue  

        x_actual, y_actual = actual_press['x'], actual_press['y']
        x_estimated, y_estimated = estimated_presses[j]['xi'], estimated_presses[j]['yi']


        distances[j] = np.sqrt((x_estimated - x_actual) ** 2 + (y_estimated - y_actual) ** 2)

  
    max_dist = np.nanmax(distances) if np.nanmax(distances) > 0 else 1  
    distances[np.isnan(distances)] = max_dist  

    if fail_distance == 0:
        fail_distance = np.percentile(distances, 95)  


    pass_rate = np.sum(distances < fail_distance) / 250  
    pass_rates9[reduction_factor] = pass_rate

    print(f"1/(2^{i}) -> Pass Rate: {pass_rate:.4f}")

#%%

import numpy as np

# Dictionary to store averaged RMSE and pass rates for each reduction factor
average_rmse_per_factor = {}
average_pass_rate_per_factor = {}

# Iterate over all reduction factors
for reduction_factor in reduction_factors:
    rmse_values = []
    pass_rate_values = []

    # Collect RMSE and pass rates from all 10 seeds
    for seed in range(10):  # 10 seeds from 0 to 9
        rmse_values.append(globals()[f'error_metrics{seed}'][reduction_factor]['RMSE'])
        pass_rate_values.append(globals()[f'pass_rates{seed}'][reduction_factor])

    # Compute the average RMSE and Pass Rate for this reduction factor
    average_rmse_per_factor[reduction_factor] = np.mean(rmse_values)
    average_pass_rate_per_factor[reduction_factor] = np.mean(pass_rate_values)

    # Print results
    print(f"Reduction Factor {reduction_factor}: Average RMSE = {average_rmse_per_factor[reduction_factor]:.4f}, Average Pass Rate = {average_pass_rate_per_factor[reduction_factor]:.4f}")

print("\n✅ Averaging completed!")


#%%
# Extract reduction factors and pass rates
filtered_reduction_factors = [rf for rf in average_pass_rate_per_factor.keys() if rf != 6]
filtered_pass_rates = [average_pass_rate_per_factor[rf] for rf in filtered_reduction_factors]

# Create evenly spaced x values
x_positions = range(len(filtered_reduction_factors))

# Plot Pass Rate vs Reduction Factor
plt.figure(figsize=(8, 5))
plt.plot(x_positions, filtered_pass_rates, marker='o', linestyle='-', color='b', label='Pass Rate')

# Formatting the plot
plt.xlabel("Reduction Factor")
plt.ylabel("Pass Rate")
plt.title("Pass Rate vs Reduction Factor - Average on 10 seeds")
plt.grid(True, linestyle="--", linewidth=0.3)

# Set custom x-ticks to ensure equal spacing
plt.xticks(x_positions, labels=[str(rf) for rf in filtered_reduction_factors])

plt.legend()
plt.show()


#%%

# Extract reduction factors and RMSE values, ignoring factor 6
filtered_reduction_factors = [rf for rf in error_metrics.keys() if rf != 6]
filtered_rmse_values = [error_metrics[rf]['RMSE'] for rf in filtered_reduction_factors]

# Create evenly spaced x values
x_positions = range(len(filtered_reduction_factors))

# Plot RMSE vs Reduction Factor
plt.figure(figsize=(8, 5))
plt.plot(x_positions, filtered_rmse_values, marker='o', linestyle='-', color='b', label='RMSE')

# Formatting the plot
plt.xlabel("Reduction Factor", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.title("RMSE vs Reduction Factor", fontsize=14)
plt.grid(True, linestyle="--", linewidth=0.5)

# Set custom x-ticks to ensure equal spacing
plt.xticks(x_positions, labels=[str(rf) for rf in filtered_reduction_factors])

# Start y-axis from zero
plt.ylim(0, max(filtered_rmse_values) + 1)

plt.legend()
plt.show()

