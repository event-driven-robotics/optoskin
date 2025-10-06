# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 16:13:25 2025

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
    'Zanyar': 'C:/Users/Zanyar/Desktop/Repo',
}

paths_to_data = {
    'sim': 'C:/Users/sim/OneDrive - Fondazione Istituto Italiano Tecnologia/data/Optical_Data_Rectangular/recording_4_9_2024_circular path',
    'sbamford': 'C:/Users/sbamford/OneDrive - Fondazione Istituto Italiano Tecnologia/data/Optical_Data_Rectangular/recording_4_9_2024_circular path',
    'Zanyar': 'C:/Users/Zanyar/Desktop/reording-3-8-241',
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
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 12,
    'figure.dpi': 240
})
#%% Load the dataset

# filePathAndName_camera2 = os.path.join(path_to_data, 'camera2')
container_camera2 = importHdf5(filePathOrName='C:/Users/koola/Desktop/WS/DATA/grid_path2_v2_camera2.hdf5')

# filePathAndName_camera1 = os.path.join(path_to_data, 'camera1')
container_camera1 = importHdf5(filePathOrName='C:/Users/koola/Desktop/WS/DATA/grid_path2_v2_camera1.hdf5')

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
press_coordinates_df = pd.read_csv('C:/Users/Zanyar/Desktop/WS/DATA/snake_grid_press_positions.csv')*1000.0
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


containers_by_press_camera1_listt = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera1'],
    press_timings_camera1,
    press_duration
)



containers_by_press_camera2_listt = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera2'],
    press_timings_camera2,
    press_duration
)


containers_by_no_press_camera1_listt = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera1'],
    gap_timings_camera1,
    gap_duration
)



containers_by_no_press_camera2_listt = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera2'],
    gap_timings_camera2,
    gap_duration
)
#%%
for i in range(289):
    print(len(containers_by_press_camera2_listt['repetition_1'][i]['ts']))

#%% Cropping the no presses

# Parameters for cropping
gap_start_time = 0.8
gap_end_time = 1.4

# Dictionary to store cropped data
containers_by_no_press_camera1_list = {}
containers_by_no_press_camera2_list = {}

# Iterate through all repetitions
for repetition_key, press_list in containers_by_no_press_camera1_listt.items():
    cropped_presses = []
    
    # Iterate through each press in the repetition
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=gap_start_time, stopTime=gap_end_time)
        cropped_presses.append(cropped_press)
    
    # Store the cropped data
    containers_by_no_press_camera1_list[repetition_key] = cropped_presses

# Iterate through all repetitions
for repetition_key, press_list in containers_by_no_press_camera2_listt.items():
    cropped_presses = []
    
    # Iterate through each press in the repetition
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=gap_start_time, stopTime=gap_end_time)
        cropped_presses.append(cropped_press)
    
    # Store the cropped data
    containers_by_no_press_camera2_list[repetition_key] = cropped_presses


#%% Cropping the presses

# Parameters for cropping
start_time_c1 = 0     #0.7
end_time_c1 =  3     #1.15
start_time_c2 = 0     #0.6
end_time_c2 =  3     #1.05

# Dictionary to store cropped data
containers_by_press_camera1_list = {}
containers_by_press_camera2_list = {}

# Iterate through all repetitions
for repetition_key, press_list in containers_by_press_camera1_listt.items():
    cropped_presses = []
    
    # Iterate through each press in the repetition
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=start_time_c1, stopTime=end_time_c1)
        cropped_presses.append(cropped_press)
    
    # Store the cropped data
    containers_by_press_camera1_list[repetition_key] = cropped_presses

# Iterate through all repetitions
for repetition_key, press_list in containers_by_press_camera2_listt.items():
    cropped_presses = []
    
    # Iterate through each press in the repetition
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=start_time_c2, stopTime=end_time_c2)
        cropped_presses.append(cropped_press)
    
    # Store the cropped data
    containers_by_press_camera2_list[repetition_key] = cropped_presses


#%% Function to calculate average event rate (events per second)
def calculate_average_event_rate(containers_by_press_list, press_duration):
    total_events = 0
    total_presses = 0
    
    for repetition_key, press_list in containers_by_press_list.items():
        for press_container in press_list:
            total_events += len(press_container['ts'])  # Count event timestamps
            total_presses += 1
    
    avg_event_rate = (total_events / total_presses) / press_duration if total_presses > 0 else 0
    return avg_event_rate

# Define press and gap durations (update these values as needed)
press_duration = 0.5  # Duration of each press in seconds
gap_duration = 0.5  # Duration of each gap in seconds

# Calculate average event rate for press and gap periods
avg_event_rate_press_c1 = calculate_average_event_rate(containers_by_press_camera1_list, press_duration)
avg_event_rate_press_c2 = calculate_average_event_rate(containers_by_press_camera2_list, press_duration)
avg_event_rate_gap_c1 = calculate_average_event_rate(containers_by_no_press_camera1_list, gap_duration)
avg_event_rate_gap_c2 = calculate_average_event_rate(containers_by_no_press_camera2_list, gap_duration)

# Print results
print("Average Event Rate (events per second):")
print(f"Camera 1 - Press: {avg_event_rate_press_c1:.2f} events/sec")
print(f"Camera 2 - Press: {avg_event_rate_press_c2:.2f} events/sec")
print(f"Camera 1 - Gap: {avg_event_rate_gap_c1:.2f} events/sec")
print(f"Camera 2 - Gap: {avg_event_rate_gap_c2:.2f} events/sec")

#%% Finding the Center of Activity for Presses 1 to 250

# Import necessary libraries
from sklearn.cluster import DBSCAN
import numpy as np
import gc  # Garbage collector for memory optimization

# Define the specific time windows to keep
time_windows = [(0, 3)]

# List to store centroids for each press
centroids_camera1 = []
centroids_camera2 = []

# Process Presses 1 to 250 for Camera 1
for i, press_container in enumerate(containers_by_press_camera1_listt['repetition_1']):
    if i >= 250:
        print("\nReached Press 250. Stopping processing for Camera 1.")
        break

    try:
        # Initialize storage for cropped data in each press
        cropped_data_x = []
        cropped_data_y = []

        # Crop each press for the given time windows
        for start_time, end_time in time_windows:
            cropped_segment = cropTime(press_container, startTime=start_time, stopTime=end_time)
            cropped_data_x.extend(cropped_segment['x'])
            cropped_data_y.extend(cropped_segment['y'])

        # Convert lists to NumPy arrays
        x_values = np.array(cropped_data_x)
        y_values = np.array(cropped_data_y)

        # If no events, store NaN centroid and continue
        if len(x_values) == 0:
            print(f"No events for Press {i+1} (Camera 1)")
            centroids_camera1.append((np.nan, np.nan))
            continue

        # Apply DBSCAN Clustering
        pixel_positions = np.column_stack((x_values, y_values))
        clustering = DBSCAN(eps=10, min_samples=10).fit(pixel_positions)
        labels = clustering.labels_

        # Remove noise (-1 is noise label in DBSCAN)
        filtered_positions = pixel_positions[labels != -1]

        if len(filtered_positions) == 0:
            print(f"No cluster found for Press {i+1} (Camera 1)")
            centroids_camera1.append((np.nan, np.nan))
            continue

        # Compute centroid
        centroid = np.mean(filtered_positions, axis=0)
        centroids_camera1.append(centroid)
        print(f"Centroid for Press {i+1} (Camera 1): X = {centroid[0]:.2f}, Y = {centroid[1]:.2f}")

    except MemoryError:
        print(f"MemoryError encountered at Press {i+1} (Camera 1). Skipping.")
        centroids_camera1.append((np.nan, np.nan))

    finally:
        gc.collect()  # Free up memory

# Convert results to DataFrame for easy analysis
centroids_df_camera1 = pd.DataFrame(centroids_camera1, columns=['X', 'Y'])


#%% Process Presses 1 to 250 for Camera 2
for i, press_container in enumerate(containers_by_press_camera2_listt['repetition_1']):
    if i >= 250:
        print("\nReached Press 250. Stopping processing for Camera 1.")
        break

    try:
        # Initialize storage for cropped data in each press
        cropped_data_x = []
        cropped_data_y = []

        # Crop each press for the given time windows
        for start_time, end_time in time_windows:
            cropped_segment = cropTime(press_container, startTime=start_time, stopTime=end_time)
            cropped_data_x.extend(cropped_segment['x'])
            cropped_data_y.extend(cropped_segment['y'])

        # Convert lists to NumPy arrays
        x_values = np.array(cropped_data_x)
        y_values = np.array(cropped_data_y)

        # If no events, store NaN centroid and continue
        if len(x_values) == 0:
            print(f"No events for Press {i+1} (Camera 1)")
            centroids_camera2.append((np.nan, np.nan))
            continue

        # Apply DBSCAN Clustering
        pixel_positions = np.column_stack((x_values, y_values))
        clustering = DBSCAN(eps=10, min_samples=10).fit(pixel_positions)
        labels = clustering.labels_

        # Remove noise (-1 is noise label in DBSCAN)
        filtered_positions = pixel_positions[labels != -1]

        if len(filtered_positions) == 0:
            print(f"No cluster found for Press {i+1} (Camera 1)")
            centroids_camera2.append((np.nan, np.nan))
            continue

        # Compute centroid
        centroid = np.mean(filtered_positions, axis=0)
        centroids_camera2.append(centroid)
        print(f"Centroid for Press {i+1} (Camera 1): X = {centroid[0]:.2f}, Y = {centroid[1]:.2f}")

    except MemoryError:
        print(f"MemoryError encountered at Press {i+1} (Camera 1). Skipping.")
        centroids_camera2.append((np.nan, np.nan))

    finally:
        gc.collect()  # Free up memory

# Convert results to DataFrame for easy analysis
centroids_df_camera2 = pd.DataFrame(centroids_camera2, columns=['X', 'Y'])

# Store the centroids for all presses (1 to 250)
centroids_df_camera1 = pd.DataFrame(centroids_camera1, columns=['X', 'Y'])
centroids_df_camera2 = pd.DataFrame(centroids_camera2, columns=['X', 'Y'])

print("\nCentroids for Camera 1:")
print(centroids_df_camera1)

print("\nCentroids for Camera 2:")
print(centroids_df_camera2)

# Save the centroids to CSV files
centroids_df_camera1.to_csv('centroids_camera1_press1_to_250.csv', index=False)
centroids_df_camera2.to_csv('centroids_camera2_press1_to_250.csv', index=False)

print("\nCentroids saved as 'centroids_camera1_press1_to_250.csv' and 'centroids_camera2_press1_to_250.csv'.")

#%% Extract X-values into Lists
x_activity1_smooth_curve = centroids_df_camera1['X'].tolist()
x_activity2_smooth_curve = centroids_df_camera2['X'].tolist()

# Print first 10 values for verification
print("\nFirst 10 X values for Camera 1 (x_activity1_smooth_curve):", x_activity1_smooth_curve[:10])
print("First 10 X values for Camera 2 (x_activity2_smooth_curve):", x_activity2_smooth_curve[:10])


x_activity1_smooth_curve[0] = np.nan
x_activity1_smooth_curve[3] = np.nan
x_activity1_smooth_curve[6] = np.nan
x_activity1_smooth_curve[13] = np.nan
x_activity1_smooth_curve[11] = np.nan
x_activity1_smooth_curve[22] = np.nan
x_activity1_smooth_curve[50] = np.nan
x_activity1_smooth_curve[188] = np.nan
x_activity1_smooth_curve[201] = np.nan
x_activity1_smooth_curve[205] = np.nan
x_activity1_smooth_curve[237] = np.nan

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
# List to store all intersection points
intersections = []

# Loop through the activities and calculate the press positions
for i in range(250):  
    # Get the activities for Camera 1 and Camera 2
    pixel_activity1 = x_activity1_smooth_curve[i]
    pixel_activity2 = x_activity2_smooth_curve[i]
    
    if np.isnan(pixel_activity2) or np.isnan(pixel_activity1):
        print(f"Skiping triangulation for press {i + 1} due to no visibility")
        intersections.append(0)
        continue
    
    # Calculate theta for Camera 1 and Camera 2
    theta1 = calculate_theta_exponential(pixel_activity1, pixel_mid_camera1, fov_camera1 / 2, skew_angle_camera1, W_camera1)
    theta2 = calculate_theta_exponential(pixel_activity2, pixel_mid_camera2, fov_camera2 / 2, skew_angle_camera2, W_camera2)

    # Calculate distances based on tangent and theta
    d11 = d1 * tan(theta1)
    d22 = d2 * tan(theta2)
    
    # Initial guess (You can adjust these values if needed)
    initial_guess_points = [po['xo'] + d11, po['yo'] + d11, po['xo'] + d22, po['yo'] + d22]


    # Solve the system
    solution = fsolve(equations, initial_guess_points, args=(po, p1, p2, d1, d2, d11, d22, theta1, theta2))


    # Extract values
    x11, y11, x22, y22 = solution
    
    # Slope calculation
    m1 = (y11 - p1['y']) / (x11 - p1['x'] )
    m2 = (y22 - p2['y']) / (x22 - p2['x'] )


    
    # Calculate lines
    ly1 = point_slope_line(m1, p1['x'], p1['y'])
    ly2 = point_slope_line(m2, p2['x'], p2['y'])
    
    # Find intersection (press position)
    initial_guess_lines = (p1['x'] + p2['x']) / 2
    intersection = find_intersection_fsolve(ly1, ly2, initial_guess_lines)

    # Check if the intersection point is inside the skin area
    if is_inside_skin(intersection['xi'], intersection['yi'], po, skin_width, skin_height):
        intersections.append(intersection)
    else:
        print(f"Intersection {i + 1} is outside the skin area and will be ignored.")
        intersections.append(None)  # Store None if outside the skin
  
    # intersections.append(intersection)
    
#%%
import importlib
importlib.reload(rsf)


#%% Calculate Error Metrics
rmse, std = rsf.calculate_rmse_and_error_metrics(intersections, press_coordinates[:250])
print(f" RMSE : {rmse}, \n STD : {std} ")

#%% differential evolution

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import gc
from scipy.optimize import fsolve, dual_annealing
from sklearn.cluster import DBSCAN


# ------------------- Objective Function -------------------

def objective_function(params):
    global p1, p2, skew_angle_camera1, skew_angle_camera2, W_camera1, W_camera2
    
    # Extract parameters
    p1_x, p1_y, p2_x, p2_y, skew_angle_camera1, skew_angle_camera2, W_camera1, W_camera2 = params
    
    # Update global parameters
    p1 = {'x': p1_x, 'y': p1_y}
    p2 = {'x': p2_x, 'y': p2_y}

    # Compute intersections
    intersections = []
    for i in range(250):  
        pixel_activity1 = x_activity1_smooth_curve[i]
        pixel_activity2 = x_activity2_smooth_curve[i]

        if np.isnan(pixel_activity1) or np.isnan(pixel_activity2):
            intersections.append(None)
            continue

        theta1 = calculate_theta_exponential(pixel_activity1, 320, 120 / 2, skew_angle_camera1, W_camera1)
        theta2 = calculate_theta_exponential(pixel_activity2, 320, 120 / 2, skew_angle_camera2, W_camera2)

        d1 = calculate_distance(po, p1)
        d2 = calculate_distance(po, p2)
        d11, d22 = d1 * tan(theta1), d2 * tan(theta2)

        initial_guess_points = [po['xo'] + d11, po['yo'] + d11, po['xo'] + d22, po['yo'] + d22]
        
        try:
            solution = fsolve(equations, initial_guess_points, args=(po, p1, p2, d1, d2, d11, d22, theta1, theta2))
        except RuntimeError:
            intersections.append(None)
            continue

        x11, y11, x22, y22 = solution
        m1, m2 = (y11 - p1['y']) / (x11 - p1['x']), (y22 - p2['y']) / (x22 - p2['x'])
        ly1, ly2 = point_slope_line(m1, p1['x'], p1['y']), point_slope_line(m2, p2['x'], p2['y'])

        initial_guess_lines = (p1['x'] + p2['x']) / 2
        intersection = find_intersection_fsolve(ly1, ly2, initial_guess_lines)

        if is_inside_skin(intersection['xi'], intersection['yi'], po, 100, 100):
            intersections.append(intersection)
        else:
            intersections.append(None)

    # Compute RMSE
    rmse, _ = rsf.calculate_rmse_and_error_metrics(intersections, press_coordinates[:250])
    print(f"*****************RMSE: {rmse}***********************")
    return rmse

# ------------------- Optimization Setup -------------------

# Define parameter bounds
bounds = [
    (40, 55),  # p1_x
    (-50, -30),  # p1_y
    (40, 55),  # p2_x
    (30, 40),  # p2_y
    (-5, 5),  # skew_angle_camera1
    (-5, 5),  # skew_angle_camera2
    (-1, 1),  # W_camera1
    (-1, 1)   # W_camera2
]

# Run Simulated Annealing Optimization
result = dual_annealing(objective_function, bounds, maxiter=1000)

# Extract optimized parameters
optimal_params = result.x
optimal_rmse = result.fun

# Print results
print("\nOptimized Parameters (Using Simulated Annealing):")
print(f"Camera 1 Position: ({optimal_params[0]}, {optimal_params[1]})")
print(f"Camera 2 Position: ({optimal_params[2]}, {optimal_params[3]})")
print(f"Skew Angle Camera 1: {optimal_params[4]}")
print(f"Skew Angle Camera 2: {optimal_params[5]}")
print(f"W Correction Camera 1: {optimal_params[6]}")
print(f"W Correction Camera 2: {optimal_params[7]}")
print(f"\nOptimized RMSE: {optimal_rmse:.4f}")

 
#%% Visualization for Estimated Presses Connected to Their Ground Truth Positions

rsf.visualize_press_estimation(
    intersections,  # Estimated press positions
    press_coordinates,  # Ground truth press coordinates
    po,  # Skin center
    {'x1': p1['x'], 'y1': p1['y']},  # Convert `p1` format to expected format
    {'x2': p2['x'], 'y2': p2['y']},  # Convert `p2` format to expected format
    fov_camera1,  # FOV of Camera 1
    fov_camera2,  # FOV of Camera 2
    rmse  # Error metric
)






