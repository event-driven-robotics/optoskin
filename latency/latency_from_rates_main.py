"""
Latency analysis for event-based opto-tactile skin
author sbamford
insert permissive license

requirements: bimvee to import data
"""


#%% Preliminaries
import matplotlib.pyplot as plt
import os
import sys
import numpy as np


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

sys.path.insert(0, r"C:\Users\Zanyar\Desktop\Repo\optical_skin\rectangular")

# Import modules from bimvee
from bimvee.split import cropTime
from bimvee.importHdf5 import importHdf5
from functions import rectangular_skin_functions as rsf
import importlib
from bimvee.split import cropSpaceTime

#%% import
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
container_camera2 = importHdf5(filePathOrName='C:/Users/Zanyar/Desktop/WS/DATA/grid_path2_v2_camera2.hdf5')
# filePathAndName_camera1 = os.path.join(path_to_data, 'camera1')
container_camera1 = importHdf5(filePathOrName='C:/Users/Zanyar/Desktop/WS/DATA/grid_path2_v2_camera1.hdf5')

#%% Cropping
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
first_gap_time_camera2 = start_time_camera2 + 6 + 3 + 1.5

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
    half_duration = press_duration / 2  # To center the split data around the press
    for repetition_index, press_timings in enumerate(press_timings_per_repetition):
        repetition_key = f'repetition_{repetition_index + 1}'
        containers_by_press[repetition_key] = []
        for press_time in press_timings:
            # Center the split data on the press time
            start_time = press_time - half_duration
            end_time = press_time + half_duration
            press_container = cropTime(container, startTime=start_time, stopTime=end_time)
            containers_by_press[repetition_key].append(press_container)
    return containers_by_press

containers_by_press_camera1_list_full = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera1'], press_timings_camera1, press_duration
)
containers_by_press_camera2_list_full = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera2'], press_timings_camera2, press_duration
)
containers_by_no_press_camera1_list_full = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera1'], gap_timings_camera1, gap_duration
)
containers_by_no_press_camera2_list_full = split_data_by_specific_times_multiple_repetitions_aligned(
    events_cropped['camera2'], gap_timings_camera2, gap_duration
)

#%% Cropping the no presses
# Parameters for cropping
gap_start_time = 0.5 # 0.8
gap_end_time = 1.9  # 1.25

# Dictionary to store cropped data
containers_by_no_press_camera1_list = {}
containers_by_no_press_camera2_list = {}

# Iterate through all repetitions
for repetition_key, press_list in containers_by_no_press_camera1_list_full.items():
    cropped_presses = []
    # Iterate through each press in the repetition
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=gap_start_time, stopTime=gap_end_time)
        cropped_presses.append(cropped_press)
    # Store the cropped data
    containers_by_no_press_camera1_list[repetition_key] = cropped_presses

for repetition_key, press_list in containers_by_no_press_camera2_list_full.items():
    cropped_presses = []
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=gap_start_time, stopTime=gap_end_time)
        cropped_presses.append(cropped_press)
    containers_by_no_press_camera2_list[repetition_key] = cropped_presses

#%% Cropping the presses
# Parameters for cropping
start_time_c1 = 1.5 - 0.85  # 0.7
end_time_c1 = 1.5 + 0.85   # 1.15
start_time_c2 = 1.5 - 0.85   # 0.6
end_time_c2 = 1.5 + 0.85    # 1.05

# Dictionary to store cropped data
containers_by_press_camera1_list = {}
containers_by_press_camera2_list = {}

for repetition_key, press_list in containers_by_press_camera1_list_full.items():
    cropped_presses = []
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=start_time_c1, stopTime=end_time_c1)
        cropped_presses.append(cropped_press)
    containers_by_press_camera1_list[repetition_key] = cropped_presses

for repetition_key, press_list in containers_by_press_camera2_list_full.items():
    cropped_presses = []
    for press_container in press_list:
        cropped_press = cropTime(press_container, startTime=start_time_c2, stopTime=end_time_c2)
        cropped_presses.append(cropped_press)
    containers_by_press_camera2_list[repetition_key] = cropped_presses

#%% 
containers_by_press_camera1 = rsf.reduce_events_seed(containers_by_press_camera1_list, 1)
containers_by_press_camera2 = rsf.reduce_events_seed(containers_by_press_camera2_list, 1)
containers_by_no_press_camera1 = rsf.reduce_events_seed(containers_by_no_press_camera1_list, 1)
containers_by_no_press_camera2 = rsf.reduce_events_seed(containers_by_no_press_camera2_list, 1)


#%% Load all data for repetition 1

full = {'trial_event_times': [],
        'bg_event_times':[]}

d1 = containers_by_press_camera1['repetition_1']
d2 = containers_by_press_camera2['repetition_1']
for r1, r2 in zip(d1, d2):
    full['trial_event_times'].append(np.sort(np.concatenate([r1['ts'], r2['ts']])))
    
d1 = containers_by_no_press_camera1['repetition_1']
d2 = containers_by_no_press_camera2['repetition_1']
for r1, r2 in zip(d1, d2):
    full['bg_event_times'].append(np.sort(np.concatenate([r1['ts'], r2['ts']])))

#%% chop 0.5s off beginning of bg segments, because there are tails of press-related activity there

chop = 0.5

new_bgs = []
for bg in full['bg_event_times']:
       new_bgs.append((bg - chop)[bg >= chop])
full['bg_event_times'] = new_bgs

#%% Reduce data by 1024x

from latency_from_rates_analysis import random_reduce

red = {'trial_event_times': [],
        'bg_event_times':[]}

for key in ['trial_event_times', 'bg_event_times']:
    for elem in full[key]:
        red[key].append(random_reduce(elem, 1024))

#%% Visualise data

import matplotlib.pyplot as plt

plt.close('all')
fig, axes = plt.subplots(1, 2)

ax = axes[0]
for ev in full['trial_event_times']:
    ax.hist(ev, 50, alpha=0.2)
#ax.relim()
#ax.autoscale_view(scalex=False, scaley=True)
y0, y1 = ax.get_ylim()

ax = axes[1]
for ev in full['bg_event_times']:
    ax.hist(ev, 50, alpha=0.2)
    ax.set_ylim([y0, y1])


#%% Visualise reduced data

import matplotlib.pyplot as plt

plt.close('all')
fig, axes = plt.subplots(1, 2)

ax = axes[0]
for ev in red['trial_event_times']:
    ax.hist(ev, 50, alpha=0.2)
#ax.relim()
#ax.autoscale_view(scalex=False, scaley=True)
y0, y1 = ax.get_ylim()

ax = axes[1]
for ev in red['bg_event_times']:
    ax.hist(ev, 50, alpha=0.2)
    ax.set_ylim([y0, y1])


#%% Analyse full data

trials = trial_event_times = full['trial_event_times']
bgs = bg_event_times = full['bg_event_times']

from latency_from_rates_analysis import preset_balanced, run_pipeline

params = preset_balanced()
results = run_pipeline(
    trials, bgs, params=params,
    h_values=np.linspace(17, 19, 3),    # sweep 5..20
    latency_max=0.10,                  # 20 ms “hit” window
    target_tpr=0.95
)


#%% Analyse reduced data

trials = trial_event_times = red['trial_event_times']
bgs = bg_event_times = red['bg_event_times']

from latency_from_rates_analysis import preset_balanced, run_pipeline

params = preset_balanced()

# play with params here
#params.alpha =
#params.smooth_sigma = 0.005
#params.dt = 
#params.hysteresis_bins = 

results = run_pipeline(
    trials, bgs, params=params,
    h_values=np.linspace(3, 8, 6),    # sweep 5..20
    latency_max=0.1,
    target_tpr=0.85
)

#%% Final plot

plt.close('all')


from latency_from_rates_plots import plot_side_by_side
grid = results['grid']
R = results['R']
med = results['rate_median']
band = results['rate_band']
latencies= results['latencies']

plot_side_by_side(grid, R, med, band, latencies, label='press,')

from latency_from_rates_analysis import build_rate_curves
grid, R, med, band = build_rate_curves(bg_event_times, params)
plot_side_by_side(grid, R, med, band, latencies, y_floor=1e-1, ax='already there', colour='r', label='no press,')

lat5 = results['latency_5_95'][0] - results['median_latency']
lat95 = results['latency_5_95'][1] - results['median_latency']
plt.axvline(lat5, ls="--", lw=2)
plt.axvline(lat95, ls="--", lw=2, label='Detection latency 5-95%')
plt.legend(loc='upper right', fontsize=5.7)  
plt.show()
