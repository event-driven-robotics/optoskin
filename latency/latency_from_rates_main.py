"""
Latency analysis for event-based opto-tactile skin
author sbamford
insert permissive license

requirements: bimvee to import data
"""

import numpy as np
from bimvee.importHdf5 import importHdf5

#%% Load all data for repetition 1

file = "path_to/all_data.hdf5"

data = importHdf5(file)

full = {'trial_event_times': [],
        'bg_event_times':[]}

d1 = data['press_camera1']['repetition_1']
d2 = data['press_camera2']['repetition_1']
for r1, r2 in zip(d1, d2):
    full['trial_event_times'].append(np.sort(np.concatenate([r1['ts'], r2['ts']])))
    
d1 = data['no_press_camera1']['repetition_1']
d2 = data['no_press_camera2']['repetition_1']
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

from latency_from_rates_roc_presets import preset_balanced, run_pipeline

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

from latency_from_rates_roc_presets import build_rate_curves
grid, R, med, band = build_rate_curves(bg_event_times, params)
plot_side_by_side(grid, R, med, band, latencies, y_floor=1e-1, ax='already there', colour='r', label='no press,')

lat5 = results['latency_5_95'][0] - results['median_latency']
lat95 = results['latency_5_95'][1] - results['median_latency']
plt.axvline(lat5, ls="--", lw=2)
plt.axvline(lat95, ls="--", lw=2, label='Detection latency 5-95%')
