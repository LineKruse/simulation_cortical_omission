"""
- Must be run from conda environment "mne", otherwise segmentation for getting label vertices fails with "segmentation fault", 
  unsure what the issue is. 
"""



import mne 
import os 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
mne.viz.set_browser_backend("matplotlib")  # or "qt"
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.simulation.metrics import (
    cosine_score,
    f1_score,
    peak_position_error,
    precision_score,
    recall_score,
    region_localization_error,
    spatial_deviation_error,
)
from functools import partial

dir = os.getcwd()
from simulators import VolSimulator, SurfSimulator, MixSimulator
dir = dir.replace('/scripts','')


############################################################################
#                              RUN SIMULATIONS       
############################################################################
folder = os.path.join(dir, f'data/simulations/OPMs/occpitial_w_add_hoc_noise')

amplitudes = [0.1]
regions = ["ctx-lh-lateraloccipital"]
extents = [2., 4., 6., 8.,10.]

for region in regions: 
    print(f'--------- Running region {region} ----------')
    if not os.path.exists(folder):
        os.mkdir(folder)

    sim_folder = folder

    #Initate  
    simulator = VolSimulator()
    simulator.set_params(output_path=sim_folder, sensor_array='opm')
    simulator.create_info_obj()

    #Generate src     
    simulator.generate_src(vol_labels=[region], save=True, plot=False)

    #Generate fwd 
    simulator.generate_fwd(save=True)

    #Plot fwd with sources 
    simulator.plot_fwd_with_sources(surface='white')

    for extent in extents: 

        print(f'- Running patch extent {extent}')

        #Generate Label obj to use for simulations defined by label, seed and extent (if seeds=None it will compute center of mass and use that as seed)
        #simulator.grow_sim_source_label(label_regex=region, location='center', extent=extent)
        simulator.grow_sim_source_label(labels=region, seeds=None, extents=extent)

        seed_pos_lh = simulator.src[0]['rr'][np.where(simulator.src[0]['vertno']==simulator.seeds[region])]
        label_pos_lh = [simulator.src[0]['rr'][v] for v in simulator.src[0]['vertno'] if v in simulator.labels[0].vertices]

        #Check vertex positions of full region, grown label and seed 
        Brain = mne.viz.get_brain_class()
        brain = Brain(
            'fsaverage',
            hemi='both',
            surf='white',
            alpha=0.5,
            background='black',
            cortex='low_contrast',
            units='m',
            subjects_dir=simulator.subjects_dir
        )
        brain.add_foci(label_pos_lh, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in label
        # #brain.add_foci(label_pos_rh, coords_as_verts=False, color='red', hemi='rh', scale_factor=0.2) #vertices in label
        brain.add_foci(seed_pos_lh, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2) #position of seed used to grow label (center of mass)
        # #brain.add_foci(seed_pos_rh, coords_as_verts=False, color='blue', hemi='rh', scale_factor=0.4) #position of seed used to grow label (center of mass)
        brain.save_image(os.path.join(simulator.figure_path, f'source_label_{region}_{extent}.png'))
        brain.close()

        for amplitude in amplitudes: 
            print(f'- Amplitude {amplitude}')

            #Simualtor raw STCs 
            simulator.create_time_series(amplitude=amplitude, latency=0.0)
            simulator.plot_time_series(save=True, show=False)
            simulator.initiate_sourcesimulator()
            simulator.add_to_sourcesimulator(labels="all") #if all, will add time seires*events for all labels in simulator.labels

            #Simulate raw 
            simulator.sim_raw(add_iir=False, add_eog=False, add_ecg=False) #FIXME currently crashing if adding eog 
            simulator.plot_raw(save=True, show=False)

            #Compute evoked 
            simulator.compute_evoked()
            #simulator.plot_joint(picks='grad', save=True, show=False)
            simulator.plot_joint(picks='mag', save=True, show=False)



""" 

############################################################################
#                     CHECKING/TESTING STUFF IN SIMS     
############################################################################
subject = 'fsaverage'
subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
fname_trans = 'fsaverage'
folder = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/test_2nA_increasing_size'

################## TESITNG SETUP of INFO STRUCTURE WIHT OPM SENSOR ARRAY #############
import pickle
opm_fname = '/Volumes/Elements/simulation_cortical_omission/data/OPM/fsaverage_OPM_alpha1_single_axis-info.fif'
opm_obj = mne.io.read_info(opm_fname)


mne_fname = '/Volumes/Elements/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif'
mne_info = mne.io.read_raw_fif(mne_fname).info

mne.viz.plot_alignment(
    opm_obj, 
    dig=False, 
    eeg=False,
    surfaces=[],
    meg=['helmet','sensors'],
    coord_frame='meg'
)
mne.viz.set_3d_view(fig, azimuth=50, elevation=90, distance=0.5)

################## PLOTTING FWD WITH SOURCES #################

#region = "Left-Caudate"
#region = "Left-Hippocampus"
#region = "Left-Thalamus-Proper"
#region = "Left-Cerebellum-Cortex"
region = "Left-Occipital"
region_path = os.path.join(folder, region)
filename = [f for f in os.listdir(region_path) if f.endswith("fwd.fif")][0]
fwd = mne.read_forward_solution(os.path.join(region_path, filename))

fig = mne.viz.create_3d_figure(size=(600, 400))
# Plot the cortex
mne.viz.plot_alignment(
    subject=subject, 
    subjects_dir=subjects_dir,
    trans=fname_trans,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)
# Show the three dipoles defined at each location in the source space
mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    trans=fname_trans,
    fwd=fwd,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)
mne.viz.set_3d_view(figure=fig, azimuth=180, distance=1, focalpoint="auto")


################## CHECKING N DIPOLES PER SIM #################
regions = os.listdir(folder)
extents = [2., 5., 10., 15]

region_list = []
extent_list = []
n_vertices_list = []

for region in regions: 
    region_path = os.path.join(folder, region)
    stc_files = [f for f in os.listdir(region_path) if f.endswith(".stc")]
    for file in stc_files: 
        hemi = "lh" if file.endswith("-lh.stc") else "rh"
        region_list.append(region + "_" + hemi)
        extent_list.append(file.split("-")[-2].split("_")[0])

        stc = mne.read_source_estimate(os.path.join(region_path, file))
        if hemi=="lh": 
            n_vert = len(stc.vertices[0])
        else: 
            n_vert = len(stc.vertices[1])
        n_vertices_list.append(n_vert)
        
df = pd.DataFrame({"region":region_list, 
                           "extent":extent_list,
                           "amplitude": np.repeat(2, len(region_list)),
                           "n_source_vertices": n_vertices_list})
df['extent'] = df['extent'].astype(float)
df = df.sort_values(by=["region","extent"], ascending=True)
df.to_csv(os.path.join(folder, 'list_sources_n_dipoles.csv')) """