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
os.chdir(os.path.join(dir,'scripts'))
from simulators import VolSimulator, SurfSimulator, MixSimulator



############################################################################
#                              VOLUME SIMULATIONS       
############################################################################

#Check volume labels available in aseg file  
#label_names = mne.get_volume_labels_from_aseg(simulator.fname_aseg)

#regions = ["Left-Cerebellum-Cortex", "Left-Thalamus-Proper","Left-Caudate", "Left-Hippocampus"]
regions = ["Left-Thalamus-Proper"]
amplitude = 1.0
folder = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/thalamic_1nA'

for region in regions:
    
    print(f"------------- Simulating from {region} ----------")
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    sim_folder= os.path.join(folder, f'{region}')

    #Initate  
    simulator = VolSimulator()
    simulator.set_params(output_path=sim_folder)
    simulator.create_info_obj()

    #Generate SRC and FWD for simulations 
    simulator.generate_src(vol_labels=[region], save=True, plot=False)
    #simulator.src

    simulator.generate_fwd(save=True)
    #simulator.fwd
    #simulator.fwd['src']

    #Plot fwd with sources 
    simulator.plot_fwd_with_sources(surface='white')

    ## Loop through patch sizes and simulate ## 
    patch_sizes = [2., 5., 8., 10., 15.]

    for extent in patch_sizes: 

        print(f"--- Running patch size {extent} ---")

        #Generate Label obj to use for simulations defined by label, seed and extent (if seeds=None it will compute center of mass and use that as seed)
        simulator.grow_sim_source_label(labels=region, seeds=None, extents=extent)
        
        #Check vertex positions of full region, grown label and seed 
        seed_pos_lh = simulator.src[0]['rr'][np.where(simulator.src[0]['vertno']==simulator.seeds[region])]
        label_pos_lh = [simulator.src[0]['rr'][v] for v in simulator.src[0]['vertno'] if v in simulator.labels[0].vertices]

        ## FIXME currently just plotting empty squares (after I set show=False)
        Brain = mne.viz.get_brain_class()
        brain = Brain(
            'fsaverage',
            hemi='both',
            surf='white',
            alpha=0.5,
            background='white',
            cortex='low_contrast',
            units='m',
            subjects_dir=simulator.subjects_dir,
        )
        #brain.add_foci(simulator.src[1]['rr'], coords_as_verts=False, color='grey', hemi='rh', alpha=0.4, scale_factor=0.2) #full region 
        #brain.add_foci(simulator.src[0]['rr'], coords_as_verts=False, color='white', hemi='lh', alpha=0.4, scale_factor=0.2) #full region 
        brain.add_foci(label_pos_lh, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in label
        # #brain.add_foci(label_pos_rh, coords_as_verts=False, color='red', hemi='rh', scale_factor=0.2) #vertices in label
        brain.add_foci(seed_pos_lh, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2) #position of seed used to grow label (center of mass)
        # #brain.add_foci(seed_pos_rh, coords_as_verts=False, color='blue', hemi='rh', scale_factor=0.4) #position of seed used to grow label (center of mass)
        brain.save_image(os.path.join(simulator.figure_path, f'source_label_{region}_{extent}.png'))

        #Simualtor raw STCs 
        simulator.create_time_series(amplitude=amplitude, latency=0.02)
        simulator.plot_time_series(save=True, show=False)
        simulator.initiate_sourcesimulator()
        simulator.add_to_sourcesimulator(labels="all") #if all, will add time seires*events for all labels in simulator.labels

        #Simulate raw 
        simulator.sim_raw(add_iir=False, add_eog=True, add_ecg=True)
        simulator.plot_raw(save=True, show=False)

        #Compute evoked 
        simulator.compute_evoked()
        simulator.plot_joint(picks='grad', save=True, show=False)
        simulator.plot_joint(picks='mag', save=True, show=False)



############################################################################
#                              SURFACE SIMULATIONS    
############################################################################
amplitude = 0.1
folder = f'/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/occpitial_{amplitude}nA'
#regions = ['lateraloccipital-lh']
regions = ["ctx-lh-lateraloccipital"]

for region in regions: 
    print(f'--------- Running region {region} ----------')
    if not os.path.exists(folder):
        os.mkdir(folder)

    sim_folder = folder

    #Initate  
    #simulator = SurfSimulator()
    simulator = VolSimulator()
    simulator.set_params(output_path=sim_folder)
    simulator.create_info_obj()

    #Generate SRC and FWD for simulations 
    #simulator.generate_src(save=True, plot=True)
     #simulator.generate_fwd(save=True)
    
    simulator.generate_src(vol_labels=[region], save=True, plot=False)
    #simulator.src

    simulator.generate_fwd(save=True)

    #Plot fwd with sources 
    simulator.plot_fwd_with_sources(surface='white')

    extents = [2., 4., 6., 8.,10.]

    for extent in extents: 

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
        brain.add_foci(label_pos_lh, coords_as_verts=False, color='green', hemi='lh', scale_factor=0.2) #vertices in label
        # #brain.add_foci(label_pos_rh, coords_as_verts=False, color='red', hemi='rh', scale_factor=0.2) #vertices in label
        brain.add_foci(seed_pos_lh, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2) #position of seed used to grow label (center of mass)
        # #brain.add_foci(seed_pos_rh, coords_as_verts=False, color='blue', hemi='rh', scale_factor=0.4) #position of seed used to grow label (center of mass)
        brain.save_image(os.path.join(simulator.figure_path, f'source_label_{region}_{extent}.png'))
        brain.close()

        #Simualtor raw STCs 
        simulator.create_time_series(amplitude=amplitude, latency=0.0)
        simulator.plot_time_series(save=True, show=False)
        simulator.initiate_sourcesimulator()
        simulator.add_to_sourcesimulator(labels="all") #if all, will add time seires*events for all labels in simulator.labels

        #Simulate raw 
        simulator.sim_raw(add_iir=False, add_eog=True, add_ecg=True)
        simulator.plot_raw(save=True, show=False)

        #Compute evoked 
        simulator.compute_evoked()
        simulator.plot_joint(picks='grad', save=True, show=False)
        simulator.plot_joint(picks='mag', save=True, show=False)



############################################################################
#                              SIMS MIXED        
############################################################################

#Check volume labels available in aseg file  
#label_names = mne.get_volume_labels_from_aseg(simulator.fname_aseg)


#vol_regions = ["Left-Cerebellum-Cortex", "Left-Thalamus-Proper"]
regions = ["Left-Thalamus-Proper", "ctx-lh-lateraloccipital"]
#surf_regions = ["lateraloccipital-lh"]
#surf_regions = ["ctx-lh-lateraloccipital"]
amplitude_vol = 1.0
amplitude_surf = 0.1
latency_vol = 0.02 #sec BEFORE surface peak 
folder = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/thalamic_1nA_occipital_01nA'

patch_sizes_surf = [0, 2., 4., 6., 8.]
patch_sizes_vol = [2., 5., 8., 10., 15.]

#for vol in vol_regions: 
print(f"------------- Simulating from {regions} ----------")
if not os.path.exists(folder):
    os.mkdir(folder)

sim_folder = folder

#Initate  
simulator = MixSimulator()
simulator.set_params(output_path=sim_folder)
simulator.create_info_obj()

#Generate SRC and FWD for simulations 
simulator.generate_src(vol_labels=regions, save=True, plot=True)
#simulator.src

simulator.generate_fwd(save=True)
#simulator.fwd
#simulator.fwd['src']

#Plot fwd with sources 
simulator.plot_fwd_with_sources(surface='white')

vol = regions[0]
surf = regions[1]

## Loop through patch sizes and simulate ## 
for vol_extent in patch_sizes_vol:
    for surf_extent in patch_sizes_surf:  
        regions = ["Left-Thalamus-Proper", "ctx-lh-lateraloccipital"]

        print(f"--- Running volume size {vol_extent} ---")
        print(f"-- Surf size = {surf_extent}")

        #Generate Label obj to use for simulations defined by label, seed and extent (if seeds=None it will compute center of mass and use that as seed)
        if surf_extent==0.0: 
            simulator.grow_sim_source_label_vol(labels=vol, seeds=None, extents=vol_extent)
            simulator.source_labels_surf = None
        else: 
            simulator.grow_sim_source_label_vol(labels=regions, seeds=None, extents=[vol_extent, surf_extent])
            simulator.source_labels_surf = [simulator.source_labels_vol[1]]
            simulator.source_labels_vol = [simulator.source_labels_vol[0]]

        #Check vertex positions of full region, grown label and seed 
        label_pos_vol = [simulator.src[0]['rr'][v] for v in simulator.src[0]['vertno'] if v in simulator.source_labels_vol[0].vertices]
        if simulator.source_labels_surf is not None: 
            label_pos_surf = [simulator.src[0]['rr'][v] for v in simulator.src[0]['vertno'] if v in simulator.source_labels_surf[0].vertices]

        ## FIXME currently just plotting empty squares (if I set show=False)
        Brain = mne.viz.get_brain_class()
        brain = Brain(
            'fsaverage',
            hemi='both',
            surf='pial',
            alpha=0.5,
            background='white',
            cortex='low_contrast',
            units='m',
            subjects_dir=simulator.subjects_dir,
        )
        #brain.add_foci(simulator.src[0]['rr'], coords_as_verts=False, color="red", hemi="lh", scale_factor=0.2)
        brain.add_foci(label_pos_vol, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in labes
        if not simulator.source_labels_surf is None: #if extent=0 this will be empty and cannot be added 
            brain.add_foci(label_pos_surf, coords_as_verts=False, color='darkgreen', hemi='lh', scale_factor=0.2) #vertices in label #vertices in label
        #brain.save_image(os.path.join(simulator.figure_path, f'source-label-{vol}_{vol_extent}-{surf}_{surf_extent}.png'))
        brain.save_image(os.path.join(simulator.figure_path, f'source-label-{surf}_{surf_extent}.png'))

        #Simualtor raw STCs 
        simulator.create_time_series(amplitude_surf=amplitude_surf, amplitude_vol=amplitude_vol, latency_vol=latency_vol)
        if not surf_extent==0: 
            simulator.plot_time_series(save=True, show=False, vol=True, surf=True)
        else: 
            simulator.plot_time_series(save=True, show=False, vol=True, surf=False)
        simulator.initiate_sourcesimulator()
        simulator.add_to_sourcesimulator(labels="all") #if all, will add time seires*events for all labels in simulator.labels

        #Simulate raw 
        simulator.sim_raw(add_iir=False, add_eog=True, add_ecg=True)
        simulator.plot_raw(save=True, show=False)

        #Compute evoked 
        simulator.compute_evoked()
        simulator.plot_joint(picks='grad', save=True, show=False)
        simulator.plot_joint(picks='mag', save=True, show=False)




############################################################################
#    COMPARE SIGNAL INCREASE by increasing patch size (with same 
#        amplitude in each dipole) or keeping one dipole and increasing 
#        amplitude      
# - Testing with V1 activation 
# - SNR increase is the same in the two analyses 
# - Analysis A) increasing patch size from 2-15 mm, with amplitude of 0.1
# - Analysis B) using CoM dipole, increasing amplitude from 0.1-1.5 
############################################################################
        

#----------------------- Increasing patch size ----------------##
amplitude = 0.1
folder = os.path.join(dir, f'data/simulations/occpitial_{amplitude}nA_increasing_size')

#regions = ['lateraloccipital-lh']
regions = ["ctx-lh-lateraloccipital"]

/Volumes/Elements/simulation_cortical_omission/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif
/Volumes/Elements/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif

for region in regions: 
    print(f'--------- Running region {region} ----------')
    if not os.path.exists(folder):
        os.mkdir(folder)

    sim_folder = folder

    #Initate  
    simulator = VolSimulator()
    simulator.set_params(output_path=sim_folder)
    simulator.create_info_obj()

    #Generate SRC and FWD for simulations 
    #simulator.generate_src(save=True, plot=True)
     #simulator.generate_fwd(save=True)
    
    simulator.generate_src(vol_labels=[region], save=True, plot=False)
    #simulator.src

    simulator.generate_fwd(save=True)

    #Plot fwd with sources 
    simulator.plot_fwd_with_sources(surface='white')

    extents = [2., 4., 6., 8.,10.]

    for extent in extents: 

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
        brain.add_foci(label_pos_lh, coords_as_verts=False, color='green', hemi='lh', scale_factor=0.2) #vertices in label
        # #brain.add_foci(label_pos_rh, coords_as_verts=False, color='red', hemi='rh', scale_factor=0.2) #vertices in label
        brain.add_foci(seed_pos_lh, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2) #position of seed used to grow label (center of mass)
        # #brain.add_foci(seed_pos_rh, coords_as_verts=False, color='blue', hemi='rh', scale_factor=0.4) #position of seed used to grow label (center of mass)
        brain.save_image(os.path.join(simulator.figure_path, f'source_label_{region}_{extent}.png'))
        brain.close()

        #Simualtor raw STCs 
        simulator.create_time_series(amplitude=amplitude, latency=0.0)
        simulator.plot_time_series(save=True, show=False)
        simulator.initiate_sourcesimulator()
        simulator.add_to_sourcesimulator(labels="all") #if all, will add time seires*events for all labels in simulator.labels

        #Simulate raw 
        simulator.sim_raw(add_iir=False, add_eog=True, add_ecg=True)
        simulator.plot_raw(save=True, show=False)

        #Compute evoked 
        simulator.compute_evoked()
        simulator.plot_joint(picks='grad', save=True, show=False)
        simulator.plot_joint(picks='mag', save=True, show=False)




#COMPUTE SNR from evoked 
extent = evo.split("-")[-2].split("_")[0]
patch_size.append(extent)
evoked = mne.read_evokeds(os.path.join(sim_path, sim, evo), baseline=(None, 0))[0]
inv_fname = evo.replace("-ave.fif", "-inv.fif")
inv_path = os.path.join(recon_path, sim, "mne", inv_fname)
inv = mne.minimum_norm.read_inverse_operator(inv_path)
snr = mne.minimum_norm.estimate_snr(evoked, inv, verbose=None)[0]
snr_max_list.append(snr.max())
snr_mean_list.append(snr.mean())




############################################################################
#                     CHECKING STUFF IN SIMS     
############################################################################
subject = 'fsaverage'
subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
fname_trans = 'fsaverage'
folder = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/test_2nA_increasing_size'


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
df.to_csv(os.path.join(folder, 'list_sources_n_dipoles.csv'))