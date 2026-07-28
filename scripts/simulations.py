
"""
- Must be run from conda environment "mne", otherwise segmentation for getting label vertices fails with "segmentation fault", 
  unsure what the issue is. 
"""



import mne 
import os 
import numpy as np 
import pandas as pd
import pickle
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
#os.chdir(os.path.join(dir, 'scripts'))
from simulators import Simulator
dir = dir.replace('/scripts','')


def run_sims(folder, sensor_type, amplitude_vol=None, amplitude_surf=None, vol_region=None, surf_region=None, extents_vol=[None], extents_surf=[None]):
    if sensor_type=="squids":
        noise_mag = 2.5114551244341402e-14 #for squid mags (noise floor = 3.0 ft/sqr(Hz))
        noise_grad = 2.6289863571141204e-12 #for squid grads (noise floor = 3.0 ft/sqr(Hz))
    elif sensor_type=="opm":
        noise_mag = 5.059862832414419e-14 #for opm mags (noise floor = 6.0 ft/sqr(Hz))
        noise_grad=None

    plot_added_noise = True #jsut doing this once, and then disabling (first iteration of patch extent)

    print(f'--------- Running vol region {vol_region} ----------')
    print(f'--------- Running surf region {surf_region} ----------')
    if not os.path.exists(folder):
        os.mkdir(folder)

    sim_folder = folder

    #Initate  
    simulator = Simulator()
    simulator.set_params(output_path=sim_folder, sensor_array=sensor_type, n_axis=n_axis)
    simulator.create_info_obj()

    #Generate src     
    if vol_region and surf_region is not None: 
        simulator.generate_src(vol_labels=[vol_region, surf_region], save=True, plot=False)
    elif vol_region is None: 
        simulator.generate_src(vol_labels=[surf_region], save=True, plot=False)
    elif surf_region is None: 
        simulator.generate_src(vol_labels=[vol_region], save=True, plot=False)
    else:
        raise KeyError("Source region must be provided")

    #Generate fwd 
    simulator.generate_fwd(save=True)

    #Plot fwd with sources 
    simulator.plot_fwd_with_sources(surface='white')

    #Info dict 
    simulation_info = dict()
    i=1

    for vol_extent in extents_vol: 
        for surf_extent in extents_surf: 

            print(f'- Running volume patch extent {vol_extent}')
            print(f'- Running surface patch extent {surf_extent}')

            #Generate Label obj to use for simulations defined by label, seed and extent (if seeds=None it will compute center of mass and use that as seed)
            if not vol_region is None: 
                simulator.grow_sim_source_label(labels=vol_region, seeds=None, extent=vol_extent, label_type='vol')
                vol_seed_pos_lh = simulator.src[0]['rr'][np.where(simulator.src[0]['vertno']==simulator.seeds[vol_region])]
                vol_label_pos_lh = [simulator.src[0]['rr'][v] for v in simulator.src[0]['vertno'] if v in simulator.source_labels_vol[0].vertices]
            if not surf_region is None: 
                simulator.grow_sim_source_label(labels=surf_region, seeds=None, extent=surf_extent, label_type='surf')
                surf_seed_pos_lh = simulator.src[0]['rr'][np.where(simulator.src[0]['vertno']==simulator.seeds[surf_region])]
                surf_label_pos_lh = [simulator.src[0]['rr'][v] for v in simulator.src[0]['vertno'] if v in simulator.source_labels_surf[0].vertices]



            #Check vertex positions of full region, grown label and seed 
            Brain = mne.viz.get_brain_class()
            brain = Brain(
                'fsaverage',
                hemi='both',
                surf='white',
                alpha=0.2,
                background='white',
                cortex='low_contrast',
                units='m',
                subjects_dir=simulator.subjects_dir
            )
            if not vol_region is None: 
                brain.add_foci(vol_label_pos_lh, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in label
                brain.add_foci(vol_seed_pos_lh, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2) #position of seed used to grow label (center of mass)
            if not surf_region is None: 
                brain.add_foci(surf_label_pos_lh, coords_as_verts=False, color='darkgreen', hemi='lh', scale_factor=0.2) #vertices in label
                brain.add_foci(surf_seed_pos_lh, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2)
            
            fig_name = f"{vol_region}_{vol_extent}_{surf_region}_{surf_extent}"
            fig_name = fig_name.replace("_None", "")
            fig_name = fig_name.replace("None", "")
            brain.save_image(os.path.join(simulator.figure_path, f'source_label_{fig_name}.png'))
            brain.close()

            #Simualtor raw STCs 
            simulator.create_time_series(amplitude_surf=amplitude_surf, amplitude_vol=amplitude_vol)
            simulator.plot_time_series(save=True, show=False)
            simulator.initiate_sourcesimulator()
            simulator.add_to_sourcesimulator(labels="all") #if all, will add time seires*events for all labels in simulator.labels

            #Simulate raw 
            simulator.sim_raw(add_iir=False, add_eog=True, add_ecg=True, noise_mag=noise_mag, noise_grad=noise_grad) 
            simulator.plot_raw(save=True, show=False)
            if plot_added_noise: 
                simulator.plot_added_noise(save=True, show=True) #OBS! this function also computes ICA on EOG and ECG components and plots
                plot_added_noise=False

            #Compute evoked 
            simulator.compute_evoked()
            
            simulator.plot_joint(picks='mag', save=True, show=False)
            if sensor_type=='squids':
                simulator.plot_joint(picks='grad', save=True, show=False)

            #Plot PSDs 
            simulator.plot_raw_psd(show=False, save=True)

            #Gather relevant info about simulations 
            sim_dict = dict()
            sim_dict['folder'] = folder
            sim_dict['vol_region'] = vol_region
            sim_dict['surf_region'] = surf_region
            sim_dict['vol_extent'] = vol_extent
            sim_dict['surf_extent'] = surf_extent
            sim_dict['vol_amplitude'] = amplitude_vol
            sim_dict['surf_amplitude'] = amplitude_surf
            sim_dict['seeds'] = simulator.seeds
            sim_dict['n_vert_source_vol'] = len(simulator.source_labels_vol[0].vertices)
            sim_dict['source_vert_vol'] = simulator.source_labels_vol[0].vertices
            sim_dict['n_vert_source_surf'] = len(simulator.source_labels_surf[0].vertices)
            sim_dict['source_vert_surf'] = simulator.source_labels_surf[0].vertices
            sim_dict['sensor_array'] = sensor_type
            sim_dict['mag_noise_std'] = noise_mag
            sim_dict['grad_noise_std'] = noise_grad
            sim_dict['tstep'] = simulator.tstep
            
            simulation_info[f'sim_{i}'] = sim_dict
            i += 1

    fname_info_dict = os.path.join(folder, 'sim_info.pkl')
    with open(fname_info_dict, 'wb') as f:
        pickle.dump(simulation_info, f)



############################################################################
#                         RUN SIMULAITONS 
############################################################################

run_thalamic_squid = False
run_thalamic_opm = False 
run_v1_squid = False
run_v1_opm = False 
run_thalamic_v1_squid = True 


if run_thalamic_squid: 
    folder = os.path.join(dir, f'data/simulations/SQUIDs/thalamic')
    sensor_type="squids"
    n_axis=1 #only used if sensor_type='opm'

    vol_region = "Left-Thalamus-Proper"
    amplitude_vol = 1.0
    extents_vol = [2., 4., 6., 8.,10.,12.]

    run_sims(folder, sensor_type, amplitude_vol=amplitude_vol, vol_region=vol_region, extents_vol=extents_vol)
    
if run_thalamic_opm: 
    folder = os.path.join(dir, f'data/simulations/OPMs/thalamic')
    sensor_type="opm"
    n_axis=1 #only used if sensor_type='opm'

    vol_region = "Left-Thalamus-Proper"
    amplitude_vol = 1.0
    extents_vol = [2., 4., 6., 8.,10.,12.]

    run_sims(folder, sensor_type, amplitude_vol=amplitude_vol, vol_region=vol_region, extents_vol=extents_vol)
    
if run_v1_squid: 
    folder = os.path.join(dir, f'data/simulations/SQUIDs/occipital')
    sensor_type="squids"
    n_axis=1 #only used if sensor_type='opm'
    
    surf_region = 'ctx-lh-lateraloccipital'
    amplitude_surf = 0.1
    extents_surf = [2., 4., 6., 8.,10.]

    run_sims(folder, sensor_type, amplitude_surf=amplitude_surf, surf_region=surf_region, extents_surf=extents_surf)
    
if run_v1_opm: 
    folder = os.path.join(dir, f'data/simulations/OPMs/occipital')
    sensor_type="opm"
    n_axis=1 #only used if sensor_type='opm'
    
    surf_region = 'ctx-lh-lateraloccipital'
    amplitude_surf = 0.1
    extents_surf = [2., 4., 6., 8.,10.]

    run_sims(folder, sensor_type, amplitude_surf=amplitude_surf, surf_region=surf_region, extents_surf=extents_surf)

if run_thalamic_v1_squid: 
    folder = os.path.join(dir, f'data/simulations/SQUIDs/thalamic_occipital')
    sensor_type="squids"
    n_axis=1 #only used if sensor_type='opm'

    vol_region = "Left-Thalamus-Proper"
    amplitude_vol = 1.0
    extents_vol = [2., 4., 6., 8.,10.,12.]
    
    surf_region = 'ctx-lh-lateraloccipital'
    amplitude_surf = 0.1
    extents_surf = [0., 2., 4., 6., 8.,10.]

    run_sims(folder, sensor_type, amplitude_vol=amplitude_vol, amplitude_surf=amplitude_surf, vol_region=vol_region, surf_region=surf_region, extents_vol=extents_vol, extents_surf=extents_surf)


 
"""
############################################################################
#                     CHECKING/TESTING STUFF IN SIMS     
############################################################################
subject = 'fsaverage'
subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
#fname_trans = 'fsaverage'
folder = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/test_2nA_increasing_size'

################## TESITNG SETUP of INFO STRUCTURE WIHT OPM SENSOR ARRAY #############
import pickle
import mne
from mne.utils._bunch import NamedInt
opm_fname = '/Volumes/Elements/simulation_cortical_omission/data/OPM/fsaverage_OPM_alpha1_single_axis-info.fif'
opm_obj = mne.io.read_info(opm_fname) #orig: 107, MEG 0113 - MEG 1022

opm2_fname = '/Volumes/Elements/simulation_cortical_omission/data/OPM/fsaverage_OPM_alpha1_dual_axis-info.fif'
opm2_obj = mne.io.read_info(opm2_fname)

bem_fname='/Volumes/Elements/simulation_cortical_omission/data/freesurfer/subjects/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif'
trans_fname = '/Volumes/Elements/simulation_cortical_omission/data/OPM/fsaverage_OPM_head_mri-trans.fif'
mne_fname = '/Volumes/Elements/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif'
mne_info = mne.io.read_raw_fif(mne_fname).info

for ch in opm_obj['chs']:
    ch['coil_type'] = NamedInt("SQ20950N", 3024)


for ch in opm2_obj['chs']:
    ch['coil_type'] = NamedInt("SQ20950N", 3024)


fig = mne.viz.plot_alignment(
    opm_obj, 
    dig=False, 
    trans = trans_fname,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    eeg=False,
    bem=bem_fname, 
    surfaces=("head", "pial"),
    #meg=['helmet','sensors'],
    meg=['sensors'],
    coord_frame='meg'
)
mne.viz.set_3d_view(fig, azimuth=50, elevation=90, distance=0.5)

fig = mne.viz.plot_alignment(
    opm2_obj, 
    dig=False, 
    trans = trans_fname,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    eeg=False,
    bem=bem_fname, 
    surfaces=("head", "pial"),
    #meg=['helmet','sensors'],
    meg=['sensors'],
    coord_frame='meg'
)
mne.viz.set_3d_view(fig, azimuth=50, elevation=90, distance=0.5)


opm2_mod = opm2_obj.copy()

#Trying a method from chat #NB: issue here, as it uses the same rotation for all sensors (but htey are not equally placed on the z-axis)
for idx in range(0, int(len(opm2_mod['chs'])/2)):
    #Get origianl coordiantes and resphape into 3x3 (rep x, y and z directions)
    orig_coords = opm2_obj['chs'][idx]['loc'][3:] #testing on MEG 0123
    R = np.array(orig_coords).reshape((3, 3),axis=1)

    # 3. Create a 90-degree rotation matrix around the Z-axis (Z-axis stays fixed)
    # Change the angle to np.pi/2 for clockwise or -np.pi/2 for counter-clockwise
    theta = np.pi / 2 
    R_rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    # 4. Multiply matrices to get the new orientation
    R_new = R @ R_rot
    # 5. Flatten back to 9 coordinates and assign
    new_coords = R_new.flatten().tolist()
    opm2_mod['chs'][idx]['loc'][3:] = new_coords

#Trying individual rotation axes - another method from chat 
from scipy.spatial.transform import Rotation as R 
opm2_mod = opm2_obj.copy()
for ch in opm2_mod['chs']:
    ch['coil_type'] = NamedInt("SQ20950N", 3024)

for idx in range(int(len(opm2_mod['chs'])/2),len(opm2_mod['chs'])): #rotating the last 107 channels (first 107 have same pos and normal orientation)
    print(idx)
    orig_coords = opm2_obj['chs'][idx]['loc'][3:] #testing on MEG 0123
    old_R = np.array(orig_coords).reshape(3, 3)
    r_dim = old_R[2,:]
    # old_dim1 = orig_coords[3:6] #take last 3 elements (z-direction)
    # old_dim2 = orig_coords[6:] #take last 3 elements (z-direction)

    #Flip 
    new_R = R.from_euler('z', '90', degrees=2).apply(r_dim)
    #new_dim1 = R.from_euler('z', '90', degrees=2).apply(old_dim1)
    #new_dim2 = R.from_euler('z', '90', degrees=2).apply(old_dim2)

    new_coords = old_R.copy()
    new_coords[2,:] = new_R
    new_coords = new_coords.flatten().tolist()
    #new_coords = np.concatenate([new_dim0, new_dim1, new_dim2])

    #Assign 
    opm2_mod['chs'][idx]['loc'][3:] = new_coords

fig = mne.viz.plot_alignment(
    opm2_mod, 
    dig=False, 
    trans = trans_fname,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    eeg=False,
    bem=bem_fname, 
    surfaces=("head", "pial"),
    #meg=['helmet','sensors'],
    meg=['sensors'],
    coord_frame='meg'
)


    # rx90 = np.array([ #rotating aroudn the local x-axis
    #     [1,0,0],
    #     [0,0,-1],
    #     [0,1,0]
    # ])

    # R_new = rx90 @ R 

    #R_new = np.array([R[2,:],R[1,:], R[0,:]])

    #new_coords = R_new.flatten().tolist()
    #new_coords = R_new.flatten() #must be flatteneed back the same way (inverse) of how we reshaped into 3,3 
    #opm2_mod['chs'][idx]['loc'][3:] = new_coords

    #confirmation test (from chat)
    print('- Confirmation test -')
    
    #Test1: 
    print(np.allclose(R @ rx90 @ rx90.T @ R.T, np.eye(3)))
    #Test2: 
    print(np.linalg.det(R)==1)
    #Test3
    R0 = np.array(orig_coords).reshape(3, 3, order="F").T
    R1 = R0 @ rx90
    ori2 = R1.T.reshape(-1, order='F')
    R2 = ori2.reshape(3,3, order='F').T
    print(np.allclose(R1, R2))

    # normal = R[:,0] #might also be R[2] #tried 2, 1, 0
    # new_normal = R[:,2] #tried 0, 1, 2
    # new_x = np.cross(R[:,1], new_normal) #tried 1, 2, 1

    # normal = R[0] # 2, 1, 0, 0
    # new_normal = R[2] # 0, 0, 1, 2
    # new_x = np.cross(R[1], new_normal) # 1, 2, 2, 1

    # new_x /=np.linalg.norm(new_x)
    # new_y = np.cross(new_normal, new_x)
    # R_new = np.column_stack((new_x, new_y, new_normal))

   

    #print(R)
    # axis = R[0,:]  # or EX / EY depending on your convention
    # theta = np.pi / 2 

    # def rotation_matrix(axis, theta):
    #     axis = axis / np.linalg.norm(axis)
    #     x, y, z = axis
    #     c = np.cos(theta)
    #     s = np.sin(theta)
    #     C = 1 - c

    #     return np.array([
    #         [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
    #         [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
    #         [z*x*C - y*s, y*z*C + x*s, c + z*z*C]
    #     ])

rx90 = np.array([ #rotating aroudn the local x-axis
    [1,0,0],
    [0,0,-1],
    [0,1,0]
])
n=R[:,2]
R_new = R @ rx90
n_new = R_new[:,2]
angle = np.degrees(np.arccos(np.clip(np.dot(n, n_new), -1, 1)))



R = sensor2[3:].reshape(3,3)


################################################
### Checking what's in each original axis and how it's affecting by rotating with rx90
################################################
Rx90 = np.array([
    [1,0,0],
    [0,0,-1],
    [0,1,0]
])

Rnew = R @ Rx90

#comparign each orig axis to each new axis 
for i in range(3):
    for j in range(3):
        a = R[:,i]
        b = Rnew[:,j]
        ang = np.degrees(np.arccos(np.clip(np.dot(a,b),-1,1)))
        print(i, j, ang)
# 0 0 0.0
# 0 1 89.99481864323936
# 0 2 89.99658061667486
# 1 0 90.00341938332514
# 1 1 89.9982450242778
# 1 2 179.45268043797353
# 2 0 89.99481864323936
# 2 1 0.0
# 2 2 90.0017549757222
# - INterpretation of this output (from chat)
#Look at column j=1: 
        # angle(0,1) ~ 90 deg
        # angle(1,1) ~ 90 deg 
        # angle(2,1) ~ 0 deg 
        # this menas that new axis 1 is identical to original axis 2 
#Look at column j = 2: 
        # angle(1,2) ~ 179 deg 
        # angle(0,2) ~ 90 deg 
        # angle (2,2) ~ 90 deg 
        # this means that new axis 2 is flipped version of original axis 1 
#So the mapping is: 
        # new axis j=0 -> mixture (original orthogonal basis vector)
        # new axis j=1 -> original axis 2 
        # new axis j=2 -> original axis 1 (flipped)
# Our transformation is effectively swwapping axes 1 and 2 (with a sign flip), 
# and this is NOT a general rotation. It is a basis permutation, not a continuous rotation. 
# Because we are not rotating in a continuous frame we get a result where it works for some, not for others, 
# depends on head location, adn looks like "partial rotation" - we are effectively changing which axis MNE interprets 
# as the sensor normal. 
        
#Interpretation of the 9 values based on all tests done: 
        # They are NOT a clean rotation matrix flattened in a row-major order 
        # They are a concatenation of three axis vectors, likely stored column-wise in MATLAB 

#How to correct this. 
    #You SHOULD NOT do: R = orig_coords.reshape(3,3)
    #You SHOULD do: R = orig_coords.reshape(3,3, order='F').T

#WHy this fixes everything - because now: 
        #columns = true sensor axes
        #cross products behave consistently 
        #rotation become true SO(3)
        #MNE visualization becomes stbale across sensors 

################################################
###
################################################

print(R @ R.T)
# [[ 1.00000001e+00  0.00000000e+00 -2.04078657e-05]
#  [ 0.00000000e+00  1.00000000e+00  0.00000000e+00]
#  [-2.04078657e-05  0.00000000e+00  9.99999952e-01]]
print(np.linalg.det(R))
#0.9999999833297526

#Look at the norms of the rows and columns 
print(np.linalg.norm(R, axis=0)) #[1.00002283 0.99997719 1.        ]
print(np.linalg.norm(R, axis=1)) #[1.         1.00000002 1.        ]

print(np.cross(R[:,0], R[:,1]))
print(R[:,2])

print(np.cross(R[0], R[1]))
print(R[2])

R1 = orig_coords.reshape(3,3)
R2 = orig_coords.reshape(3,3, order="F")

#### FROM CHAT: 
#Another possibility
# FieldTrip's grad.ori for OPMs is often not a rotation matrix in the sense of "rotate the sensor body." Rather, it is the orientation of the coil sensitivity axes.
# If the visualization (or forward model) only uses one axis of that matrix (typically the sensitive axis), then rotating the entire matrix by 90° does not necessarily make the displayed sensor appear perpendicular.
# This would exactly explain why some appear correct and others don't.

fig = mne.viz.plot_alignment(
    opm2_mod, 
    dig=False, 
    trans = trans_fname,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    eeg=False,
    bem=bem_fname, 
    surfaces=("head", "pial"),
    #meg=['helmet','sensors'],
    meg=['sensors'],
    coord_frame='meg'
)


### MNE sample OPM dataset 
data_path = mne.datasets.opm.data_path()
subject = "OPM_sample"
subjects_dir = data_path / "subjects"
raw_fname = data_path / "MEG" / "OPM" / "OPM_SEF_raw.fif"
bem_fname = subjects_dir / subject / "bem" / f"{subject}-5120-5120-5120-bem-sol.fif"
fwd_fname = data_path / "MEG" / "OPM" / "OPM_sample-fwd.fif"
coil_def_fname = data_path / "MEG" / "OPM" / "coil_def.dat"
raw = mne.io.read_raw_fif(raw_fname, preload=True)
raw.info['chs'][0]['loc']

for ch in raw.info['chs']:
    print(ch['ch_name'])
    print(ch['loc'])

bem = mne.read_bem_solution(bem_fname)
trans = mne.transforms.Transform("head", "mri")  # identity transformation

#change coil types so its easier to see orientation 
from mne.utils._bunch import NamedInt
for ch in raw.info['chs']:
    ch['coil_type'] = NamedInt("SQ20950N", 3024)

with mne.use_coil_def(coil_def_fname):
    fig = mne.viz.plot_alignment(
        raw.info,
        trans=trans,
        subject=subject,
        subjects_dir=subjects_dir,
        surfaces=("head", "pial"),
        bem=bem,
        meg=['helmet','sensors']
    )


#Attempting at rotating 90 degrees
orig_coords = raw.info['chs'][0]['loc'][3:]
R = np.array(orig_coords).reshape((3, 3))

# 3. Create a 90-degree rotation matrix around the Z-axis (Z-axis stays fixed)
# Change the angle to np.pi/2 for clockwise or -np.pi/2 for counter-clockwise
theta = np.pi / 2 
#theta = np.pi
R_rot = np.array([
    [0, 0, 1],
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0]
])

# 4. Multiply matrices to get the new orientation
R_new = R @ R_rot

# 5. Flatten back to 9 coordinates
new_coords = R_new.flatten().tolist()
raw.info['chs'][0]['loc'][3:] = new_coords


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