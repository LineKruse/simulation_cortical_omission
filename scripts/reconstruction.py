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
import seaborn as sns
from helper_functions import compute_RLE

dir = os.getcwd()
dir = dir.replace('/scripts','')


folder = 'test_increasing_snr_methods/occpitial_0.1nA_increasing_size'
#folder = 'test_increasing_snr_methods/occpitial_onedip_increasing_amplitude'
recon_path = os.path.join(dir,'data/reconstructions')
recon_folder = os.path.join(recon_path, folder)
sim_folder = os.path.join(dir, f'data/simulations/{folder}')
#vol_regions = ['Left-Caudate','Right-Caudate','Left-Cerebellum-Cortex','Right-Cerebellum-Cortex','Left-Hippocampus','Right-Hippocampus','Left-Thalamus-Proper','Right-Thalamus-Proper']
surf_regions = ['lateraloccipital-lh']
subject = 'fsaverage'
subjects_dir = os.path.join(dir,'data/freesurfer/subjects')
aseg_fname= os.path.join(dir, 'data/freesurfer/fsaverage/mri/aparc+aseg.mgz')
surf_spacing = 'oct6'
vol_spacing = 5.0 
fname_raw = os.path.join(dir, 'data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif')
fname_bem = os.path.join(subjects_dir, subject, 'bem','fsaverage-5120-5120-5120-bem-sol.fif') 
fname_trans = 'fsaverage'

#Create info object 
raw_sample = mne.io.read_raw_fif(fname_raw)
info = raw_sample.info

#Load src and fwd (differnet from those used for simulations)
src_mix = mne.read_source_spaces(os.path.join(recon_path,f'mixed_surf{surf_spacing}_vols{vol_spacing}_src.fif'))
fwd_mix = mne.read_forward_solution(os.path.join(recon_path,f'mixed_surf{surf_spacing}_vols{vol_spacing}_fwd.fif'))

""" #Create mixed source space including all volume regions we want + surf 
# - OBS: surf src is currently the same as used in sims - something should be changed here so that they are not identical (e.g., spacing or smth)
src_surf = mne.setup_source_space(subject, spacing=surf_spacing, add_dist="patch", subjects_dir=subjects_dir) 
src_vol = mne.setup_volume_source_space(
    subject=subject, 
    mri=aseg_fname,
    sphere=(0,0,0,0.12),
    volume_label = vol_regions, 
    bem=fname_bem, 
    subjects_dir=subjects_dir,
    sphere_units="m"
)
src_mix = src_surf + src_vol
mne.write_source_spaces(os.path.join(recon_path,f'mixed_surf{surf_spacing}_vols{vol_spacing}_src.fif'), src_mix, overwrite=True)

#Compute FWD 
fwd_mix = mne.make_forward_solution(info, fname_trans, src_mix, fname_bem, mindist=5.0)
mne.write_forward_solution(os.path.join(recon_path,f'mixed_surf{surf_spacing}_vols{vol_spacing}_fwd.fif'), fwd_mix, overwrite=True)
 """

#Plot FWD with sources 
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
    fwd=fwd_mix,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)
mne.viz.set_3d_view(figure=fig, azimuth=180, distance=1, focalpoint="auto")


############################################################################
#                     RUN MNE RECON 
############################################################################

snr = 3.0
lambda2 = 1.0/snr**2
method = "dSPM"

#Set/create paths 

#sim_path = os.path.join(sim_folder,"Left-Thalamus-Proper")
sim_path = sim_folder
sim_recon_path = os.path.join(recon_path, folder)
if not os.path.exists(sim_recon_path):
    os.mkdir(sim_recon_path)
sim_recon_path_mne = os.path.join(sim_recon_path, 'mne')
if not os.path.exists(sim_recon_path_mne):
    os.mkdir(sim_recon_path_mne)
sim_recon_path_mne_figures = os.path.join(sim_recon_path_mne, "figures")
if not os.path.exists(sim_recon_path_mne_figures):
    os.mkdir(sim_recon_path_mne_figures)

sim_epochs = [epo for epo in os.listdir(sim_path) if epo.endswith(f"-epo.fif")]

for epo in sim_epochs: 
    fname_stc_str = epo.replace("-epo.fif","-mne-stc.h5")
    if not os.path.exists(os.path.join(sim_recon_path_mne, fname_stc_str)):

        #Load simulated epochs 
        epochs = mne.read_epochs(os.path.join(sim_path, epo))

        #Load corresponding evokeds 
        fname_evoked = epo.replace("-epo", "-ave")
        evoked = mne.read_evokeds(os.path.join(sim_path, fname_evoked))[0] #returned as list of 1 element 

        #Compute noise covariance from epochs 
        noise_cov_est = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='empirical')

        #To use 3 dipole orientations at each point (loose orientation), set fixed=False and loose=1.0
        # - NB! Only if subcortical (or mixed?)
        inverse_operator = make_inverse_operator(evoked.info, fwd_mix, noise_cov_est, fixed=False, loose=1.0) #use new fwd and src 
        stc_mne = apply_inverse(evoked, inverse_operator, lambda2, method, pick_ori=None)

        #Save
        fname_inv = epo.replace("-epo", "-inv")
        fname_stc = epo.replace("-epo.fif","-mne")
        mne.minimum_norm.write_inverse_operator(os.path.join(sim_recon_path_mne, fname_inv), inverse_operator, overwrite=True)
        stc_mne.save(os.path.join(sim_recon_path_mne, fname_stc), ftype='h5',overwrite=True)

        #Plot stc  
        # Use peak getter to move visualization to the time point of the peak magnitude
        peak_vert, peak_time = stc_mne.get_peak(mode="abs")
        smoothing_steps = 7
        stc_mne_copy = stc_mne
        brain = stc_mne.plot(
            initial_time=peak_time,
            hemi="lh",
            src=src_mix,
            surface="white",
            subjects_dir=subjects_dir,
            smoothing_steps=smoothing_steps,
        )
        fname_stc_image = epo.replace("-epo.fif", "-mne-stc.png")
        fname_stc_movie = epo.replace("-epo.fif", "-mne-stc-movie")
        brain.save_image(filename=os.path.join(sim_recon_path_mne_figures, fname_stc_image))
        # brain.save_movie(filename=os.path.join(sim_recon_path_mne, "figures",fname_stc_movie), time_dilation=20, tmin=0.05, tmax=0.16, framerate=10,
        #                   interpolation='linear', time_viewer=True)
        brain.close()


############################################################################
#                     RUN LCMV RECON    
############################################################################


print(f"------- Running region {folder} ----------")

#Set/create paths 
sim_path = sim_folder
sim_recon_path = os.path.join(recon_path, folder)
if not os.path.exists(sim_recon_path):
    os.mkdir(sim_recon_path)
sim_recon_path_lcmv = os.path.join(sim_recon_path, 'lcmv')
if not os.path.exists(sim_recon_path_lcmv):
    os.mkdir(sim_recon_path_lcmv)
sim_recon_path_lcmv_figures = os.path.join(sim_recon_path_lcmv, "figures")
if not os.path.exists(sim_recon_path_lcmv_figures):
    os.mkdir(sim_recon_path_lcmv_figures)

sim_epochs = [epo for epo in os.listdir(sim_path) if epo.endswith(f"-epo.fif")]

for epo in sim_epochs: 
    print(f"-- Computing LCMV for {epo}")

    fname_filters = epo.replace("-epo.fif","-filters-lcmv.h5")
    #if not os.path.exists(os.path.join(sim_recon_path_lcmv, fname_filters)):

    #Load simulated epochs 
    epochs = mne.read_epochs(os.path.join(sim_path, epo))

    #Load corresponding evokeds 
    fname_evoked = epo.replace("-epo", "-ave")
    evoked = mne.read_evokeds(os.path.join(sim_path, fname_evoked))[0] #returned as list of 1 element 

    #Compute the data covariance matrix from epochs 
    data_cov = mne.compute_covariance(epochs, tmin=0.01, tmax=0.25, method='empirical') #number of samples used=2923 

    #Compute noise covariance matrix (used for whitening, to account for hte different amplitdue scales of the grads and mags)
    noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='empirical') #same tmin as used for epoching 

    #Compute spatial filter (lcmv)
    #We will optimize the orientation of hte sources such that the output power is maxiized (pick_ori='max_power')
    filters = mne.beamformer.make_lcmv(
        evoked.info,
        fwd_mix, 
        data_cov, 
        reg=0.05, #accounting for slight rank-deficiency in data cov 
        noise_cov=noise_cov, 
        pick_ori='max-power', #optimizing orientation fo sources such that output power is maximized 
        weight_norm='unit-noise-gain', #mitigating depth bias 
        rank=None)
    fname_filters = epo.replace("-epo.fif","-filters-lcmv.h5")
    filters.save(os.path.join(sim_recon_path_lcmv, fname_filters), overwrite=True)

    #Apply the spatial filter 
    stc_lcmv = mne.beamformer.apply_lcmv(evoked, filters) #76 time samples (within epoch)
    fname_stc = epo.replace("-epo.fif","-lcmv")
    stc_lcmv.save(os.path.join(sim_recon_path_lcmv, fname_stc), ftype='h5',overwrite=True)

    #Plot stc  
    # Use peak getter to move visualization to the time point of the peak magnitude
    peak_vert, peak_time = stc_lcmv.get_peak(mode="abs")
    smoothing_steps = 7
    
    stc_lcmv_copy = stc_lcmv.copy()
    stc_lcmv_copy._data = abs(stc_lcmv_copy._data)
    brain = stc_lcmv_copy.plot(
        initial_time=peak_time,
        hemi="lh",
        src=src_mix,
        surface="white",
        subjects_dir=subjects_dir,
        smoothing_steps=smoothing_steps,
    )
    fname_stc_image = epo.replace("-epo.fif", "-lcmv-stc.png")
    fname_stc_movie = epo.replace("-epo.fif", "-lcmv-stc-movie")
    brain.save_image(filename=os.path.join(sim_recon_path_lcmv_figures, fname_stc_image))
    # brain.save_movie(filename=os.path.join(sim_recon_path_mne, "figures",fname_stc_movie), time_dilation=20, tmin=0.05, tmax=0.16, framerate=10,
    #                   interpolation='linear', time_viewer=True)
    brain.close()


