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
os.chdir('/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/scripts')
from helper_functions import compute_RLE

folder = 'thalamic_1nA_occipital_01nA'
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
recon_folder = os.path.join(recon_path, folder)
sim_folder = f'/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/{folder}'
vol_regions = ['Left-Caudate','Right-Caudate','Left-Cerebellum-Cortex','Right-Cerebellum-Cortex','Left-Hippocampus','Right-Hippocampus','Left-Thalamus-Proper','Right-Thalamus-Proper']
surf_regions = ['lateraloccipital-lh']
subject = 'fsaverage'
subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
aseg_fname='/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/fsaverage/mri/aparc+aseg.mgz'
surf_spacing = 'oct6'
vol_spacing = 5.0 
fname_raw = raw_fname =  '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_bem = os.path.join(subjects_dir, subject, 'bem','fsaverage-5120-5120-5120-bem-sol.fif') 
fname_trans = 'fsaverage'

#Create info object 
raw_sample = mne.io.read_raw_fif(fname_raw)
info = raw_sample.info

#Load src and fwd 
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



############################################################################
#              COMPUTE REGION LOCALIZATIONE ERROR (RLE)    
############################################################################

recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations'

thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

#Recon for all sims used the same fwd/src 
fwd_recon = mne.read_forward_solution(os.path.join(recon_path, 'mixed_surfoct6_vols5.0_fwd.fif'))
src_recon = fwd_recon['src']

#### OCCIPITAL ONLY #### 
sims_list = ['occipital_01nA']

region_list = []
amplitude_list = []
extent_list = []
n_vertices_list = []
threshold_list = []
rle_mne_list = []
rle_lcmv_list = []

for sim in sims_list: 
    
    #region_name = os.listdir(os.path.join(sim_path, sim))[0]
    region_sims = [f for f in os.listdir(os.path.join(sim_path, sim)) if f.endswith("-lh.stc")]

    for stc in region_sims: 
        amplitude = stc.split("-")[-3].split("_")[0]
        extent = stc.split("-")[-2].split("_")[0]

        name = stc.split("mm")[0]

        #Load true STCs
        stc_true = mne.read_source_estimate(os.path.join(sim_path, sim, stc))
        n_vertices_sim = len(stc_true.vertices[0])
        ##### MNE ######

        #Load reconstructed STCs 
        recon_name_mne = str(name+'mm-mne-stc.h5')
        recon_name_lcmv = str(name+'mm-lcmv-stc.h5')
        stc_est_mne = mne.read_source_estimate(os.path.join(recon_path, sim, 'mne', recon_name_mne))
        stc_est_lcmv = mne.read_source_estimate(os.path.join(recon_path, sim, 'lcmv', recon_name_lcmv))
        
        #Crop both est stcs to start at 0 
        stc_est_mne = stc_est_mne.crop(tmin=0, tmax=None,include_tmax=False)
        stc_est_lcmv = stc_est_lcmv.crop(tmin=0, tmax=None,include_tmax=False)

        #Load SRCs 
        src_sim_name = [f for f in os.listdir(os.path.join(sim_path, sim)) if "-fwd.fif" in f][0]
        fwd_sim = mne.read_forward_solution(os.path.join(sim_path, sim, src_sim_name))
        src_sim = fwd_sim['src']

        #Crop true STC to have same length as one source time course (the reconstructed STC length)
        stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*stc_est_mne.data.shape[1],include_tmax=False)

        for threshold in thresholds: 
            thres_str = str(f"{threshold}%")
            rle_mne = compute_RLE(stc_true, stc_est_mne, src_sim, src_recon, threshold=thres_str)
            rle_lcmv = compute_RLE(stc_true, stc_est_lcmv, src_sim, src_recon, threshold=thres_str)

            region_list.append(sim.split("_")[0])
            amplitude_list.append(amplitude)
            extent_list.append(extent)
            n_vertices_list.append(n_vertices_sim)
            threshold_list.append(threshold)
            rle_mne_list.append(rle_mne)
            rle_lcmv_list.append(rle_lcmv)


df_rle_surf = pd.DataFrame({'region':region_list, 
                       'amplitude': amplitude_list,
                       'patch_size':extent_list,
                       'n_vertices_sim':n_vertices_list,
                       'threshold': threshold_list,
                       'rle_mne': rle_mne_list,
                       'rle_lcmv':rle_lcmv_list})
df_rle_surf.to_csv(os.path.join(recon_path, sims_list[0], 'rle_occipital.csv'))

rle_sub = df_rle_surf
#rle_sub = rle_df[rle_df['region']==sims_list[0].split("_")[0]]
rle_sub['rle_mne_mm'] = rle_sub['rle_mne']*1000
rle_sub['rle_lcmv_mm'] = rle_sub['rle_lcmv']*1000

fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=False)
sns.lineplot(data=rle_sub, x="threshold", y="rle_mne_mm", hue="patch_size", ax=ax[0])
sns.scatterplot(data=rle_sub, x="threshold", y="rle_mne_mm", hue="patch_size", legend=False, ax=ax[0])
sns.lineplot(data=rle_sub, x="threshold", y="rle_lcmv_mm", hue="patch_size", ax=ax[1])
sns.scatterplot(data=rle_sub, x="threshold", y="rle_lcmv_mm", hue="patch_size", legend=False, ax=ax[1])
ax[0].set_title("MNE")
ax[1].set_title("LCMV")
ax[0].legend(title="Size (mm)")
ax[1].legend(title="Size (mm)")
ax[0].set_ylabel("RLE (mm)")
ax[1].set_ylabel("RLE (mm)")
ax[0].set_ylim(0,50)
ax[1].set_ylim(0,50)
plt.suptitle(f"Occipital (0.1 nA)\nRegion Localization Error (RLE), spatiotemporal")
plt.savefig(os.path.join(recon_path, sims_list[0],f'rle_{sim}.png'))
plt.close()
    

#### THALAMUS ONLY #### 
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations'

sims_list = ['thalamic_1nA']

region_list = []
amplitude_list = []
extent_list = []
n_vertices_list = []
threshold_list = []
rle_mne_list = []
rle_lcmv_list = []

for sim in sims_list: 
    
    region_name = [f for f in os.listdir(os.path.join(sim_path, sim)) if not "." in f][0]
    region_sims = [f for f in os.listdir(os.path.join(sim_path, sim, region_name)) if f.endswith("-lh.stc")]

    for stc in region_sims: 
        amplitude = stc.split("-")[-3].split("_")[0]
        extent = stc.split("-")[-2].split("_")[0]

        name = stc.split("mm")[0]

        #Load true STCs
        stc_true = mne.read_source_estimate(os.path.join(sim_path, sim, region_name, stc))
        n_vertices_sim = len(stc_true.vertices[0])
        ##### MNE ######

        #Load reconstructed STCs 
        recon_name_mne = str(name+'mm-mne-stc.h5')
        recon_name_lcmv = str(name+'mm-lcmv-stc.h5')
        stc_est_mne = mne.read_source_estimate(os.path.join(recon_path, sim, 'mne', recon_name_mne))
        stc_est_lcmv = mne.read_source_estimate(os.path.join(recon_path, sim, 'lcmv', recon_name_lcmv))

        #Crop both to start at 0 
        stc_est_mne = stc_est_mne.crop(tmin=0, tmax=None,include_tmax=False)
        stc_est_lcmv = stc_est_lcmv.crop(tmin=0, tmax=None,include_tmax=False)

        #Load SRCs 
        src_sim_name = [f for f in os.listdir(os.path.join(sim_path, sim, region_name)) if "-fwd.fif" in f][0]
        fwd_sim = mne.read_forward_solution(os.path.join(sim_path, sim, region_name, src_sim_name))
        src_sim = fwd_sim['src']

        #Crop true STC to have same length as one source time course (the reconstructed STC length)
        stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*stc_est_mne.data.shape[1],include_tmax=False)

        for threshold in thresholds: 
            thres_str = str(f"{threshold}%")
            rle_mne = compute_RLE(stc_true, stc_est_mne, src_sim, src_recon, threshold=thres_str)
            rle_lcmv = compute_RLE(stc_true, stc_est_lcmv, src_sim, src_recon, threshold=thres_str)

            region_list.append(sim.split("_")[0])
            amplitude_list.append(amplitude)
            extent_list.append(extent)
            n_vertices_list.append(n_vertices_sim)
            threshold_list.append(threshold)
            rle_mne_list.append(rle_mne)
            rle_lcmv_list.append(rle_lcmv)


df_rle_vol = pd.DataFrame({'region':region_list, 
                       'amplitude': amplitude_list,
                       'patch_size':extent_list,
                       'n_vertices_sim':n_vertices_list,
                       'threshold': threshold_list,
                       'rle_mne': rle_mne_list,
                       'rle_lcmv':rle_lcmv_list})
df_rle_vol.to_csv(os.path.join(recon_path, sims_list[0], 'rle_thalamus.csv'))

rle_sub = df_rle_vol
#rle_sub = rle_df[rle_df['amplitude']==str(amp)]
rle_sub['rle_mne_mm'] = rle_sub['rle_mne']*1000
rle_sub['rle_lcmv_mm'] = rle_sub['rle_lcmv']*1000

fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=False)
sns.lineplot(data=rle_sub, x="threshold", y="rle_mne_mm", hue="patch_size", ax=ax[0])
sns.scatterplot(data=rle_sub, x="threshold", y="rle_mne_mm", hue="patch_size", legend=False, ax=ax[0])
sns.lineplot(data=rle_sub, x="threshold", y="rle_lcmv_mm", hue="patch_size", ax=ax[1])
sns.scatterplot(data=rle_sub, x="threshold", y="rle_lcmv_mm", hue="patch_size", legend=False, ax=ax[1])
ax[0].set_title("MNE")
ax[1].set_title("LCMV")
ax[0].set_ylabel("RLE (mm)")
ax[1].set_ylabel("RLE (mm)")
ax[0].legend(title="Size (mm)")
ax[1].legend(title="Size (mm)")
ax[0].set_ylim(0,65)
ax[1].set_ylim(0,65)
#ax[0].set_ylim(0.)
plt.suptitle(f"Thalamus (1.0 nA)\nRegion Localization Error (RLE), spatiotemporal")
plt.savefig(os.path.join(recon_path, sims_list[0],f'rle_thalamus_1nA.png'))
plt.close()
    


#### MIXED #### 
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations'

sims_list = ['thalamic_1nA_occipital_01nA']

region_list = []
amplitude_list = []
extent_vol_list = []
extent_surf_list = []
n_vertices_list = []
threshold_list = []
rle_mne_list = []
rle_lcmv_list = []

for sim in sims_list: 
    
    #region_name = [f for f in os.listdir(os.path.join(sim_path, sim)) if not "." in f][0]
    region_name=sim
    region_sims = [f for f in os.listdir(os.path.join(sim_path, sim)) if f.endswith("-lh.stc")]

    for stc in region_sims: 
        amplitude = stc.split("_")[1]
        extent_vol = stc.split("Thalamus-Proper-lh_")[1].split("_")[0]
        if not "--lh" in stc: 
            extent_surf = stc.split("occipital-lh_")[1].split("_")[0]
        else: 
            extent_surf = str(0.0)

        name = stc.split("-lh.")[0]

        #Load true STCs
        stc_true = mne.read_source_estimate(os.path.join(sim_path, sim, stc))
        n_vertices_sim = len(stc_true.vertices[0])
        ##### MNE ######

        #Load reconstructed STCs 
        recon_name_mne = str(name+'-mne-stc.h5')
        recon_name_lcmv = str(name+'-lcmv-stc.h5')
        stc_est_mne = mne.read_source_estimate(os.path.join(recon_path, sim, 'mne', recon_name_mne))
        stc_est_lcmv = mne.read_source_estimate(os.path.join(recon_path, sim, 'lcmv', recon_name_lcmv))

        #Crop both to start at 0 
        stc_est_mne = stc_est_mne.crop(tmin=0, tmax=None,include_tmax=False)
        stc_est_lcmv = stc_est_lcmv.crop(tmin=0, tmax=None,include_tmax=False)

        #Load SRCs 
        src_sim_name = [f for f in os.listdir(os.path.join(sim_path, sim)) if "-fwd.fif" in f][0]
        fwd_sim = mne.read_forward_solution(os.path.join(sim_path, sim, src_sim_name))
        src_sim = fwd_sim['src']

        #Crop true STC to have same length as one source time course (the reconstructed STC length)
        stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*stc_est_mne.data.shape[1],include_tmax=False)

        for threshold in thresholds: 
            thres_str = str(f"{threshold}%")
            rle_mne = compute_RLE(stc_true, stc_est_mne, src_sim, src_recon, threshold=thres_str)
            rle_lcmv = compute_RLE(stc_true, stc_est_lcmv, src_sim, src_recon, threshold=thres_str)

            region_list.append(sim)
            amplitude_list.append(amplitude)
            extent_vol_list.append(extent_vol)
            extent_surf_list.append(extent_surf)
            n_vertices_list.append(n_vertices_sim)
            threshold_list.append(threshold)
            rle_mne_list.append(rle_mne)
            rle_lcmv_list.append(rle_lcmv)


df_rle_mix = pd.DataFrame({'region':region_list, 
                       'amplitude': amplitude_list,
                       'patch_size_vol':extent_vol_list,
                       'patch_size_surf':extent_surf_list,
                       'n_vertices_sim':n_vertices_list,
                       'threshold': threshold_list,
                       'rle_mne': rle_mne_list,
                       'rle_lcmv':rle_lcmv_list})
df_rle_mix.to_csv(os.path.join(recon_path, 'rle_mix.csv'))

rle_sub = df_rle_mix
surf_sizes = [0.0, 2.0, 4.0, 6.0, 8.0]
rle_sub['rle_mne_mm'] = rle_sub['rle_mne']*1000
rle_sub['rle_lcmv_mm'] = rle_sub['rle_lcmv']*1000


for surf_size in surf_sizes: 

    rle_sub_surf = rle_sub[rle_sub['patch_size_surf']==str(surf_size)]
    max_mne = rle_sub_surf.rle_mne_mm.max()
    max_lcmv = rle_sub_surf.rle_lcmv_mm.max()
    max = np.array((max_mne, max_lcmv)).max()

    fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=False)
    sns.lineplot(data=rle_sub_surf, x="threshold", y="rle_mne_mm", hue="patch_size_vol", ax=ax[0])
    sns.scatterplot(data=rle_sub_surf, x="threshold", y="rle_mne_mm", hue="patch_size_vol", legend=False, ax=ax[0])
    sns.lineplot(data=rle_sub_surf, x="threshold", y="rle_lcmv_mm", hue="patch_size_vol", ax=ax[1])
    sns.scatterplot(data=rle_sub_surf, x="threshold", y="rle_lcmv_mm", hue="patch_size_vol", legend=False, ax=ax[1])
    ax[0].set_title("MNE")
    ax[1].set_title("LCMV")
    ax[0].set_ylabel("RLE (mm)")
    ax[1].set_ylabel("RLE (mm)")
    ax[0].legend(title="Size (mm)")
    ax[1].legend(title="Size (mm)")
    ax[0].set_ylim(0, max)
    ax[1].set_ylim(0,max)
    plt.suptitle(f"Thalamus + Occipital {surf_size} mm \nRegion Localization Error (RLE), spatiotemporal")
    plt.savefig(os.path.join(recon_path, sims_list[0],f'rle_thalamus_1nA_occipital_{surf_size}mm.png'))
    plt.close()


#### Plot with threshold 70% thalamic size against V1 size 
# - Chose 70% becuase htat is where thalamic alone has lowest error in general 

rle_sub = df_rle_mix[df_rle_mix.threshold==70]
rle_sub['rle_mne_mm'] = rle_sub['rle_mne']*1000
rle_sub['rle_lcmv_mm'] = rle_sub['rle_lcmv']*1000
rle_sub.patch_size_surf = pd.to_numeric(rle_sub.patch_size_surf)

max_mne = rle_sub.rle_mne_mm.max()
max_lcmv = rle_sub.rle_lcmv_mm.max()
max = np.array((max_mne, max_lcmv)).max()

fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=False)
sns.lineplot(data=rle_sub, x="patch_size_surf", y="rle_mne_mm", hue="patch_size_vol", ax=ax[0])
sns.scatterplot(data=rle_sub, x="patch_size_surf", y="rle_mne_mm", hue="patch_size_vol", legend=False, ax=ax[0])
sns.lineplot(data=rle_sub, x="patch_size_surf", y="rle_lcmv_mm", hue="patch_size_vol", ax=ax[1])
sns.scatterplot(data=rle_sub, x="patch_size_surf", y="rle_lcmv_mm", hue="patch_size_vol", legend=False, ax=ax[1])
ax[0].set_title("MNE")
ax[1].set_title("LCMV")
ax[0].set_ylabel("RLE (mm)")
ax[1].set_ylabel("RLE (mm)")
ax[0].legend(title="Size (mm)")
ax[1].legend(title="Size (mm)")
ax[0].set_ylim(0, max)
ax[1].set_ylim(0,max)
plt.suptitle(f"Thalamus + Occipital (threshold=70%)\nRegion Localization Error (RLE), spatiotemporal")
plt.savefig(os.path.join(recon_path, sims_list[0],f'rle_thalamus_1nA_occipital_01nA_thres70.png'))
plt.close()


############################################################################
#              COMPUTE AND PLOT SNR FOR ALL EVOKEDS   
############################################################################

###### Occipital 
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations'
sim = 'occipital_01nA'
evokeds = [f for f in os.listdir(os.path.join(sim_path, sim)) if "-ave.fif" in f]
region = 'occipital'
patch_size = []
snr_max_list = []
snr_mean_list = []
for evo in evokeds: 
    extent = evo.split("-")[-2].split("_")[0]
    patch_size.append(extent)
    evoked = mne.read_evokeds(os.path.join(sim_path, sim, evo), baseline=(None, 0))[0]
    inv_fname = evo.replace("-ave.fif", "-inv.fif")
    inv_path = os.path.join(recon_path, sim, "mne", inv_fname)
    inv = mne.minimum_norm.read_inverse_operator(inv_path)
    snr = mne.minimum_norm.estimate_snr(evoked, inv, verbose=None)[0]
    snr_max_list.append(snr.max())
    snr_mean_list.append(snr.mean())

    plt.figure()
    fig = mne.viz.plot_snr_estimate(evoked, inv, show=False)
    fig.savefig(os.path.join(recon_path, sim, f"snr_plot_occipital_{extent}.png"))


df_snr_surf = pd.DataFrame({'region':"occipital", 
                       'patch_size': patch_size,
                       'snr_mean': snr_mean_list,
                       'snr_max':snr_max_list})
df_snr_surf.to_csv(os.path.join(recon_path, sim, 'snr_surf.csv'))
df_snr_surf = pd.read_csv(os.path.join(recon_path, sim, 'snr_surf.csv'))

###### Thalamic 
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations'
sim = 'thalamic_1nA'
evokeds = [f for f in os.listdir(os.path.join(sim_path, sim, 'Left-Thalamus-Proper')) if "-ave.fif" in f]
region = 'thalamus'
patch_size = []
snr_max_list = []
snr_mean_list = []
for evo in evokeds: 
    extent = evo.split("-")[-2].split("_")[0]
    patch_size.append(extent)
    evoked = mne.read_evokeds(os.path.join(sim_path, sim,'Left-Thalamus-Proper', evo), baseline=(None, 0))[0]
    inv_fname = evo.replace("-ave.fif", "-inv.fif")
    inv_path = os.path.join(recon_path, sim, "mne", inv_fname)
    inv = mne.minimum_norm.read_inverse_operator(inv_path)
    snr = mne.minimum_norm.estimate_snr(evoked, inv, verbose=None)[0]
    snr_max_list.append(snr.max())
    snr_mean_list.append(snr.mean())

    # plt.figure()
    # fig = mne.viz.plot_snr_estimate(evoked, inv, show=False)
    # fig.savefig(os.path.join(recon_path, sim, f"snr_plot_{region}_{extent}.png"))


df_snr_vol = pd.DataFrame({'region':region,
                       'patch_size': patch_size,
                       'snr_mean': snr_mean_list,
                       'snr_max':snr_max_list})
df_snr_vol.to_csv(os.path.join(recon_path, sim, 'snr_vol.csv'))
df_snr_vol = pd.read_csv(os.path.join(recon_path, sim, 'snr_vol.csv'))

###### Thalamic + Occipital (mixed)
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions'
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations'
sim = 'thalamic_1nA_occipital_01nA'
evokeds = [f for f in os.listdir(os.path.join(sim_path, sim)) if "-ave.fif" in f]
region = 'thalamus_occipital'
patch_size_surf = []
patch_size_vol = []
snr_max_list = []
snr_mean_list = []
for evo in evokeds: 
    extent_vol = evo.split("Proper-lh_")[1].split("_")[0]
    if "lateraloccipital" in evo:
        extent_surf = evo.split("lateraloccipital-lh_")[1].split("_")[0]
    else: 
        extent_surf = str(0.0)
    patch_size_vol.append(extent_vol)
    patch_size_surf.append(extent_surf)
    evoked = mne.read_evokeds(os.path.join(sim_path, sim, evo), baseline=(None, 0))[0]
    inv_fname = evo.replace("-ave.fif", "-inv.fif")
    inv_path = os.path.join(recon_path, sim, "mne", inv_fname)
    inv = mne.minimum_norm.read_inverse_operator(inv_path)
    snr = mne.minimum_norm.estimate_snr(evoked, inv, verbose=None)[0]
    snr_max_list.append(snr.max())
    snr_mean_list.append(snr.mean())

    plt.figure()
    fig = mne.viz.plot_snr_estimate(evoked, inv, show=False)
    fig.savefig(os.path.join(recon_path, sim, f"snr_plot_thalamus{extent_vol}_occipital{extent_surf}.png"))


df_snr_mix = pd.DataFrame({'region':region,
                       'patch_size_surf': patch_size_surf,
                       'patch_size_vol': patch_size_vol,
                       'snr_mean': snr_mean_list,
                       'snr_max':snr_max_list})
df_snr_mix.to_csv(os.path.join(recon_path, sim, 'snr_mix.csv'))


##PLOT MIX 
df_snr_mix.patch_size_surf = pd.to_numeric(df_snr_mix.patch_size_surf)
df_snr_mix.patch_size_vol = pd.to_numeric(df_snr_mix.patch_size_vol)
plt.figure()
sns.lineplot(data=df_snr_mix, x="patch_size_surf", y="snr_max", hue="patch_size_vol")
plt.title("Mixed (thalamic + occipital)")
plt.legend(title="Thalamic size (mm)")
plt.xlabel("Occipital size (mm)")
plt.ylabel("SNR")
plt.savefig(os.path.join(recon_path, "thalamic_1nA_occipital_01nA",f'snr_thalamus_1nA_occipital_01nA.png'))

## PLOT VOL 
df_snr_surf.patch_size = pd.to_numeric(df_snr_surf.patch_size)
df_snr_vol.patch_size = pd.to_numeric(df_snr_vol.patch_size)
snr_df_comb = pd.concat((df_snr_surf, df_snr_vol))
plt.figure()
sns.lineplot(data=snr_df_comb, x="patch_size", y="snr_max", hue="region", palette=['darkgreen', 'darkred'])
sns.scatterplot(data=snr_df_comb, x="patch_size", y="snr_max", hue="region", palette=['darkgreen', 'darkred'], legend=False)
plt.title("SNR Comparison")
plt.xlabel("Size (mm)")
plt.ylabel("SNR")
plt.savefig(os.path.join(recon_path, 'thalamic_1nA',f'snr_thalamus_1nA_vs_occipital_01nA.png'))
plt.savefig(os.path.join(recon_path, 'occipital_01nA',f'snr_thalamus_1nA_vs_occipital_01nA.png'))


fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=False)
sns.lineplot(data=rle_sub, x="patch_size_surf", y="rle_mne_mm", hue="patch_size_vol", ax=ax[0])
sns.scatterplot(data=rle_sub, x="patch_size_surf", y="rle_mne_mm", hue="patch_size_vol", legend=False, ax=ax[0])
sns.lineplot(data=rle_sub, x="patch_size_surf", y="rle_lcmv_mm", hue="patch_size_vol", ax=ax[1])
sns.scatterplot(data=rle_sub, x="patch_size_surf", y="rle_lcmv_mm", hue="patch_size_vol", legend=False, ax=ax[1])
ax[0].set_title("MNE")
ax[1].set_title("LCMV")
ax[0].set_ylabel("RLE (mm)")
ax[1].set_ylabel("RLE (mm)")
ax[0].legend(title="Size (mm)")
ax[1].legend(title="Size (mm)")
ax[0].set_ylim(0, max)
ax[1].set_ylim(0,max)
plt.suptitle(f"Thalamus + Occipital (threshold=70%)\nRegion Localization Error (RLE), spatiotemporal")
plt.savefig(os.path.join(recon_path, sims_list[0],f'rle_thalamus_1nA_occipital_01nA_thres70.png'))
plt.close()



############################################################################
#              THALAMUS - PLOT PEAK ACTIVATION AT THALAMIC PEAK (0.075)
# - in both true and estimated stc      
############################################################################
        
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/thalamic_1nA/Left-Thalamus-Proper'
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/thalamic_1nA'
recon_path_lcmv = os.path.join(recon_path, 'lcmv')

stc_list_lcmv = [f for f in os.listdir(recon_path_lcmv) if '-stc.h5' in f]

src_recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/mixed_surfoct6_vols5.0_src.fif'
src_recon = mne.read_source_spaces(src_recon_path)

for stc in stc_list_lcmv: 

    extent = stc.split("-")[-3].split(".")[0]

    #Load estimated stc 
    stc_est = mne.read_source_estimate(os.path.join(recon_path_lcmv, stc))

    #Crop to start at 0 (epochs had -200 ms included)
    stc_est_crop = stc_est.crop(tmin=0, tmax=None)


    #Load true stc 
    stc_true_path = os.path.join(sim_path, stc.replace("-lcmv-stc.h5","-lh.stc"))
    stc_true = mne.read_source_estimate(stc_true_path)

    #Crop to have same length (=epoch) as estimated stc 
    stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*100,include_tmax=False)

    #Extract simulated vertices in true stc from thalamus 
    fwd_sim = mne.read_forward_solution(os.path.join(sim_path, 'Left-Thalamus-Proper-xx_mm-fsaverage-fwd.fif'))
    src_sim = mne.read_source_spaces(os.path.join(sim_path, 'Left-Thalamus-Proper-xx_mm-fsaverage-src.fif'))

    #Find time sample of thalamic peak (0.075) and V1 peak (0.95)
    thalamic_time_idx = int(0.075/stc_true.tstep)
    occipital_time_idx = int(0.095/stc_true.tstep)

    #Find position of peak vertex/ices in true 
    stc_true_max = stc_true.data[:,thalamic_time_idx].max() #all are the same in sims
    peak_true_pos = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find position of peak vertex in estimated 
    peak_true = stc_true.get_peak(tmin=0.065, tmax=0.085, mode='abs', vert_as_index=False, time_as_index=False)
    peak_est = stc_est.get_peak(tmin=0.065, tmax=0.085, mode='abs', vert_as_index=False, time_as_index=False)
    
    #Find true positions 
    true_positions = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find corresponding vertex position in src (for estimated)
    for i in range(0, len(src_recon)):
        if i>1: 
            if peak_est[0] in src_recon[i]['vertno']:
                peak_est_pos = src_recon[i]['rr'][peak_est[0]]
                peak_est_vert = None
        else: 
            if peak_est[0] in src_recon[i]['vertno']:
                peak_est_vert = peak_est[0]
                peak_est_pos = None



    # #Create colormap based on stc value for each vertex 
    # data_time = stc_true_crop.data
    # data_time = np.mean(data_time, axis=1)
    # min_val, max_val = min(data_time), max(data_time)

    # # use the coolwarm colormap that is built-in, and goes from blue to red
    # cmap = matplotlib.cm.coolwarm
    # norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    
    # # convert your distances to color coordinates
    # color_list = cmap(data_time)

    #Plot on brain with vertices colored by signal 
    Brain = mne.viz.get_brain_class()
    brain = Brain(
        'fsaverage',
        hemi='both',
        surf='white',
        alpha=0.5,
        background='white',
        cortex='low_contrast',
        units='m',
        subjects_dir=subjects_dir,
    )

    #brain.add_foci(positions_true, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in label
    brain.add_foci(true_positions, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2, alpha=0.2) #vertices in label
    if peak_est_pos is not None: 
        brain.add_foci(peak_est_pos, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.6) #vertices in label
    if peak_est_vert is not None: 
        brain.add_foci(peak_est_vert, coords_as_verts=True, color='red', hemi='lh', scale_factor=0.6) #vertices in label
    brain.save_image(os.path.join(recon_path,"lcmv", "figures", f'peak_true_vs_est_thalamic_{extent}.png'))


    

############################################################################
#          OCCIPITAL - PLOT PEAK ACTIVATION AT OCCIPITAL PEAK (0.075)
# - in both true and estimated stc      
############################################################################
        
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/occipital_01nA'
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/occipital_01nA'
recon_path_lcmv = os.path.join(recon_path, 'lcmv')

stc_list_lcmv = [f for f in os.listdir(recon_path_lcmv) if '-stc.h5' in f]

src_recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/mixed_surfoct6_vols5.0_src.fif'
src_recon = mne.read_source_spaces(src_recon_path)

for stc in stc_list_lcmv: 

    extent = stc.split("-")[-3].split(".")[0]

    #Load estimated stc 
    stc_est = mne.read_source_estimate(os.path.join(recon_path_lcmv, stc))

    #Crop to start at 0 (epochs had -200 ms included)
    stc_est_crop = stc_est.crop(tmin=0, tmax=None)


    #Load true stc 
    stc_true_path = os.path.join(sim_path, stc.replace("-lcmv-stc.h5","-lh.stc"))
    stc_true = mne.read_source_estimate(stc_true_path)

    #Crop to have same length (=epoch) as estimated stc 
    stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*100,include_tmax=False)

    #Extract simulated vertices in true stc from thalamus 
    fwd_sim = mne.read_forward_solution(os.path.join(sim_path, 'ctx-lh-lateraloccipital-xx_mm-fsaverage-fwd.fif'))
    src_sim = mne.read_source_spaces(os.path.join(sim_path, 'ctx-lh-lateraloccipital-xx_mm-fsaverage-src.fif'))

    #Find time sample of thalamic peak (0.075) and V1 peak (0.95)
    thalamic_time_idx = int(0.075/stc_true.tstep)
    occipital_time_idx = int(0.095/stc_true.tstep)

    #Find position of peak vertex/ices in true 
    stc_true_max = stc_true.data[:,thalamic_time_idx].max() #all are the same in sims
    peak_true_pos = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find position of peak vertex in estimated 
    peak_true = stc_true.get_peak(tmin=0.085, tmax=0.105, mode='abs', vert_as_index=False, time_as_index=False)
    peak_est = stc_est.get_peak(tmin=0.085, tmax=0.105, mode='abs', vert_as_index=False, time_as_index=False)
    
    #Find true positions 
    true_positions = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find corresponding vertex position in src (for estimated)
    for i in range(0, len(src_recon)):
        if i>1: 
            if peak_est[0] in src_recon[i]['vertno']:
                peak_est_pos = src_recon[i]['rr'][peak_est[0]]
                peak_est_vert = None
        else: 
            if peak_est[0] in src_recon[i]['vertno']:
                peak_est_vert = peak_est[0]
                peak_est_pos = None



    # #Create colormap based on stc value for each vertex 
    # data_time = stc_true_crop.data
    # data_time = np.mean(data_time, axis=1)
    # min_val, max_val = min(data_time), max(data_time)

    # # use the coolwarm colormap that is built-in, and goes from blue to red
    # cmap = matplotlib.cm.coolwarm
    # norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    
    # # convert your distances to color coordinates
    # color_list = cmap(data_time)

    #Plot on brain with vertices colored by signal 
    Brain = mne.viz.get_brain_class()
    brain = Brain(
        'fsaverage',
        hemi='both',
        surf='white',
        alpha=0.5,
        background='white',
        cortex='low_contrast',
        units='m',
        subjects_dir=subjects_dir,
    )

    #brain.add_foci(positions_true, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in label
    brain.add_foci(true_positions, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2, alpha=0.2) #vertices in label
    if peak_est_pos is not None: 
        brain.add_foci(peak_est_pos, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.6) #vertices in label
    if peak_est_vert is not None: 
        brain.add_foci(peak_est_vert, coords_as_verts=True, color='red', hemi='lh', scale_factor=0.6) #vertices in label
    brain.save_image(os.path.join(recon_path,"lcmv", "figures", f'peak_true_vs_est_occipital_{extent}.png'))




############################################################################
#          MIXED - PLOT PEAK ACTIVATION AT OCCIPITAL PEAK (0.095) and thalamic peak (0.075)
# - in both true and estimated stc      
############################################################################
        
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/thalamic_1nA_occipital_01nA'
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/thalamic_1nA_occipital_01nA'
recon_path_lcmv = os.path.join(recon_path, 'lcmv')

stc_list_lcmv = [f for f in os.listdir(recon_path_lcmv) if '-stc.h5' in f]

src_recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/mixed_surfoct6_vols5.0_src.fif'
src_recon = mne.read_source_spaces(src_recon_path)

for stc in stc_list_lcmv: 

    extent_vol = stc.split("Thalamus-Proper-lh_")[1].split(".")[0]
    if "lateraloccipital-lh_" in stc: 
        extent_surf = stc.split("lateraloccipital-lh_")[1].split(".")[0]
    else: 
        extent_surf = "0"

    #Load estimated stc 
    stc_est = mne.read_source_estimate(os.path.join(recon_path_lcmv, stc))

    #Crop to start at 0 (epochs had -200 ms included)
    stc_est_crop = stc_est.crop(tmin=0, tmax=None)


    #Load true stc 
    stc_true_path = os.path.join(sim_path, stc.replace("-lcmv-stc.h5","-lh.stc"))
    stc_true = mne.read_source_estimate(stc_true_path)

    #Crop to have same length (=epoch) as estimated stc 
    stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*100,include_tmax=False)

    #Extract simulated vertices in true stc from thalamus 
    fwd_sim = mne.read_forward_solution(os.path.join(sim_path, 'Left-Thalamus-Proper_ctx-lh-lateraloccipital-xx_mm-fsaverage-fwd.fif'))
    src_sim = mne.read_source_spaces(os.path.join(sim_path, 'Left-Thalamus-Proper_ctx-lh-lateraloccipital-xx_mm-fsaverage-src.fif'))

    #Find time sample of thalamic peak (0.075) and V1 peak (0.95)
    thalamic_time_idx = int(0.075/stc_true.tstep)
    occipital_time_idx = int(0.095/stc_true.tstep)

    #Find position of peak vertex/ices in true 
    stc_true_max = stc_true.data[:,thalamic_time_idx].max() #all are the same in sims
    peak_true_pos = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find position of peak vertex in estimated 
    first_peak_true = stc_true.get_peak(tmin=0.065, tmax=0.085, mode='abs', vert_as_index=False, time_as_index=False)
    second_peak_true = stc_true.get_peak(tmin=0.085, tmax=0.105, mode='abs', vert_as_index=False, time_as_index=False)
    first_peak_est = stc_est.get_peak(tmin=0.065, tmax=0.085, mode='abs', vert_as_index=False, time_as_index=False)
    second_peak_est = stc_est.get_peak(tmin=0.085, tmax=0.105, mode='abs', vert_as_index=False, time_as_index=False)
    
    #Find true positions 
    true_positions = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find corresponding vertex position in src (for estimated)
    for i in range(0, len(src_recon)):
        if i>1: 
            if first_peak_est[0] in src_recon[i]['vertno']:
                first_peak_est_pos = src_recon[i]['rr'][first_peak_est[0]]
                first_peak_est_vert = None
            if second_peak_est[0] in src_recon[i]['vertno']:
                second_peak_est_pos = src_recon[i]['rr'][second_peak_est[0]]
                second_peak_est_vert = None
        else: 
            if first_peak_est[0] in src_recon[i]['vertno']:
                first_peak_est_vert = first_peak_est[0]
                first_peak_est_pos = None
            if second_peak_est[0] in src_recon[i]['vertno']:
                second_peak_est_vert = second_peak_est[0]
                second_peak_est_pos = None



    # #Create colormap based on stc value for each vertex 
    # data_time = stc_true_crop.data
    # data_time = np.mean(data_time, axis=1)
    # min_val, max_val = min(data_time), max(data_time)

    # # use the coolwarm colormap that is built-in, and goes from blue to red
    # cmap = matplotlib.cm.coolwarm
    # norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    
    # # convert your distances to color coordinates
    # color_list = cmap(data_time)

    #Plot on brain with vertices colored by signal 
    Brain = mne.viz.get_brain_class()
    brain = Brain(
        'fsaverage',
        hemi='both',
        surf='white',
        alpha=0.5,
        background='white',
        cortex='low_contrast',
        units='m',
        subjects_dir=subjects_dir,
    )

    #brain.add_foci(positions_true, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) #vertices in label
    brain.add_foci(true_positions, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2, alpha=0.2) #vertices in label
    if first_peak_est_pos is not None: 
        brain.add_foci(first_peak_est_pos, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.6) #vertices in label
    if second_peak_est_pos is not None: 
        brain.add_foci(second_peak_est_pos, coords_as_verts=False, color='darkgreen', hemi='lh', scale_factor=0.6) #vertices in label
    if first_peak_est_vert is not None: 
        brain.add_foci(first_peak_est_vert, coords_as_verts=True, color='red', hemi='lh', scale_factor=0.6) #vertices in label
    if second_peak_est_vert is not None: 
        brain.add_foci(second_peak_est_vert, coords_as_verts=True, color='darkgreen', hemi='lh', scale_factor=0.6)
    brain.save_image(os.path.join(recon_path,"lcmv", "figures", f'peak_true_vs_est_thalamus_{extent_vol}mm_occipital_{extent_surf}mm.png'))




############################################################################
#      MIXED - PLOT DISTANCE between TRUE and ESTIMATED thalamic peak 
#           and occipital peak as function of V1 size 
# - one figure per thalamic patch size (2, 5, 8, 10 and 15 mm)    
############################################################################
        
sim_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/thalamic_1nA_occipital_01nA'
recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/thalamic_1nA_occipital_01nA'
recon_path_lcmv = os.path.join(recon_path, 'lcmv')

stc_list_lcmv = [f for f in os.listdir(recon_path_lcmv) if '-stc.h5' in f]

src_recon_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/reconstructions/mixed_surfoct6_vols5.0_src.fif'
src_recon = mne.read_source_spaces(src_recon_path)

src_sim = mne.read_source_spaces(os.path.join(sim_path, 'Left-Thalamus-Proper_ctx-lh-lateraloccipital-xx_mm-fsaverage-src.fif'))

fname_aseg = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/fsaverage/mri/aparc+aseg.mgz'
regions = ["Left-Thalamus-Proper", "ctx-lh-lateraloccipital"]
source_pos = get_vol_label_vertices(fname_aseg, volume_labels=regions)
thalamus_pos = source_pos[0]
occipital_pos = source_pos[1]
thalamic_vertices = src_sim[0]['vertno'][0:len(thalamus_pos)]
occipital_vertices = src_sim[0]['vertno'][len(thalamus_pos):]

vol_extent_list = []
surf_extent_list = []
n_vert_sim_thalamic_list = []
n_vert_sim_occipital_list = []
dist_thalamic_peak_list = []
dist_occipital_peak_list = []

for stc in stc_list_lcmv: 

    v1_present = False
    extent_vol = stc.split("Thalamus-Proper-lh_")[1].split(".")[0]
    if "lateraloccipital-lh_" in stc: 
        extent_surf = stc.split("lateraloccipital-lh_")[1].split(".")[0]
        v1_present=True
    else: 
        extent_surf = "0"

    #Load estimated stc 
    stc_est = mne.read_source_estimate(os.path.join(recon_path_lcmv, stc))

    #Crop to start at 0 (epochs had -200 ms included)
    stc_est_crop = stc_est.crop(tmin=0, tmax=None)


    #Load true stc 
    stc_true_path = os.path.join(sim_path, stc.replace("-lcmv-stc.h5","-lh.stc"))
    stc_true = mne.read_source_estimate(stc_true_path)

    #Crop to have same length (=epoch) as estimated stc 
    stc_true = stc_true.crop(tmin=None, tmax=stc_true.tstep*100,include_tmax=False)

    #Find time sample of thalamic peak (0.075) and V1 peak (0.95)
    thalamic_time_idx = int(0.075/stc_true.tstep)
    occipital_time_idx = int(0.095/stc_true.tstep)

    #Get positions from the vertices acutally activated in simulation 
    stc_thalamic_vertices = [v for v in stc_true.vertices[0] if v in thalamic_vertices]
    stc_occipital_vertices = [v for v in stc_true.vertices[0] if v in occipital_vertices]
    n_vert_sim_thalamic = len(stc_thalamic_vertices)
    n_vert_sim_occipital = len(stc_occipital_vertices)

    thalamic_true_pos = src_sim[0]['rr'][stc_thalamic_vertices]
    thalamic_true_pos = thalamic_true_pos.mean(axis=0) #get centroid 

    if v1_present:
        occipital_true_pos = src_sim[0]['rr'][stc_occipital_vertices]
        occipital_true_pos = occipital_true_pos.mean(axis=0) #get centroid 
    else: 
        occipital_true_pos = None
    
    #Find position of peak vertex in estimated 
    first_peak_est = stc_est.get_peak(tmin=0.065, tmax=0.085, mode='abs', vert_as_index=False, time_as_index=False)
    second_peak_est = stc_est.get_peak(tmin=0.085, tmax=0.105, mode='abs', vert_as_index=False, time_as_index=False)
    
    #Find true positions 
    true_positions_all = src_sim[0]['rr'][stc_true.vertices[0]]

    #Find corresponding vertex position in src (for estimated)
    for i in range(0, len(src_recon)): 
        #loop through srcs in src_recon (one per surf/volume) to find which one the peak vertex is in + get the position 
        if i==1: #lh and rh surfs have same vertex numbers - only looking in lh here 
            continue
        else: 
            if first_peak_est[0] in src_recon[i]['vertno']:
                first_peak_est_pos = src_recon[i]['rr'][first_peak_est[0]]
            if second_peak_est[0] in src_recon[i]['vertno']:
                second_peak_est_pos = src_recon[i]['rr'][second_peak_est[0]]
        
    #Comptue eucledian distance between true and estimated for each peak 
    from scipy.spatial.distance import cdist
    dist_thalamic_peak = cdist([thalamic_true_pos], [first_peak_est_pos], metric="euclidean")[0][0]
    if v1_present: 
        dist_occipital_peak = cdist([occipital_true_pos], [second_peak_est_pos], metric="euclidean")[0][0]
    else: 
        dist_occipital_peak = 0.0

    #Append to lists 
    vol_extent_list.append(extent_vol)
    surf_extent_list.append(extent_surf)
    n_vert_sim_thalamic_list.append(n_vert_sim_thalamic)
    n_vert_sim_occipital_list.append(n_vert_sim_occipital)
    dist_thalamic_peak_list.append(dist_thalamic_peak)
    dist_occipital_peak_list.append(dist_occipital_peak)

peak_dist_df = pd.DataFrame({
    'vol_extent': vol_extent_list, 
    'surf_extent': surf_extent_list,
    'n_vert_thal': n_vert_sim_thalamic_list,
    'n_vert_occipital': n_vert_sim_occipital_list,
    'dist_thalamic_peak': dist_thalamic_peak_list,
    'dist_occipital_peak': dist_occipital_peak_list
})
peak_dist_df.to_csv(os.path.join(recon_path, 'dist_peaks_thalamus_1nA_occipital_01nA.csv'))

#Plot - one fig per thalamic extent (one line per temporal peak)
peak_dist_df.surf_extent = pd.to_numeric(peak_dist_df.surf_extent)
peak_dist_df.vol_extent = pd.to_numeric(peak_dist_df.vol_extent)


fig, ax = plt.subplots(2,3, figsize=(12,6), sharey=True, sharex=True)
sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==2], x="surf_extent", y="dist_thalamic_peak", color='red', label='Thalamic peak (75 ms)', ax=ax[0,0])
sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==2], x="surf_extent", y="dist_occipital_peak", color='darkgreen', label='Occipital peak (95 ms)', ax=ax[0,0])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==2], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[0,0])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==2], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[0,0])
ax[0,0].set_xlabel('V1 extent activated')
ax[0,0].set_ylabel('Euclidean distance')
ax[0,0].set_title("Thalamic 2 mm")

sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==5], x="surf_extent", y="dist_thalamic_peak", color='red',  ax=ax[0,1])
sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==5], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[0,1])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==5], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[0,1])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==5], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[0,1])
ax[0,1].set_title("Thalamic 5 mm")

sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==8], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[0,2])
sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==8], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[0,2])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==8], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[0,2])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==8], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[0,2])
ax[0,2].set_title("Thalamic 8 mm")

sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==10], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[1,0])
sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==10], x="surf_extent", y="dist_occipital_peak", color='darkgreen',  ax=ax[1,0])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==10], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[1,0])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==10], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[1,0])
ax[1,0].set_xlabel('V1 extent activated')
ax[1,0].set_ylabel('Euclidean distance')
ax[1,0].set_title("Thalamic 10 mm")

sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==15], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[1,1])
sns.lineplot(data=peak_dist_df[peak_dist_df.vol_extent==15], x="surf_extent", y="dist_occipital_peak", color='darkgreen',  ax=ax[1,1])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==15], x="surf_extent", y="dist_thalamic_peak", color='red', ax=ax[1,1])
sns.scatterplot(data=peak_dist_df[peak_dist_df.vol_extent==15], x="surf_extent", y="dist_occipital_peak", color='darkgreen', ax=ax[1,1])
ax[1,1].set_xlabel('V1 extent activated')
ax[1,1].set_title("Thalamic 15 mm")

plt.suptitle("Distance bewteen true and estimated source peaks", fontsize=15)
plt.savefig(os.path.join(recon_path, 'dist_peaks_by_v1_size.png'))
plt.show()


#Plot - one fig per temporal peak (one line per thalamic size)
palette1 = sns.color_palette("flare", 5)
palette2 = sns.color_palette("crest", 5)

fig, ax = plt.subplots(1,2, figsize=(12,6), sharey=True, sharex=False)

sns.lineplot(data=peak_dist_df, x="surf_extent", y="dist_thalamic_peak", hue='vol_extent', palette=palette1,ax=ax[0])
sns.scatterplot(data=peak_dist_df, x="surf_extent", y="dist_thalamic_peak", hue='vol_extent', palette=palette1, legend=None, ax=ax[0])
ax[0].set_xlabel('V1 extent activated')
ax[0].set_ylabel('Euclidean distance')
ax[0].set_title("Thalamic peak (75 ms)")
sns.move_legend(ax[0], title='Thalamic extent (mm)', loc='best')

sns.lineplot(data=peak_dist_df, x="surf_extent", y="dist_occipital_peak", hue='vol_extent', palette=palette2,  ax=ax[1])
sns.scatterplot(data=peak_dist_df, x="surf_extent", y="dist_occipital_peak", hue='vol_extent', palette=palette2, legend=None, ax=ax[1])
ax[1].set_xlabel('V1 extent activated')
ax[1].set_title("Occipital peak (95 ms)")
sns.move_legend(ax[1], title='Thalamic extent (mm)', loc='best')

plt.suptitle("Distance bewteen true and estimated source peaks", fontsize=15)
plt.savefig(os.path.join(recon_path, 'dist_peaks_by_v1_size_2.png'))
plt.show()

Brain = mne.viz.get_brain_class()
brain = Brain(
    'fsaverage',
    hemi='both',
    surf='white',
    alpha=0.5,
    background='white',
    cortex='low_contrast',
    units='m',
    subjects_dir=subjects_dir,
)

brain.add_foci(thalamic_true_pos, coords_as_verts=False, color='blue', hemi='lh', scale_factor=0.2)
brain.add_foci(occipital_true_pos, coords_as_verts=False, color='red', hemi='lh', scale_factor=0.2) 