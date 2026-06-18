
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

############################################################################
#              COMPUTE REGION LOCALIZATIONE ERROR (RLE)    
############################################################################

recon_path = os.path.join(dir,'data/reconstructions')
sim_path = os.path.join(dir,'data/simulations')

#thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
thresholds=[70]

#Recon for all sims used the same fwd/src 
fwd_recon = mne.read_forward_solution(os.path.join(recon_path, 'mixed_surfoct6_vols5.0_fwd.fif'))
src_recon = fwd_recon['src']

#### OCCIPITAL ONLY #### 
sims_list = ['test_increasing_snr_methods/occpitial_0.1nA_increasing_size', 'test_increasing_snr_methods/occpitial_onedip_increasing_amplitude']

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


df_rle_surf = pd.DataFrame({'region':'occipital', 
                       'amplitude': amplitude_list,
                       'patch_size':extent_list,
                       'n_vertices_sim':n_vertices_list,
                       'threshold': threshold_list,
                       'rle_mne': rle_mne_list,
                       'rle_lcmv':rle_lcmv_list})
df_rle_surf['name'] = "size"
df_rle_surf.name[df_rle_surf.patch_size=='0.0'] = 'amplitude'
df_rle_surf.to_csv(os.path.join('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/rle.csv'))

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

###### Occipital - 0.1 nA, increasing patch size  
recon_path = os.path.join(dir,'data/reconstructions')
sim_path = os.path.join(dir,'data/simulations')
sim = 'test_increasing_snr_methods/occpitial_0.1nA_increasing_size'
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


df_snr_size = pd.DataFrame({'region':"occipital", 
                        'amplitude':0.1,
                       'patch_size': patch_size,
                       'snr_mean': snr_mean_list,
                       'snr_max':snr_max_list})
df_snr_size.to_csv(os.path.join(recon_path, sim, 'snr_01nA_increasing_size.csv'))
df_snr_size = pd.read_csv(os.path.join(recon_path, sim, 'snr_01nA_increasing_size.csv'))

###### Occipital, one dipole, increasing amplitude 
recon_path = os.path.join(dir,'data/reconstructions')
sim_path = os.path.join(dir,'data/simulations')
sim = 'test_increasing_snr_methods/occpitial_onedip_increasing_amplitude'
evokeds = [f for f in os.listdir(os.path.join(sim_path, sim)) if "-ave.fif" in f]
region = 'occipital'
patch_size = []
amplitude_list = []
snr_max_list = []
snr_mean_list = []
for evo in evokeds: 
    extent = evo.split("-")[-2].split("_")[0]
    patch_size.append(extent)
    amplitude = evo.split('_nA')[0].split('-')[-1]
    amplitude_list.append(amplitude)
    evoked = mne.read_evokeds(os.path.join(sim_path, sim, evo), baseline=(None, 0))[0]
    inv_fname = evo.replace("-ave.fif", "-inv.fif")
    inv_path = os.path.join(recon_path, sim, "mne", inv_fname)
    inv = mne.minimum_norm.read_inverse_operator(inv_path)
    snr = mne.minimum_norm.estimate_snr(evoked, inv, verbose=None)[0]
    snr_max_list.append(snr.max())
    snr_mean_list.append(snr.mean())

    # plt.figure()
    # fig = mne.viz.plot_snr_estimate(evoked, inv, show=False)
    # fig.savefig(os.path.join(recon_path, sim, f"snr_plot_{region}_{extent}.png"))


df_snr_amplitude = pd.DataFrame({'region':region,
                        'amplitude': amplitude_list,
                       'patch_size': patch_size,
                       'snr_mean': snr_mean_list,
                       'snr_max':snr_max_list})
df_snr_amplitude.to_csv(os.path.join(recon_path, sim, 'snr_increasing_amplitude.csv'))
df_snr_amplitude = pd.read_csv(os.path.join(recon_path, sim, 'snr_increasing_amplitude.csv'))


df_snr_size['name'] = 'size'
df_snr_amplitude['name'] = 'amplitude'
df_snr = pd.concat((df_snr_size, df_snr_amplitude))
df_snr.patch_size = pd.to_numeric(df_snr.patch_size)
df_snr.to_csv(os.path.join(recon_path, 'test_increasing_snr_methods/snr.csv'))

# ###### Thalamic + Occipital (mixed)
# recon_path = os.path.join(dir,'data/reconstructions')
# sim_path = os.path.join(dir,'data/simulations')
# sim = 'thalamic_1nA_occipital_01nA'
# evokeds = [f for f in os.listdir(os.path.join(sim_path, sim)) if "-ave.fif" in f]
# region = 'thalamus_occipital'
# patch_size_surf = []
# patch_size_vol = []
# snr_max_list = []
# snr_mean_list = []
# for evo in evokeds: 
#     extent_vol = evo.split("Proper-lh_")[1].split("_")[0]
#     if "lateraloccipital" in evo:
#         extent_surf = evo.split("lateraloccipital-lh_")[1].split("_")[0]
#     else: 
#         extent_surf = str(0.0)
#     patch_size_vol.append(extent_vol)
#     patch_size_surf.append(extent_surf)
#     evoked = mne.read_evokeds(os.path.join(sim_path, sim, evo), baseline=(None, 0))[0]
#     inv_fname = evo.replace("-ave.fif", "-inv.fif")
#     inv_path = os.path.join(recon_path, sim, "mne", inv_fname)
#     inv = mne.minimum_norm.read_inverse_operator(inv_path)
#     snr = mne.minimum_norm.estimate_snr(evoked, inv, verbose=None)[0]
#     snr_max_list.append(snr.max())
#     snr_mean_list.append(snr.mean())

#     plt.figure()
#     fig = mne.viz.plot_snr_estimate(evoked, inv, show=False)
#     fig.savefig(os.path.join(recon_path, sim, f"snr_plot_thalamus{extent_vol}_occipital{extent_surf}.png"))


# df_snr_mix = pd.DataFrame({'region':region,
#                        'patch_size_surf': patch_size_surf,
#                        'patch_size_vol': patch_size_vol,
#                        'snr_mean': snr_mean_list,
#                        'snr_max':snr_max_list})
# df_snr_mix.to_csv(os.path.join(recon_path, sim, 'snr_mix.csv'))


##Plot SNR for increasing size vs increasing volume 
df_snr_size.patch_size = pd.to_numeric(df_snr_size.patch_size)
df_snr_amplitude.amplitude = pd.to_numeric(df_snr_amplitude.amplitude)
fig, ax = plt.subplots(1, 2, sharey=True)
sns.lineplot(data=df_snr_size, x='patch_size', y='snr_max', color='red', ax=ax[0])
sns.scatterplot(data=df_snr_size, x='patch_size', y='snr_max', color='red', ax=ax[0])
sns.lineplot(data=df_snr_amplitude, x='amplitude', y='snr_max', color='blue', ax=ax[1])
sns.scatterplot(data=df_snr_amplitude, x='amplitude', y='snr_max', color='blue', ax=ax[1])
plt.savefig('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/snr.png')
plt.show()

#Plot RLE as function of SNR 
df_rle = df_rle_surf.copy()
df_rle.amplitude = pd.to_numeric(df_rle.amplitude)
df_rle.patch_size = pd.to_numeric(df_rle.patch_size)
df = pd.merge(df_snr[['name','amplitude','patch_size','snr_mean','snr_max']], df_rle, on=['name','amplitude','patch_size'])
df.to_csv('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/rle_by_snr.csv')

plt.figure()
sns.lineplot(data=df, x='snr_max',y='rle_lcmv', hue='name')
sns.scatterplot(data=df, x='snr_max',y='rle_lcmv', hue='name', legend=None)
plt.legend(title='Method', loc='upper right')
plt.ylabel('RLE (mm)')
plt.xlabel('SNR (max)')
plt.suptitle('LCMV Reconstruction')
plt.savefig('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/rle_by_snr_lcmv.png')
plt.show()

plt.figure()
sns.lineplot(data=df[df.snr_max<5.0], x='snr_max',y='rle_lcmv', hue='name')
sns.scatterplot(data=df[df.snr_max<5.0], x='snr_max',y='rle_lcmv', hue='name', legend=None)
plt.legend(title='Method', loc='upper right')
plt.ylabel('RLE (mm)')
plt.xlabel('SNR (max)')
plt.suptitle('LCMV Reconstruction')
plt.savefig('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/rle_by_snr_lcmv_zoom.png')
plt.show()

plt.figure()
sns.lineplot(data=df, x='snr_max',y='rle_mne', hue='name')
sns.scatterplot(data=df, x='snr_max',y='rle_mne', hue='name', legend=None)
plt.legend(title='Method', loc='upper right')
plt.ylabel('RLE (mm)')
plt.xlabel('SNR (max)')
plt.suptitle('MNE Reconstruction')
plt.savefig('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/rle_by_snr_mne.png')
plt.show()

plt.figure()
sns.lineplot(data=df[df.snr_max<5.0], x='snr_max',y='rle_mne', hue='name')
sns.scatterplot(data=df[df.snr_max<5.0], x='snr_max',y='rle_mne', hue='name', legend=None)
plt.legend(title='Method', loc='upper right')
plt.ylabel('RLE (mm)')
plt.xlabel('SNR (max)')
plt.suptitle('MNE Reconstruction')
plt.savefig('/Volumes/Elements/simulation_cortical_omission/data/reconstructions/test_increasing_snr_methods/rle_by_snr_mne_zoom.png')
plt.show()

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