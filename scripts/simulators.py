import mne 
import os 
import numpy as np 
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
import os


class Simulator():

    def __init__(self):
        self.dir = os.getcwd().replace('/scripts','')
        #os.chdir(os.path.join(dir, 'scripts'))
        #os.chdir(dir)
        pass

    def set_params(self, output_path, sensor_array='squids', n_axis=2):
        opm_str = 'single' if n_axis==1 else 'dual'
        
        self.random_state = 42
        self.sensor_array = sensor_array

        #Paths 
        self.raw_fname =  os.path.join(self.dir,'data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif')
        self.raw_opm_info_fname = os.path.join(self.dir, f'data/OPM/fsaverage_OPM_alpha1_{opm_str}_axis-info.fif')
        self.subject = 'fsaverage'
        self.subjects_dir = os.path.join(self.dir, 'data/freesurfer/subjects')
        self.fname_trans = 'fsaverage' #use built-in trans file for fsaverage 
        self.fname_bem = os.path.join(self.subjects_dir, self.subject, 'bem','fsaverage-5120-5120-5120-bem-sol.fif')
        self.fname_aseg = os.path.join(self.dir, 'data/freesurfer/fsaverage/mri/aparc+aseg.mgz')
        if sensor_array=='squids':
            self.fname_cov = os.path.join(self.dir, 'data/MNE-sample-data/MEG/sample/sample_audvis-cov.fif')
        elif sensor_array=='opm':
            #self.fname_cov = os.path.join(self.dir, 'data/MNE-sample-data/MEG/sample/sample_audvis-cov.fif')
            self.fname_cov = None
        else: 
            raise ValueError('Sensory array must be one of ["squids", "opm"]')
        self.output_path = output_path
        self.figure_path = os.path.join(self.output_path, 'figures')

        if not os.path.exists(self.output_path): 
            os.mkdir(self.output_path)
        if not os.path.exists(self.figure_path):
            os.mkdir(self.figure_path)

        self.extents_vol = None
        self.extents_surf = None
        self.source_labels_vol = None
        self.source_labels_surf = None
        self.seeds = dict()


    def create_info_obj(self):
        self.sample_raw = mne.io.read_raw_fif(self.raw_fname)
        if self.sensor_array=='squids':
            self.info = self.sample_raw.pick(picks=['meg']).info
        elif self.sensor_array=='opm':
            self.info = mne.io.read_info(self.raw_opm_info_fname)
        else: 
            raise ValueError(f'No raw info object available for {self.sensor_array} sensor array')

    
    def data_fun(self, amplitude, latency=0):
            """Generate random source time courses.
            #FIXME latency now is relative to the peak of the original wave (i.e., the one used for surf sims)
            #latency should be inputted in s 
            """
            rng = np.random.RandomState(42)
            self.amp_nA = amplitude 
            amp = amplitude*1e-9
            timeseries = (amp #50e-9 = 50 nAm
                * np.sin(20.0 * self.times)
                * np.exp(-((self.times - 0.15 + 0.05 * rng.randn(1)) ** 2) / 0.01)
            )
            if not latency==0: 
                shift_samples = int(latency/self.tstep)
                timeseries = timeseries[shift_samples:] 
                timeseries = np.concatenate((timeseries, (np.repeat(timeseries[-1], shift_samples))))
                timeseries[0] = 0.0
                diff = timeseries[5]-timeseries[0]
                inc = diff/5
                timeseries[1] = timeseries[0]+inc
                timeseries[2] = timeseries[1]+inc
                timeseries[3] = timeseries[2]+inc
                timeseries[4] = timeseries[3]+inc

            return(timeseries)
    

    def create_time_series(self, amplitude_surf, amplitude_vol):

        #self.tstep = 1.0 / self.info['sfreq']
        self.tstep = 1.0 / 150.15374755859375 #sfreq in mne sample raw.info - used in all sims currently 
        self.times = np.arange(100, dtype=np.float64)*self.tstep

        n_events = 100
        events = np.zeros((n_events, 3), int)
        events[:, 0] = 200 * np.arange(n_events)  # Events sample.
        events[:, 2] = 1  # All events have the sample id.

        self.events = events

        if not amplitude_surf is None: 
            self.source_time_series_surf = self.data_fun(amplitude=amplitude_surf)
        else:
            self.source_time_series_surf = None
        if not amplitude_vol is None: 
            self.source_time_series_vol = self.data_fun(amplitude=amplitude_vol, latency=0.02) #will peak 0.02 sec (20 ms) before surface peak 
        else: 
            self.source_time_series_vol = None

    def generate_src(self, vol_labels, save=True, plot=True, overwrite=True):
        """Generate discrete src with dipole positions inputted as coords in unit 'm' 
        - Will use all subcortical labels provided in vol_labels
        - The source_simulator functions add a simulated STC for both hemispheres no matter what 
        (one will just be empty if only label from one hemi is provided). This creates an error if the src only 
        has one input (only one hemi). So even if just given a label for one hemi, we are always creating an src 
        for that label in both hemispheres. 
        """

        from helper_functions import get_vol_label_vertices

        #Check that vol_labels is in list 
        if isinstance(vol_labels, str): 
            vol_labels = [vol_labels]

        #Extract vertices from aseg file for each label and create pos dict 
        self.label_vertices = dict()

        vol_labels_lh = [l for l in vol_labels if any(m in l.lower() for m in ['lh','left'])]
        vol_labels_rh = [l for l in vol_labels if any(m in l.lower() for m in ['rh','right'])]

        #If one hemi is not in labels, use the same label as for the other hemi 
        if len(vol_labels_lh)==0: 
            vol_labels_lh = []
            for lab in vol_labels_rh: 
                if "rh" in lab: 
                    vol_labels_lh.append(str(lab.replace("rh", "lh")))
                elif "Right" in lab: 
                    vol_labels_lh.append(str(lab.replace("Right", "Left")))
                else: 
                    raise ValueError("Label names in vol_labels do not have the correct hemi annotation. Must be one of (lh, rh, Left, Right)")

        if len(vol_labels_rh)==0: 
            vol_labels_rh = []
            for lab in vol_labels_lh: 
                if "lh" in lab: 
                    vol_labels_rh.append(str(lab.replace("lh", "rh")))
                elif "Left" in lab: 
                    vol_labels_rh.append(str(lab.replace("Left", "Right")))
                else: 
                    raise ValueError("Label names in vol_labels do not have the correct hemi annotation. Must be one of (lh, rh, Left, Right)")

        # SETUP SRC FOR LH 
        #Load volume vertices from aseg file and transform positions to MRI (RAS) coords in units m 
        #Returns list of vertex positions for each label in vol_labels_hemi
        verts_list_lh = get_vol_label_vertices(self.fname_aseg, vol_labels_lh, units='m') 
        for i, name in enumerate(vol_labels_lh): 
            self.label_vertices[name] = verts_list_lh[i]

        #Create pos dict with rr and nn positions of vertices for the labels to use 
        pos_lh = dict()
        for i in range(0, len(vol_labels_lh)):
            if i==0: 
                rr_concat = verts_list_lh[i]
            else: 
                rr_concat = np.concatenate((rr_concat, verts_list_lh[i]))
        pos_lh['rr'] = rr_concat
        nn_lh = rr_concat.copy() #copy shape 
        nn_lh[:,:] = [0.,0.,1.] #replace values with orientation values (using volume src default here)
        pos_lh['nn'] = nn_lh

        #Setup source space for those vertices 
        src_lh = mne.setup_volume_source_space(
            subject='fsaverage',
            mri = None, 
            pos = pos_lh, 
            bem=self.fname_bem,
            subjects_dir=self.subjects_dir,
            sphere_units="m",
        )

        # SETUP SRC FOR RH 
        #Load volume vertices from aseg file and transform positions to MRI (RAS) coords in units m 
        #Returns list of vertex positions for each label in vol_labels_hemi
        verts_list_rh = get_vol_label_vertices(self.fname_aseg, vol_labels_rh, units='m') 
        for i, name in enumerate(vol_labels_rh): 
            self.label_vertices[name] = verts_list_rh[i]

        #Create pos dict with rr and nn positions of vertices for the labels to use 
        pos_rh = dict()
        for i in range(0, len(vol_labels_rh)):
            if i==0: 
                rr_concat = verts_list_rh[i]
            else: 
                rr_concat = np.concatenate((rr_concat, verts_list_rh[i]))
        pos_rh['rr'] = rr_concat
        nn_rh = rr_concat.copy() #copy shape 
        nn_rh[:,:] = [0.,0.,1.] #replace values with orientation values (using volume src default here)
        pos_rh['nn'] = nn_rh

        #Setup source space for those vertices 
        src_rh = mne.setup_volume_source_space(
            subject='fsaverage',
            mri = None, 
            pos = pos_rh, 
            bem=self.fname_bem,
            subjects_dir=self.subjects_dir,
            sphere_units="m",
        )

        src_vol = src_lh + src_rh
        self.src = src_vol 
        self.labels_str = "_".join(l for l in vol_labels_lh)

        #Save 
        self.fname_src = f'{self.labels_str}-fsaverage-src.fif'
        if save: 
            mne.write_source_spaces(os.path.join(self.output_path, self.fname_src), src_vol, overwrite=overwrite)

        #Plot 
        if plot: 
            fig_name = f'src-{self.labels_str}.png'
            fig = mne.viz.plot_bem(src=src_vol, subject=self.subject, subjects_dir=self.subjects_dir, show=False)
            fig.savefig(os.path.join(self.figure_path, fig_name))
            plt.close()
            

    def generate_fwd(self, save=True, overwrite=True):
            self.fwd_fname = f'{self.labels_str}-fsaverage-fwd.fif'
            src = mne.read_source_spaces(os.path.join(self.output_path,self.fname_src))
            self.fwd = mne.make_forward_solution(self.info, self.fname_trans, src, self.fname_bem, mindist=5.0)
            
            if save: 
                mne.write_forward_solution(os.path.join(self.output_path, self.fwd_fname), self.fwd, overwrite=overwrite)

    def grow_sim_source_label(self, labels, seeds=None, extent=2.0, label_type=None):
        """ Find center of mass of labels, and grow a new label includign all vertices wihtin distance (extent).
        Currently based on seed defined as the center of mass of the full region if seed is None. Com can be computed 
        using median or mean of the vertex coords. 

        label: the label from which to grow the label (and from where the seed origins, if provided)
        seed: seed to use as starting point, if None then the center of mass is computed and used as seed 
        extent: (int | "full") size of resulting label in mm, if full, returning a label including all vertices in the region (from aseg file)
        """
        from helper_functions import _center_of_mass, grow_labels

        if label_type=='vol':
            self.extents_vol = extent
        elif label_type=='surf':
            self.extents_surf = extent
        else: 
            raise ValueError("label_type must be one of ['vol', 'surf']")

        if isinstance(labels, str):
            labels = [labels]


        #Check that labels are actually in src 
        labs_in_src = [name for name in self.label_vertices.keys()]
        for l in labels: 
            if not l in labs_in_src: 
                raise KeyError(f"Source label {l} is not present in src")
            
        #Check if seed vertex number must be updated (if other labels occur in src[hemi] prior to the surf label)
        # - As labels vertices are added to the src, they get continuous vertex numbers (so first label added iwth have vertex numbers starting from 0, next label added with have numbers starting from len of first label, etc.)
        # - Below, when finding the cneter of mass to grow from we are only using the vertices within the specific label of interest,
        #   hence, the vertex number of the returned seed will be relative to the number of vertices in that label only 
        # - Thus, if the src of one hemi has more than one label (when simulating two structures), the seed number will only fit if that label was the first one added
        # - Here, checking whether there are any labels in src occurring before the current label in question, and in that case 
        #   extracting the number of vertices in those to be added to the seed number later 
        # - In the case that the current label in question is the only one simulated (only one in src) the seed number will stay the same
        # - NB: Checking and updating only left hemisphere, as these are the only labels we are simulating 
        lh_labels = [l for l in labs_in_src if any(m in l.lower() for m in ['lh','left'])]
        #get idx of current label, i.e., order in which it was added to the src (if not 0, seed vertno must be updated with length of vertices added before this)
        idx_map = {key: i for i, key in enumerate(self.label_vertices)}
        idx_label = idx_map.get(labels[0])
        #loop through all label keys before current label in src, and add the number of vertices together 
        seed_update=0
        for i in range(0,idx_label):
            seed_update =+ len(self.label_vertices[labs_in_src[i]])


        #Check which hemis are present in labels (and for which to grow labels)
        hemis_in_labels = []
        if any(l for l in labels if any(m in l.lower() for m in ['lh','left'])):
            hemis_in_labels.append('lh')
        if any(l for l in labels if any(m in l.lower() for m in ['rh','right'])):
            hemis_in_labels.append('rh')
        print(f'- Hemis in labels: {hemis_in_labels}')

        if not seeds: 
            #Compute center of mass 
            seeds = []
            for l in labels: 
                seed = _center_of_mass(self.label_vertices[l])
                seed = seed + seed_update
                seeds.append(seed)
                self.seeds[l] = seed

        #FIXME - check hwo this works wtih the hemis param - should select hemis and grow label based on hemis indicated in label name 
                # - also check that it actually matches the extent and label and names pairwise
        source_labels = grow_labels(
            self.fwd['src'], 
            #self.src,
            self.subject,
            seeds=seeds, #one seed per label per hemis
            extents=extent, #either one value per label, otherwise it will use same extent for each label 
            hemis=hemis_in_labels,
            subjects_dir=self.subjects_dir,
            n_jobs=None,
            names=labels,
            colors=None,
        )

        if label_type == 'vol':
            self.source_labels_vol = source_labels
        elif label_type == 'surf':
            self.source_labels_surf = source_labels


    def initiate_sourcesimulator(self):        
        self.source_simulator = mne.simulation.SourceSimulator(self.fwd['src'], tstep=self.tstep)


    def add_to_sourcesimulator(self, labels="all", save=True, overwrite=True):
        """
        FIXME: edit so that we can specify which source time series to sim for which label 
        Currently just adding the same generic one to each label of same type (surf or vol)
        """
        if not self.source_labels_vol is None: 
            str1 = "-".join([self.source_labels_vol[i].name + f'_{self.extents_vol}_mm' for i, x in enumerate(self.source_labels_vol)])
        else: 
            str1 = ""
        if not self.source_labels_surf is None: 
            str2 = "-".join([self.source_labels_surf[i].name + f'_{self.extents_surf}_mm' for i, x in enumerate(self.source_labels_surf)])
        else: 
            str2 = ""
        self.stc_fname = f"{str1}-{str2}"

        if labels=="all":
            if self.source_labels_vol is not None: 
                for vol_source in self.source_labels_vol: 
                    print(f"-- Adding volume time series for volume {vol_source}")
                    self.source_simulator.add_data(vol_source, self.source_time_series_vol, self.events)
            if self.source_labels_surf is not None: 
                for surf_source in self.source_labels_surf: 
                    print(f"--Adding surface time series for cortical region {surf_source}")
                    self.source_simulator.add_data(surf_source, self.source_time_series_surf, self.events)
        elif labels=="surf":
            for surf_source in self.source_labels_surf: 
                self.source_simulator.add_data(surf_source, self.source_time_series_surf, self.events)
        elif labels=="vol":
            for vol_source in self.source_labels_vol: 
                self.source_simulator.add_data(vol_source, self.source_time_series_vol, self.events)
        else: #only add those provided in input parameter labels 
            self.source_labels_all = [self.source_labels_vol, self.source_labels_surf]
            for label_source in self.source_labels_all: 
                if label_source.name in labels: 
                    if label_source.name in self.source_labels_vol: 
                        self.source_simulator.add_data(label_source, self.source_time_series_vol, self.events)
                    elif label_source.name in self.source_labels_surf: 
                        self.source_simulator.add_data(label_source, self.source_time_series_surf, self.events)

        if save: 
            self.source_simulator.get_stc().save(os.path.join(self.output_path, self.stc_fname), overwrite=overwrite)


    def add_eog_no_sphere(self, head_pos=None, interp="cos2",n_jobs=None, verbose=None):
        from helper_functions import _add_exg_no_sphere
        return _add_exg_no_sphere(self.raw, "blink", head_pos, interp, n_jobs, self.random_state, self.sensor_array, self.fname_bem)

    def add_ecg_no_sphere(self, head_pos=None, interp="cos2",n_jobs=None, verbose=None):
        from helper_functions import _add_exg_no_sphere
        return _add_exg_no_sphere(self.raw, "ecg", head_pos, interp, n_jobs, self.random_state, self.sensor_array, self.fname_bem)


    def sim_raw(self, add_iir=True, add_eog=True, add_ecg=True, save=True, overwrite=True, noise_mag=2e-14, noise_grad=5e-13):

        picks = mne.pick_types(self.info, meg=True, exclude="bads")

        self.raw = mne.simulation.simulate_raw(self.info, self.source_simulator, forward=self.fwd)
        self.raw = self.raw.pick(picks=["meg", "stim"], exclude="bads")
        
        if self.sensor_array=='opm': 
            self.noise_std = dict({'mag':noise_mag}) #currently usign 40 Ft noise level for OPM (suggsted in some places, unsure if they refer to the same thing)
        elif self.sensor_array=='squids':
            if noise_mag is None:  
                self.noise_std=dict({'grad':None, 'mag':None}) #default in MNE, 5fT/sqr(Hz) for grads, 20fT/sqr(Hz) for mags
            else: 
                self.noise_std=dict({'grad':noise_grad, 'mag':noise_mag}) #default in MNE, 5fT/sqr(Hz) for grads, 20fT/sqr(Hz) for mags
        else: 
            raise ValueError(f'Sensor array {self.sensor_array} is not supported for simulations')

        noise_std_str = ''
        for key in self.noise_std.keys(): 
            if key=='mag':
                noise_std_str+='_'+str(key)+f'_{self.noise_std[key]}'
            else: 
                noise_std_str+=str(key)+f'_{self.noise_std[key]}'
        self.noise_std_str = noise_std_str

        if add_iir: #FIXME only works for SQUID sims right now, as it computes iir filter based on mne sample data 
            iir_filter = mne.time_frequency.fit_iir_model_raw(self.sample_raw, order=5, picks=picks, tmin=60, tmax=180)[1]
        else: 
            iir_filter = None 

        if noise_mag is not None: 
            self.noise_cov = mne.make_ad_hoc_cov(self.raw.info, std=self.noise_std)
            mne.simulation.add_noise(
                self.raw, cov=self.noise_cov, iir_filter=None, random_state=self.random_state
            )
        else:
            self.noise_cov = None

        if add_eog: 
            #mne.simulation.add_eog(self.raw, random_state=self.random_state)
            self.add_eog_no_sphere(head_pos=None, interp="cos2",n_jobs=None, verbose=None)
        if add_ecg: 
            #mne.simulation.add_ecg(self.raw, random_state=self.random_state)
            self.add_ecg_no_sphere(head_pos=None, interp="cos2",n_jobs=None, verbose=None)
        
        if save: 
            self.raw_fname = f"{self.stc_fname}-raw.fif"
            self.raw.save(os.path.join(self.output_path, self.raw_fname), overwrite=overwrite)


    def compute_evoked(self, save=True, overwrite=True):
        if self.source_labels_vol is not None: 
            tmax = (len(self.source_time_series_vol) - 1) * self.tstep
        elif self.source_labels_surf is not None: 
            tmax = (len(self.source_time_series_surf) - 1) * self.tstep
        tmin = -0.2
        self.epochs = mne.Epochs(self.raw, self.events, 1, tmin=tmin, tmax=tmax, baseline=(None, 0))
        self.evoked = self.epochs.average()

        if save: 
            self.epochs_fname = f"{self.stc_fname}-epo.fif"
            self.evoked_fname = f"{self.stc_fname}-ave.fif"
            self.epochs.save(os.path.join(self.output_path, self.epochs_fname), overwrite=overwrite)
            self.evoked.save(os.path.join(self.output_path, self.evoked_fname), overwrite=overwrite)

    
    def plot_stc(self, surface='inflated', hemi='lh'):

        stc = self.source_simulator.get_stc()
        stc_cropped = self.source_simulator.get_stc(
            start_sample=0, stop_sample=len(self.source_time_series)
        )

        """ #THIS PLOT ONLY works for surface srcs (as it requires triangular mesh (tris) in src, which vol or discrete srcs don't have)
        mne.viz.plot_sparse_source_estimates(
            fwd["src"], stc_cropped, bgcolor=(1, 1, 1), opacity=0.5, high_resolution=True
        )

        #DOES NOT WORK because stc is not of type VolSourceEstimate 
        mne.viz.plot_volume_source_estimates(
            stc_cropped, fwd['src'], subject='fsaverage', subjects_dir=subjects_dir, mode='glass_brain'
        ) """

        mne.viz.plot_source_estimates(
            stc_cropped, subject=self.subject, subjects_dir=self.subjects_dir, surface=surface, hemi=hemi, src=self.fwd['src'],
            alpha=0.2, 
        )       


    def plot_fwd_with_sources(self, surface='white'):
        
        fwd = mne.read_forward_solution(os.path.join(self.output_path, self.fwd_fname))
            
        fig = mne.viz.create_3d_figure(size=(600, 400))
        # Plot the cortex
        mne.viz.plot_alignment(
            subject=self.subject,
            subjects_dir=self.subjects_dir,
            trans=self.fname_trans,
            surfaces=surface,
            coord_frame="mri",
            fig=fig,
        )
        # Show the three dipoles defined at each location in the source space
        mne.viz.plot_alignment(
            subject=self.subject,
            subjects_dir=self.subjects_dir,
            trans=self.fname_trans,
            fwd=fwd,
            surfaces=surface,
            coord_frame="mri",
            fig=fig,
        )
        mne.viz.set_3d_view(figure=fig, azimuth=180, distance=1, focalpoint="auto")
    
    def plot_time_series(self, save=True, show=True):
        if self.source_labels_vol is not None: 
            plt.plot(np.arange(0,len(self.source_time_series_vol)*self.tstep, self.tstep), self.source_time_series_vol, color='darkred', label="Thalamus")
        if self.source_labels_surf is not None: 
            plt.plot(np.arange(0,len(self.source_time_series_surf)*self.tstep, self.tstep), self.source_time_series_surf, color='darkgreen', label="Occipital")
        plt.xlabel("Time (S)")
        plt.ylabel("Amplitude (nAm)")
        plt.legend()
        if save: 
            plt.savefig(os.path.join(self.figure_path, f"time_series_sim_{self.amp_nA}_nA.png"))
        if show: 
            plt.show()
        plt.close()
    
    def plot_raw(self, save=True, show=True):
        self.raw.plot(show=show)
        if save: 
            plt.savefig(os.path.join(self.figure_path, f'raw_{self.stc_fname}.png'))
        plt.close()
    
    def plot_added_noise(self, save=True, show=True):
        """
        Plotting ECG evokeds, ICA components for EOG and ECG artefacts and noise cov matrix 
        """
        import seaborn as sns

        #Noise cov 
        if self.noise_cov is not None: 
            noise_cov_mat = np.diag(self.noise_cov.data)
            plt.figure()
            sns.heatmap(noise_cov_mat)
            plt.suptitle(f"Sensor array: {self.sensor_array}, noise diag std: {self.noise_std}")
            if save: 
                plt.savefig(os.path.join(self.figure_path, f"added_noise_cov_{self.stc_fname}.png"))
            if show: 
                plt.show()
            plt.close()

        #Isolate ECG artefacts and plot topo 
        ecg_evoked = mne.preprocessing.create_ecg_epochs(self.raw).average()
        ecg_evoked.apply_baseline(baseline=(None, -0.2))

        fig = ecg_evoked.plot_joint(picks='mag', show=show)
        if save: 
            fig.savefig(os.path.join(self.figure_path, 'added_ecg_evoked_mag.png'))
            plt.close()
        if self.sensor_array=='squids': # grads only present in squid sims 
            fig = ecg_evoked.plot_joint(picks='grad', show=show)
            if save: 
                fig.savefig(os.path.join(self.figure_path, 'added_ecg_evoked_grad.png'))
                plt.close()

        #Run and plot ICA 
        ica = mne.preprocessing.ICA(n_components=15, random_state=97)
        ica.fit(self.raw, reject=dict(eeg=200e-6))

        fig = ica.plot_sources(self.raw, show_scrollbars=True, show=show) #looks like ICA000 is EOG, ICA001 is ECG
        if save: 
            fig.savefig(os.path.join(self.figure_path, 'added_ecg_eog_ica_sources.png'))
            plt.close()
        fig = ica.plot_components(show=show)
        if save: 
            fig.savefig(os.path.join(self.figure_path, 'added_ecg_eog_ica_components.png'))
            plt.close()
        fig = ica.plot_properties(self.raw, picks=0, show=show)
        if save: 
            fig[0].savefig(os.path.join(self.figure_path, 'added_ecg_eog_ica_properties_00.png'))
            plt.close()
        fig = ica.plot_properties(self.raw, picks=1, show=show)
        if save: 
            fig[0].savefig(os.path.join(self.figure_path, 'added_ecg_eog_ica_properties_01.png'))
            plt.close()

    
    def plot_raw_psd(self, save=True, show=True):
        spectrum = self.raw.compute_psd()
        fig = spectrum.plot(show=show)
        if save: 
            fig.savefig(os.path.join(self.figure_path, f"raw_psd_{self.stc_fname}.png"))
            plt.close()

        #plot noise level 
        fig = spectrum.plot(dB=False, average=True, amplitude=True,show=show)
        if save: 
            fig.savefig(os.path.join(self.figure_path, f"raw_noise_level_{self.stc_fname}.png"))
            plt.close()
    
    def plot_evoked_psd(self, save=True, show=True):
        plt.figure()
        plt.psd(self.evoked.data[0])
        if save: 
            plt.savefig(os.path.join(self.figure_path, f"evoked_psd_{self.stc_fname}"))
        if show: 
            plt.show()
        plt.close()

    def plot_evoked(self, save=True):
        self.evoked.plot(time_unit="s")
        plt.close()
    
    def plot_joint(self, picks, save=True, show=True):
        fig = self.evoked.plot_joint(picks=picks, show=show, times=[0.075, 0.18])
        if save: 
            fig.savefig(os.path.join(self.figure_path, f"evoked_joint-{self.stc_fname}{picks}.png"))
            plt.close()
    



""" 
class MixSimulator():

    def __init__(self):
        os.chdir('/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/scripts')
        pass

    def set_params(self, output_path):
        self.random_state = 42

        #Paths 
        self.raw_fname =  '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif'
        self.subject = 'fsaverage'
        self.subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
        self.fname_trans = 'fsaverage' #use built-in trans file for fsaverage 
        self.fname_bem = os.path.join(self.subjects_dir, self.subject, 'bem','fsaverage-5120-5120-5120-bem-sol.fif')
        self.fname_aseg = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/fsaverage/mri/aparc+aseg.mgz'
        self.fname_cov = os.path.join('/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis-cov.fif')
        self.output_path = output_path
        self.figure_path = os.path.join(self.output_path, 'figures')

        if not os.path.exists(self.output_path): 
            os.mkdir(self.output_path)
        if not os.path.exists(self.figure_path):
            os.mkdir(self.figure_path)

        #SRC spacing 
        self.surf_spacing = 'oct6'
        self.vol_spacing = 5.0 


    def create_info_obj(self):
        self.sample_raw = mne.io.read_raw_fif(self.raw_fname)
        self.info = self.sample_raw.pick(picks=['meg']).info
    
    def data_fun(self, amplitude, latency=0):
            """Generate random source time courses.
            #FIXME latency now is relative to the peak of the original wave (which is at 0.066 ms - used for cortical sims)
            #latency should be inputted in s 
            """
            rng = np.random.RandomState(42)
            self.amp_nA = amplitude 
            amp = amplitude*1e-9
            timeseries = (amp #50e-9 = 50 nAm
                * np.sin(20.0 * self.times)
                * np.exp(-((self.times - 0.15 + 0.05 * rng.randn(1)) ** 2) / 0.01)
            )

            if not latency==0: 
                shift_samples = int(latency/self.tstep)
                timeseries = timeseries[shift_samples:] 
                timeseries = np.concatenate((timeseries, (np.repeat(timeseries[-1], shift_samples))))
                timeseries[0] = 0.0
                diff = timeseries[5]-timeseries[0]
                inc = diff/5
                timeseries[1] = timeseries[0]+inc
                timeseries[2] = timeseries[1]+inc
                timeseries[3] = timeseries[2]+inc
                timeseries[4] = timeseries[3]+inc
            return(timeseries)


    def create_time_series(self, amplitude_surf, amplitude_vol, latency_vol):

        self.tstep = 1.0 / self.info['sfreq']
        self.times = np.arange(100, dtype=np.float64)*self.tstep

        n_events = 100
        events = np.zeros((n_events, 3), int)
        events[:, 0] = 200 * np.arange(n_events)  # Events sample.
        events[:, 2] = 1  # All events have the sample id.

        self.events = events

        if not amplitude_surf==0: 
            self.source_time_series_surf = self.data_fun(amplitude=amplitude_surf)
        if not amplitude_vol==0: 
            self.source_time_series_vol = self.data_fun(amplitude=amplitude_vol, latency=0.02) #will peak 0.02 sec (20 ms) before surface peak 


    def generate_src(self, vol_labels, save=True, plot=True, overwrite=True):
        """Generate mixed discrete src with dipole positions inputted as coords in unit 'm' 
        - Will use all subcortical labels provided in vol_labels
        - The source_simulator functions add a simulated STC for both hemispheres no matter what 
        (one will just be empty if only label from one hemi is provided). This creates an error if the src only 
        has one input (only one hemi). So even if just given a label for one hemi, we are always creating an src 
        for that label in both hemispheres. 
        - FIXME! Current NOT setup to run volume+surface labels at the same time - must be added 
        """

        from helper_functions import get_vol_label_vertices

        #Check that vol_labels is in list 
        if isinstance(vol_labels, str): 
            vol_labels = [vol_labels]

        #Extract vertices from aseg file for each label and create pos dict 
        self.label_vertices = dict()

        vol_labels_lh = [l for l in vol_labels if any(m in l.lower() for m in ['lh','left'])]
        vol_labels_rh = [l for l in vol_labels if any(m in l.lower() for m in ['rh','right'])]

        #If one hemi is not in labels, use the same label as for the other hemi 
        if len(vol_labels_lh)==0: 
            vol_labels_lh = []
            for lab in vol_labels_rh: 
                if "rh" in lab: 
                    vol_labels_lh.append(str(lab.replace("rh", "lh")))
                elif "Right" in lab: 
                    vol_labels_lh.append(str(lab.replace("Right", "Left")))
                else: 
                    raise ValueError("Label names in vol_labels do not have the correct hemi annotation. Must be one of (lh, rh, Left, Right)")

        if len(vol_labels_rh)==0: 
            vol_labels_rh = []
            for lab in vol_labels_lh: 
                if "lh" in lab: 
                    vol_labels_rh.append(str(lab.replace("lh", "rh")))
                elif "Left" in lab: 
                    vol_labels_rh.append(str(lab.replace("Left", "Right")))
                else: 
                    raise ValueError("Label names in vol_labels do not have the correct hemi annotation. Must be one of (lh, rh, Left, Right)")

        # SETUP SRC FOR LH 
        #Load volume vertices from aseg file and transform positions to MRI (RAS) coords in units m 
        #Returns list of vertex positions for each label in vol_labels_hemi
        verts_list_lh = get_vol_label_vertices(self.fname_aseg, vol_labels_lh, units='m') 
        for i, name in enumerate(vol_labels_lh): 
            self.label_vertices[name] = verts_list_lh[i]

        #Create pos dict with rr and nn positions of vertices for the labels to use 
        pos_lh = dict()
        #pos_lh_vol_sub = dict()  

        #Add volume vertices 
        for i in range(0, len(vol_labels_lh)):
            if i==0: 
                rr_concat = verts_list_lh[i]
            else: 
                rr_concat = np.concatenate((rr_concat, verts_list_lh[i]))

        
        nn_lh = rr_concat.copy() #copy shape 
        nn_lh[:,:] = [0.,0.,1.] #replace values with orientation values (using volume src default here)

        #Add to pos dict for only vol (to be used in grow labels)
        #pos_lh_vol_sub['rr'] = rr_concat 
        #pos_lh_vol_sub['nn'] = nn_lh

        # #Add surface vertices 
        # src_surf = mne.setup_source_space(self.subject, spacing=self.surf_spacing, add_dist="patch", subjects_dir=self.subjects_dir) 
        # rr_concat = np.concatenate((src_surf[0]['rr'][src_surf[0]['inuse'].astype(bool)], rr_concat))
        # nn_lh = np.concatenate((src_surf[0]['nn'][src_surf[0]['inuse'].astype(bool)], nn_lh))
        

        pos_lh['rr'] = rr_concat
        pos_lh['nn'] = nn_lh

        #Setup source space for those vertices 
        src_lh = mne.setup_volume_source_space(
            subject='fsaverage',
            mri = None, 
            pos = pos_lh, 
            bem=self.fname_bem,
            subjects_dir=self.subjects_dir,
            sphere_units="m",
        )

        # src_lh_sub = mne.setup_volume_source_space(
        #     subject='fsaverage',
        #     mri = None, 
        #     pos = pos_lh_vol_sub, 
        #     bem=self.fname_bem,
        #     subjects_dir=self.subjects_dir,
        #     sphere_units="m",
        # )

        # SETUP SRC FOR RH 
        #Load volume vertices from aseg file and transform positions to MRI (RAS) coords in units m 
        #Returns list of vertex positions for each label in vol_labels_hemi
        verts_list_rh = get_vol_label_vertices(self.fname_aseg, vol_labels_rh, units='m') 
        for i, name in enumerate(vol_labels_rh): 
            self.label_vertices[name] = verts_list_rh[i]

        #Create pos dict with rr and nn positions of vertices for the labels to use 
        pos_rh = dict()
        #pos_rh_vol_sub = dict()

        for i in range(0, len(vol_labels_rh)):
            if i==0: 
                rr_concat = verts_list_rh[i]
            else: 
                rr_concat = np.concatenate((rr_concat, verts_list_rh[i]))
        
        nn_rh = rr_concat.copy() #copy shape 
        nn_rh[:,:] = [0.,0.,1.] #replace values with orientation values (using volume src default here)
        
        #pos_rh_vol_sub['rr'] = rr_concat 
        #pos_rh_vol_sub['nn'] = nn_rh

        #Add surface vertices 
        # rr_concat = np.concatenate((src_surf[1]['rr'][src_surf[0]['inuse'].astype(bool)], rr_concat))
        # nn_rh = np.concatenate((src_surf[1]['nn'][src_surf[0]['inuse'].astype(bool)], nn_rh))

        pos_rh['rr'] = rr_concat
        pos_rh['nn'] = nn_rh

        #Setup source space for those vertices 
        src_rh = mne.setup_volume_source_space(
            subject='fsaverage',
            mri = None, 
            pos = pos_rh, 
            bem=self.fname_bem,
            subjects_dir=self.subjects_dir,
            sphere_units="m",
        )

        # src_rh_sub = mne.setup_volume_source_space(
        #     subject='fsaverage',
        #     mri = None, 
        #     pos = pos_rh_vol_sub, 
        #     bem=self.fname_bem,
        #     subjects_dir=self.subjects_dir,
        #     sphere_units="m",
        # )

        src_vol = src_lh + src_rh
        self.src = src_vol 
        self.labels_str = str("_".join(l for l in vol_labels_lh))

        #self.src_vol_sub = src_lh_sub + src_rh_sub

        #Save 
        self.fname_src = f'{self.labels_str}-xx_mm-fsaverage-src.fif'
        if save: 
            mne.write_source_spaces(os.path.join(self.output_path, self.fname_src), src_vol, overwrite=overwrite)

        #Plot 
        if plot: 
            fig_name = f'src-{self.labels_str}-xx_mm.png'
            fig = mne.viz.plot_bem(src=src_vol, subject=self.subject, subjects_dir=self.subjects_dir, show=False)
            fig.savefig(os.path.join(self.figure_path, fig_name))
            plt.close()
            

    def generate_fwd(self, save=True, overwrite=True):
            self.fwd_fname = f'{self.labels_str}-xx_mm-fsaverage-fwd.fif'
            src = mne.read_source_spaces(os.path.join(self.output_path,self.fname_src))
            self.fwd = mne.make_forward_solution(self.info, self.fname_trans, src, self.fname_bem, mindist=5.0)

            #self.fwd_vol_sub = mne.make_forward_solution(self.info, self.fname_trans, self.src_vol_sub, self.fname_bem, mindist=5.0)
            
            if save: 
                mne.write_forward_solution(os.path.join(self.output_path, self.fwd_fname), self.fwd, overwrite=overwrite)

    def grow_sim_source_label_vol(self, labels, seeds=None, extents=2.0):
        """ Find center of mass of labels, and grow a new label includign all vertices wihtin distance (extent).
        Currently based on seed defined as the center of mass of the full region if seed is None. Com can be computed 
        using median or mean of the vertex coords. 

        label: the label from which to grow the label (and from where the seed origins, if provided)
        seed: seed to use as starting point, if None then the center of mass is computed and used as seed 
        extent: (int | "full") size of resulting label in mm, if full, returning a label including all vertices in the region (from aseg file)

        Using a subset src with only the volume vertices - otherwise the grow labels function grows region 
        into surface vertices as well (if the src also contains those).
        FIXME: currently only works if only ONE volume label is simulated at a time, fix to allow more 
        """
        from helper_functions import _center_of_mass, grow_labels

        if isinstance(extents, list):
            self.extents_vol=extents[0]
            self.extents_surf=extents[1]
        else: 
            self.extents_vol=extents
            self.extents_surf = 0

        if self.extents_vol==0: 
            self.source_labels_vol = None 
        
        else: 
            if isinstance(labels, str):
                labels = [labels]

            #Check that labels are actually in src 
            labs_in_src = [name for name in self.label_vertices.keys()]
            for l in labels: 
                if not l in labs_in_src: 
                    raise KeyError(f"Source label {l} is not present in src")

            #Check which hemis are present in labels (and for which to grow labels)
            hemis_in_labels = []
            if any(l for l in labels if any(m in l.lower() for m in ['lh','left'])):
                hemis_in_labels.append('lh')
            if any(l for l in labels if any(m in l.lower() for m in ['rh','right'])):
                hemis_in_labels.append('rh')
            print(f'- Hemis in labels: {hemis_in_labels}')


            #This function returns the index of the center seed 
            # - but the index is in range(0, n_vertices in label)
            # - thus, our mixed src has more vertex numbers and the index of this seed does not correspond to the same index in our src (because the src also has surface vertices, and they occur first)
            # - therefore we must find the matching seed index in our src 
            if not seeds: 
                #Compute center of mass 
                self.seeds = dict()
                seeds = []
                for idx, l in enumerate(labels): 
                    seed = _center_of_mass(self.label_vertices[l])
                    seeds.append(seed)
                    self.seeds[l] = seed

                    
            self.source_labels_vol = grow_labels(
                #self.fwd['src'], 
                self.fwd['src'],
                #self.src,
                self.subject,
                seeds=seeds, #one seed per label per hemis
                extents=extents, #either one value per label, otherwise it will use same extent for each label 
                hemis=hemis_in_labels,
                subjects_dir=self.subjects_dir,
                n_jobs=None,
                names=labels,
                colors=None,
            )

            ##Update vertex numbers to match those in the full mixed src 
            #surf_verts_len = len(self.src[idx]['vertno'])-len(self.label_vertices[l])
            #vol_label_updated = source_labels_vol.copy()
            #vol_label_updated[0].vertices = source_labels_vol[0].vertices + surf_verts_len
            #self.source_labels_vol = vol_label_updated 

    
    def grow_sim_source_label_surf(self, label_regex='lateraloccipital-lh', location='center', extent=5.0):
        
        self.extents_surf=extent

        if self.extents_surf==0: 
            self.source_labels_surf = None
        
        else: 
            if isinstance(label_regex, str):
                label_regex = [label_regex]
            
            self.selected_labels = []
            self.source_labels_surf = []
            
            for l in label_regex: 
                self.selected_labels.append(mne.read_labels_from_annot(
                    self.subject, regexp=l, subjects_dir=self.subjects_dir
                )[0])

            for source in self.selected_labels: 
                self.source_labels_surf.append(mne.label.select_sources(
                    self.subject, 
                    source,
                    location=location, #uses center of mass (CoM) of provided label as seed 
                    extent=extent, 
                    grow_outside=False,
                    subjects_dir=self.subjects_dir,
                    name = source.name,
                    random_state=self.random_state
                ))


    def initiate_sourcesimulator(self):        
        self.source_simulator = mne.simulation.SourceSimulator(self.fwd['src'], tstep=self.tstep)


    def add_to_sourcesimulator(self, labels="all", save=True, overwrite=True):
        """
        FIXME: edit so that we can specify which source time series to sim for which label 
        Currently just adding the same generic one to each label 
        """

        if labels=="all":
            if self.source_labels_vol is not None: 
                for vol_source in self.source_labels_vol: 
                    print(f"-- Adding volume time series for volume {vol_source}")
                    self.source_simulator.add_data(vol_source, self.source_time_series_vol, self.events)
            if self.source_labels_surf is not None: 
                for surf_source in self.source_labels_surf: 
                    print(f"--Adding surface time series for cortical region {surf_source}")
                    self.source_simulator.add_data(surf_source, self.source_time_series_surf, self.events)
        elif labels=="surf":
            for surf_source in self.source_labels_surf: 
                self.source_simulator.add_data(surf_source, self.source_time_series_surf, self.events)
        elif labels=="vol":
            for vol_source in self.source_labels_vol: 
                self.source_simulator.add_data(vol_source, self.source_time_series_vol, self.events)
        else: #only add those provided in input parameter labels 
            self.source_labels_all = [self.source_labels_vol, self.source_labels_surf]
            for label_source in self.source_labels_all: 
                if label_source.name in labels: 
                    if label_source.name in self.source_labels_vol: 
                        self.source_simulator.add_data(label_source, self.source_time_series_vol, self.events)
                    elif label_source.name in self.source_labels_surf: 
                        self.source_simulator.add_data(label_source, self.source_time_series_surf, self.events)

        if save: #FIXME currently naming only works when extents is the same for all vols or all surfs 
            str1 = "-".join([self.source_labels_vol[i].name + f'_{self.extents_vol}_mm' for i, x in enumerate(self.source_labels_vol)])
            if not self.source_labels_surf is None: 
                str2 = "-".join([self.source_labels_surf[i].name + f'_{self.extents_surf}_mm' for i, x in enumerate(self.source_labels_surf)])
            else: 
                str2 = ""
            self.stc_fname = f"mix_{self.amp_nA}_nA_{str1}-{str2}"
            self.source_simulator.get_stc().save(os.path.join(self.output_path, self.stc_fname), overwrite=overwrite)


    def sim_raw(self, add_iir=True, add_eog=True, add_ecg=True, save=True, overwrite=True):

        picks = mne.pick_types(self.info, meg=True, exclude="bads")

        self.raw = mne.simulation.simulate_raw(self.info, self.source_simulator, forward=self.fwd)
        self.raw = self.raw.pick(picks=["meg", "stim"], exclude="bads")

        noise_cov = mne.read_cov(self.fname_cov)

        if add_iir: 
            iir_filter = mne.time_frequency.fit_iir_model_raw(self.sample_raw, order=5, picks=picks, tmin=60, tmax=180)[1]
            mne.simulation.add_noise(
                self.raw, cov=noise_cov, iir_filter=iir_filter, random_state=self.random_state
            )
        else: 
            mne.simulation.add_noise(
                self.raw, cov=noise_cov, iir_filter=None, random_state=self.random_state
            )
        if add_eog: 
            mne.simulation.add_eog(self.raw, random_state=self.random_state)
        if add_ecg: 
            mne.simulation.add_ecg(self.raw, random_state=self.random_state)
        
        if save: 
            self.raw_fname = f"{self.stc_fname}-raw.fif"
            self.raw.save(os.path.join(self.output_path, self.raw_fname), overwrite=overwrite)



    def compute_evoked(self, save=True, overwrite=True):
        tmax = (len(self.source_time_series_vol) - 1) * self.tstep
        tmin = -0.2
        self.epochs = mne.Epochs(self.raw, self.events, 1, tmin=tmin, tmax=tmax, baseline=(None, 0))
        self.evoked = self.epochs.average()

        if save: 
            self.epochs_fname = f"{self.stc_fname}-epo.fif"
            self.evoked_fname = f"{self.stc_fname}-ave.fif"
            self.epochs.save(os.path.join(self.output_path, self.epochs_fname), overwrite=overwrite)
            self.evoked.save(os.path.join(self.output_path, self.evoked_fname), overwrite=overwrite)

    
    def plot_stc(self, surface='inflated', hemi='lh'):

        stc = self.source_simulator.get_stc()
        stc_cropped = self.source_simulator.get_stc(
            start_sample=0, stop_sample=len(self.source_time_series_vol)
        )

        """ #THIS PLOT ONLY works for surface srcs (as it requires triangular mesh (tris) in src, which vol or discrete srcs don't have)
        mne.viz.plot_sparse_source_estimates(
            fwd["src"], stc_cropped, bgcolor=(1, 1, 1), opacity=0.5, high_resolution=True
        )

        #DOES NOT WORK because stc is not of type VolSourceEstimate 
        mne.viz.plot_volume_source_estimates(
            stc_cropped, fwd['src'], subject='fsaverage', subjects_dir=subjects_dir, mode='glass_brain'
        ) """

        mne.viz.plot_source_estimates(
            stc_cropped, subject=self.subject, subjects_dir=self.subjects_dir, surface=surface, hemi=hemi, src=self.fwd['src'],
            alpha=0.2, 
        )       


    def plot_fwd_with_sources(self, surface='white'):
        
        fwd = mne.read_forward_solution(os.path.join(self.output_path, self.fwd_fname))
            
        fig = mne.viz.create_3d_figure(size=(600, 400))
        # Plot the cortex
        mne.viz.plot_alignment(
            subject=self.subject,
            subjects_dir=self.subjects_dir,
            trans=self.fname_trans,
            surfaces=surface,
            coord_frame="mri",
            fig=fig,
        )
        # Show the three dipoles defined at each location in the source space
        mne.viz.plot_alignment(
            subject=self.subject,
            subjects_dir=self.subjects_dir,
            trans=self.fname_trans,
            fwd=fwd,
            surfaces=surface,
            coord_frame="mri",
            fig=fig,
        )
        mne.viz.set_3d_view(figure=fig, azimuth=180, distance=1, focalpoint="auto")
    
    def plot_time_series(self, vol=True, surf=True, save=True, show=True):
        if vol: 
            #plt.plot(np.arange(0,len(self.source_time_series_vol)*self.tstep, self.tstep), self.source_time_series_vol, color='darkred', label=self.source_labels_vol[0].name)
            plt.plot(np.arange(0,len(self.source_time_series_vol)*self.tstep, self.tstep), self.source_time_series_vol, color='darkred', label="Thalamus")
        if surf: 
            #plt.plot(np.arange(0,len(self.source_time_series_surf)*self.tstep, self.tstep), self.source_time_series_surf, color='darkgreen', label=self.source_labels_surf[0].name)
            plt.plot(np.arange(0,len(self.source_time_series_surf)*self.tstep, self.tstep), self.source_time_series_surf, color='darkgreen', label="Occipital")
        plt.xlabel("Time (S)")
        plt.ylabel("Amplitude (nAm)")
        plt.legend()
        if save: 
            plt.savefig(os.path.join(self.figure_path, f"time_series_sim_{self.amp_nA}_nA.png"))
        if show: 
            plt.show()
        plt.close()
    
    def plot_raw(self, save=True, show=True):
        self.raw.plot(show=show)
        if save: 
            raw_fig_fname = f"raw-{self.stc_fname}.png"
            plt.savefig(os.path.join(self.figure_path, raw_fig_fname))
        if show: 
            plt.show()
        plt.close()
    
    def plot_raw_psd(self, save=True, show=True):
        self.raw.compute_psd().plot(show=show)
        if save: 
            raw_psd_fig_fname = f"raw_psd-{self.stc_fname}.png"
            plt.savefig(os.path.join(self.figure_path, raw_psd_fig_fname))
            plt.close()
    
    def plot_evoked_psd(self, save=False, show=True):
        plt.figure()
        plt.psd(self.evoked.data[0])
        if save: 
            evoked_psd_fig_fname = f"evoked_psd-{self.stc_fname}.png"
            plt.savefig(os.path.join(self.figure_path, evoked_psd_fig_fname))
        if show: 
            plt.show()
        plt.close()

    def plot_evoked(self, save=True):
        self.evoked.plot(time_unit="s")
    
    def plot_joint(self, picks, save=True, show=True):
        fig = self.evoked.plot_joint(picks=picks, show=show, times=[0.075, 0.095, 0.18, 0.2])
        if save: 
            joint_fig_fname = f"evoked_joint-{self.stc_fname}-{picks}.png"
            fig.savefig(os.path.join(self.figure_path, joint_fig_fname))
        if show: 
            plt.show()
        plt.close()
    
 """