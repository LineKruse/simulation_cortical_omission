import os 
import mne 
import numpy as np 
import pandas as pd 


### Plot cortial atlas from freesurfer 
# - Simulation script currently uses the HCP MMP 1.0 atlas (Human Connectome Project)

output_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/parcellations'
Brain = mne.viz.get_brain_class()

mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir)
labels = mne.read_labels_from_annot(
    "fsaverage", "HCPMMP1", "lh", subjects_dir=subjects_dir
)
brain = Brain(
    "fsaverage",
    "lh",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="high_contrast",
    background="white",
    alpha=0.5,
    size=(800, 600),
)
#brain.add_annotation('HCPMMP1', borders=True, color='white')
v1_label = [l for l in labels if 'V1' in l.name][0] #taking left V1
a1_label = [l for l in labels if 'A1' in l.name][0] #taking left A1 
s1_label = [l for l in labels if '3b' in l.name][0]
brain.add_label(v1_label, borders=False, color='red')
brain.add_label(a1_label, borders=False, color='darkgreen')
brain.add_label(s1_label, borders=False, color='darkblue')
brain.save_image(os.path.join(output_path, 'surface_regions_used_hcpmmp1_atlas_white.png'))


x = mne.vertex_to_mni(v1_label.vertices, hemis=0, subject='fsaverage', subjects_dir=subjects_dir)
v1_label.vertices
v1_label.pos

### Plot subcortical labels used 
fname_aseg = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/fsaverage/mri/aparc+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg(fname_aseg)
volume_labels = ["Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex", "Left-Thalamus-Proper","Right-Thalamus-Proper", "Left-Caudate", "Right-Caudate","Left-Hippocampus", "Right-Hippocampus"] 
volume_labels = ["Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Hippocampus"] 

brain = Brain(
    'fsaverage',
    hemi='lh',
    surf='white',
    alpha=0.5,
    background='white',
    cortex='low_contrast',
    units='m',
    subjects_dir=subjects_dir
)
brain.add_volume_labels(aseg="aparc+aseg", labels=volume_labels)
brain.show_view(azimuth=0) #rotate to better see 
brain.save_image(os.path.join(output_path, 'volume_regions_used_aparcaseg_white.png'))



############################## 
# Check how the add volume labels gets the positions from these labels 
import nibabel as nib
aseg = nib.load(fname_aseg)
aseg_data = np.array(aseg.dataobj) # (256, 256, 256) 
vox_mri_t = aseg.header.get_vox2ras_tkr() # (4, 4)
mult = 1e-3 # if self._units == "m" else 1, defined in the Brain() obj, default unit is "m"
vox_mri_t[:3] *= mult #(4, 4)

# read freesurfer lookup table
from mne._freesurfer import read_freesurfer_lut
from mne.utils.check import _to_rgb
from mne.surface import _marching_cubes

lut, fs_colors = read_freesurfer_lut()

#### Extract and save MNI coords of each region - SURFACE and VOLUME 
smooth=0.9 #default 
fill_hole_size = None #default 
labels=volume_labels
colors = [fs_colors[label] / 255 for label in labels]
colors = [_to_rgb(color, name=f"colors[{ci}]") for ci, color in enumerate(colors)]
surfs = _marching_cubes(
            aseg_data,
            [lut[label] for label in labels],
            smooth=smooth,
            fill_hole_size=fill_hole_size,
        )

from mne.transforms import apply_trans
#surfs has 4 elements, onen per label inputted 
#each surf element has a verts array and a triangels array 
#the vox_mri_t trans is then applied to the vertices and transposed (.T)

cer_lh_surf = surfs[0]
cer_lh_vertices = apply_trans(vox_mri_t, surfs[0][0]).T
cer_lh_triangles = cer_lh_surf[1]

for label, color, (verts, triangles) in zip(labels, colors, surfs):
    print(label)
    print(verts)
    print(triangles)
    if len(verts) == 0:  # not in aseg vals
        print(
            f"Value {lut[label]} not found for label "
            f"{repr(label)} in anatomical segmentation file "
        )
        continue
    verts = apply_trans(vox_mri_t, verts)
    for _ in Brain._iter_views("vol"):
        print(_)
        actor, _ = self._renderer.mesh(
            *verts.T,
            triangles=triangles,
            color=color,
            opacity=alpha,
            reset_camera=False,
            render=False,
        )
        self._add_actor("volume_labels", actor)

x = zip(labels, colors, surfs)