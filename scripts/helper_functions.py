import mne 
import numpy as np 
import nibabel as nib 
import os 
import pandas as pd 


def _marching_cubes(image, level, smooth=0, fill_hole_size=None, use_flying_edges=True):
    """Compute marching cubes on a 3D image.
    LK notes: copied directly from mne.surface, no changes made 
    For some reason this function gave segmentation error when imported from mne.surface
    but not when computing it line by line"""
    # vtkDiscreteMarchingCubes would be another option, but it merges
    # values at boundaries which is not what we want
    # https://kitware.github.io/vtk-examples/site/Cxx/Medical/GenerateModelsFromLabels/  # noqa: E501
    # Also vtkDiscreteFlyingEdges3D should be faster.
    # If we ever want not-discrete (continuous/float) marching cubes,
    # we should probably use vtkFlyingEdges3D rather than vtkMarchingCubes.
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes, vtkImageData
    from vtkmodules.vtkFiltersCore import vtkThreshold
    from vtkmodules.vtkFiltersGeneral import (
        vtkDiscreteFlyingEdges3D,
        vtkDiscreteMarchingCubes,
    )
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
    from mne.surface import _vtk_smooth

    if image.ndim != 3:
        raise ValueError(f"3D data must be supplied, got {image.shape}")

    level = np.array(level)
    if level.ndim != 1 or level.size == 0 or level.dtype.kind not in "ui":
        raise TypeError(
            "level must be non-empty numeric or 1D array-like of int, "
            f"got {level.ndim}D array-like of {level.dtype} with "
            f"{level.size} elements"
        )

    # vtkImageData indexes as slice, row, col (Z, Y, X):
    # https://discourse.vtk.org/t/very-confused-about-imdata-matrix-index-order/6608/2
    # We can accomplish this by raveling with order='F' later, so we might as
    # well make a copy with Fortran order now.
    # We also use double as passing integer types directly can be problematic!
    image = np.array(image, dtype=float, order="F")
    image_shape = image.shape

    # fill holes
    if fill_hole_size is not None:
        for val in level:
            bin_image = image == val
            mask = image == 0  # don't go into other areas
            bin_image = binary_dilation(bin_image, iterations=fill_hole_size, mask=mask)
            image[bin_image] = val

    data_vtk = numpy_to_vtk(image.ravel(order="F"), deep=False)

    mc = vtkDiscreteFlyingEdges3D() if use_flying_edges else vtkDiscreteMarchingCubes()
    # create image
    imdata = vtkImageData()
    imdata.SetDimensions(image_shape)
    imdata.SetSpacing([1, 1, 1])
    imdata.SetOrigin([0, 0, 0])
    imdata.GetPointData().SetScalars(data_vtk)

    # compute marching cubes on smoothed data
    mc.SetNumberOfContours(len(level))
    for li, lev in enumerate(level):
        mc.SetValue(li, lev)
    mc.SetInputData(imdata)
    mc.Update()
    mc = _vtk_smooth(mc.GetOutput(), smooth)

    # get verts and triangles
    selector = vtkThreshold()
    selector.SetInputData(mc)
    dsa = vtkDataSetAttributes()
    selector.SetInputArrayToProcess(
        0,
        0,
        0,
        imdata.FIELD_ASSOCIATION_POINTS
        if use_flying_edges
        else imdata.FIELD_ASSOCIATION_CELLS,
        dsa.SCALARS,
    )
    geometry = vtkGeometryFilter()
    geometry.SetInputConnection(selector.GetOutputPort())

    out = list()
    for val in level:
        try:
            selector.SetLowerThreshold
        except AttributeError:
            selector.ThresholdBetween(val, val)
        else:
            # default SetThresholdFunction is between, so:
            selector.SetLowerThreshold(val)
            selector.SetUpperThreshold(val)
        geometry.Update()
        polydata = geometry.GetOutput()
        rr = vtk_to_numpy(polydata.GetPoints().GetData())
        tris = vtk_to_numpy(polydata.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        rr = np.ascontiguousarray(rr)
        tris = np.ascontiguousarray(tris)
        out.append((rr, tris))
    return out



def get_vol_label_vertices(fname_aseg, volume_labels, units='m'):
    """
    Function to extract vertex positions in correct coords for volume labels. 

    This codes is chunks taken from the function viz.Brain.add_volume_label(), 
    just to figure out they extract the volume vertex positions and transforms them to plot on white or pial mesh 
    """
    from mne._freesurfer import read_freesurfer_lut
    #from mne.surface import _marching_cubes
    import nibabel as nib
    from mne.transforms import apply_trans

    if isinstance(volume_labels, str):
        volume_labels = [volume_labels]
    
    print(f"- Volume labels inputted to get_vol_label_vertices: {volume_labels}")

    aseg = nib.load(fname_aseg)
    aseg_data = np.array(aseg.dataobj) # (256, 256, 256) 
    vox_mri_t = aseg.header.get_vox2ras_tkr() # (4, 4)
    mult = 1e-3 if units=='m' else 1 # 
    vox_mri_t[:3] *= mult #(4, 4)

    lut, fs_colors = read_freesurfer_lut() #lookup table 

    smooth=0.9 #default 
    fill_hole_size = None #default 
    labels=volume_labels
    surfs = _marching_cubes(
                aseg_data,
                [lut[label] for label in labels],
                smooth=smooth,
                fill_hole_size=fill_hole_size,
            )
    labels_coords = []
    for i, lab in enumerate(labels):
        print(f"- Getting vertex coords for label {lab}")
        label_coords = apply_trans(vox_mri_t, surfs[i][0])
        labels_coords.append(label_coords) 

    #surf[0] has vertices, surf[1] has triangles 
    print(f"- verts_list has length {len(labels_coords)}")
    return [lab for lab in labels_coords]



def _verts_within_dist(src, seed, hemi='lh', extent=5.0, metric='euclidean'):
    """
    src: source space from which the seed is taken (this function assumes rr positions are in unit "m")
    seeds: vertex number in src to use as seed, int 
    hemi: hemispheres for which to grow label 
    extent: distance from seed within which to extract vertices, in mm 

    """
    from scipy.spatial.distance import pdist, squareform

    if not hemi in ['lh', 'rh']:
        raise KeyError(
            "hemi must be one of 'lh'|'rh' "
        )

    hemi_idx = 0 if hemi=='lh' else 1 
    max_dist = extent*0.001 # convert max dist to unit "m"

    #Extract vertex coordinates (in MRI RAS coords, mm)
    volume_coords = src[hemi_idx]['rr']
    volume_verts = src[hemi_idx]['vertno']

    #Compute distance of every pair of vertices
    dist_matrix = pdist(volume_coords, metric="euclidean") #max=0.07, min 0.0 (distance in meters)

    #Convert to square matrix (n_vertices, n_vertices)
    dist_matrix = squareform(dist_matrix) #array, shape (100, 100)

    #Make own version of _verts_within_dist() that works on this matrix 
    source = seed #just usign vertex no 50 as seed 
    dist_map = {}
    verts_added_last = []
    #for source in sources: 
    dist_map[source] = 0
    verts_added_last.append(source)


    #Add neighbros until no more neighbors within max_dist can be found 
    #OBS: this runs, but the distances are too small (all 100 are added), check the coords and metrics 
    while len(verts_added_last) > 0: 
        verts_added = []
        for i in verts_added_last: 
            v_dist = dist_map[i] 
            for vert in volume_verts: 
                n_dist = v_dist + dist_matrix[vert, i] #take distance bewteen last added vert and iterating verts
                #Check if this vert has already been added
                if vert in dist_map: 
                    if n_dist < dist_map[vert]:
                        dist_map[vert] = n_dist
                else: 
                    #else, check if it is wthin the max distance (extent parameter)
                    if n_dist <= max_dist: 
                        dist_map[vert] = n_dist
                        #we found anew vertex within max_dist 
                        verts_added.append(vert)
        verts_added_last = verts_added

    verts = np.sort(np.array(list(dist_map.keys()), int)) #vertex numbers to include in label 
    dist = np.array([dist_map[v] for v in verts]) #distance from seed in m, (inputted as Label.values)

    return verts, dist 


def grow_labels(
    src, 
    subject,
    seeds,
    extents,
    hemis,
    subjects_dir=None,
    n_jobs=None,
    names=None,
    colors=None,
):
    """Generate circular labels in source space with region growing.

    This function generates a number of labels in source space by growing
    regions starting from the vertices defined in "seeds". For each seed, a
    label is generated containing all vertices within a maximum geodesic
    distance on the white matter surface from the seed.

    Parameters
    ----------
    %(subject)s
    src   : source space from which the seeds are taken (function assumes rr positions are in meters)
    seeds : int | list
        Seed, or list of seeds. Each seed can be either a vertex number or
        a list of vertex numbers.
    extents : array | float
        Extents (radius in mm) of the labels.
    hemis : array | str
        Hemispheres to use for the labels ("lh" or "rh")
    %(subjects_dir)s
    %(n_jobs)s
        Likely only useful if tens or hundreds of labels are being expanded
        simultaneously. Does not apply with ``overlap=False``.
    names : None | list of str
        Assign names to the new labels (list needs to have the same length as
        seeds).
    colors : array, shape (n, 4) or (, 4) | None
        How to assign colors to each label. If None then unique colors will be
        chosen automatically (default), otherwise colors will be broadcast
        from the array. The first three values will be interpreted as RGB
        colors and the fourth column as the alpha value (commonly 1).

    Returns
    -------
    labels : list of Label
        The labels' ``comment`` attribute contains information on the seed
        vertex and extent; the ``values``  attribute contains distance from the
        seed in millimeters.

    Notes
    -----
    "extents" and "hemis" can either be arrays with the same length as
    seeds, which allows using a different extent and hemisphere for
    label, or integers, in which case the same extent and hemisphere is
    used for each label.
    """
    #Make sure subject dir is there 
    if subjects_dir==None: 
        raise KeyError(
            "subjects_dir parameter is required"
        )

    if isinstance(seeds, int):
        seeds = [seeds]
    if isinstance(extents, float):
        extents = [extents]
    if isinstance(extents, str):
        extents = [extents]
    n_seeds = len(seeds)

    if len(extents) != 1 and len(extents) != n_seeds:
        raise ValueError("The extents parameter has to be of length 1 or len(seeds)")

    if len(hemis) != 1 and len(hemis) != n_seeds:
        raise ValueError("The hemis parameter has to be of length 1 or len(seeds)")

    if colors is not None:
        if len(colors.shape) == 1:  # if one color for all seeds
            n_colors = 1
            n = colors.shape[0]
        else:
            n_colors, n = colors.shape

        if n_colors != n_seeds and n_colors != 1:
            msg = (
                f"Number of colors ({n_colors}) and seeds ({n_seeds}) are not "
                "compatible."
            )
            raise ValueError(msg)
        if n != 4:
            msg = f"Colors must have 4 values (RGB and alpha), not {n}."
            raise ValueError(msg)

    # make the arrays the same length as seeds
    if len(extents) == 1:
        extents = np.tile(extents, n_seeds)

    if len(hemis) == 1:
        hemis = np.tile(hemis, n_seeds)

    # names
    if names is None:
        names = [f"Label_{ii}-{h}" for ii, h in enumerate(hemis)]
    else:
        if np.isscalar(names):
            names = [names]
        if len(names) != n_seeds:
            raise ValueError(
                "The names parameter has to be None or have length len(seeds)"
            )
        for i, hemi in enumerate(hemis):
            if not names[i].endswith(hemi):
                names[i] = "-".join((names[i], hemi))
    names = np.array(names)

    # create the patches
    labels = _grow_labels(src, seeds, extents, hemis, names, subject)

    if colors is None:
        # add a unique color to each label
        label_colors = mne.label._n_colors(len(labels))
    else:
        # use specified colors
        label_colors = np.empty((len(labels), 4))
        label_colors[:] = colors

    for label, color in zip(labels, label_colors):
        label.color = color

    return labels

def _grow_labels(src, seeds, extents, hemis, names, subject):
    """Parallelize grow_labels."""
    labels = []
    for seed, extent, hemi, name in zip(seeds, extents, hemis, names):
        print(seed, extent, hemi, name)

        label_verts, label_dist = _verts_within_dist(src, seed, hemi=hemi, extent=extent)

        # create a label
        seed_repr = str(seed)
        hemi_idx = 0 if hemi=='lh' else 1

        if isinstance(extent, str):
            comment = f"Circular label: seed={seed_repr}, extent=one dipole (CoM)"
        else: 
            comment = f"Circular label: seed={seed_repr}, extent={extent:0.1f}mm"
        label = mne.Label(
            vertices=label_verts,
            pos=src[hemi_idx]['rr'][label_verts],
            values=label_dist,
            hemi=hemi,
            comment=comment,
            name=str(name),
            subject=subject,
        )
        labels.append(label)
    return labels


### CURRENTLY NOT WORKING CORRECTLY - finding com only on surface of cerebellum, not mass 
# - (same thing happening for the grow labels - figure out why)
def _center_of_mass(verts, values=None, method='median'):
    import numpy as np

    """
    verts: vertex positions in meters (from src from which you want to create a Label)
    values: can e.g., be distance from seed, will be used as weighting param in calculation of center of mass 
    """
    if values is None: 
        values = np.repeat(1, len(verts))

    #Find coords of center of mass (using median to reduce influence of outlier points)
    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]

    if method=='median':
        com = np.median(x), np.median(y), np.median(z) 
    elif method=='mean':
        com = np.mean(x), np.mean(y), np.mean(z) 
    else: 
        raise ValueError("method must be one of ('median', 'mean')")

    #For each vertex pos, take difference from com (average across x, y, z)
    #Find the index of the vertex with the smallest distance from com (closests to com)
    vertex = np.argmin(
        np.mean(verts-com, axis=1), 
    )

    return(vertex)


def _abs_col_sum(x):
    return np.abs(x).sum(axis=1)

def compute_RLE(stc_true, stc_est, src_sim, src_recon, threshold="90%", per_sample=False):
    from mne.simulation.metrics import _uniform_stc, _thresholding
    from scipy.spatial.distance import cdist
    #stc_true, stc_est = _uniform_stc(stc_true, stc_est)

    if not per_sample: 
        #Apply threshold (removes points < threshold * max in abs data) if threshold is relative (in %), otherwise just removes below threshold - sets them to 0.0 
        #In both true and est
        stc_true, stc_est = _thresholding(stc_true, stc_est, threshold=threshold)

        #Compute dipole localization error 
        p = _abs_col_sum(stc_true._data)
        q = _abs_col_sum(stc_est._data)
        idx1 = np.nonzero(p)[0] #non zero vertices (thresholded) in true stc 
        idx2 = np.nonzero(q)[0] #non zero vertices (thresholded) in est stc 
        points_sim = []
        points_recon = []
        
        for i in range(len(src_sim)):
            points_sim.append(src_sim[i]['rr'][stc_true.vertices[i]])
        for i in range(len(src_recon)):
            points_recon.append(src_recon[i]['rr'][stc_est.vertices[i]])
        
        points_sim = np.concatenate(points_sim, axis=0)
        points_recon = np.concatenate(points_recon, axis=0)

        if len(idx1) and len(idx2):
            D = cdist(points_sim[idx1], points_recon[idx2]) #matrix (idx1, idx2)
            D_min_1 = np.min(D, axis=0) #len = idx2 
            D_min_2 = np.min(D, axis=1) #len = idx1
            return (np.mean(D_min_1) + np.mean(D_min_2))/2.0 #original code uses floor division // (but with head coords this will always give 0.0??)

        else: 
            return(np.inf)


""" 


################# TESTING FUNCTIONS ##################
#Transform vertex position coords for volume labels 
fname_aseg = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/fsaverage/mri/aparc+aseg.mgz'
subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
subject='fsaverage'
fname_bem = os.path.join(subjects_dir, subject, 'bem','fsaverage-5120-5120-5120-bem-sol.fif')
raw_fname =  '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_trans = 'fsaverage'

raw_sample = mne.io.read_raw_fif(raw_fname)
info = raw_sample.info


#volume_labels = ["Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"]
volume_labels = ["Left-Thalamus-Proper", "Right-Thalamus-Proper"]
#cer_lh_verts, cer_rh_verts = get_vol_label_vertices(fname_aseg, volume_labels)
cer_lh_verts, cer_rh_verts = get_vol_label_vertices(fname_aseg, volume_labels, units='m')

src_vol_test = mne.read_source_spaces('/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/volume-5.0_mm-fsaverage-src.fif')
nn = np.concatenate([src_vol_test[0]['nn'], src_vol_test[0]['nn']])#Taken from an old volume src, the nn's are all identical for all volume srcs and all vertices 
nn_lh = nn[0:len(cer_lh_verts),] 
nn_rh = nn[0:len(cer_rh_verts),] 

pos_vol_lh = dict()
pos_vol_lh['rr'] = cer_lh_verts
pos_vol_lh['nn'] = nn_lh

pos_vol_rh = dict()
pos_vol_rh['rr'] = cer_rh_verts
pos_vol_rh['nn'] = nn_rh

src_vol_lh = mne.setup_volume_source_space(
    subject='fsaverage',
    #mri = fname_aseg,
    mri = None, 
    #pos = 2.0, 
    pos = pos_vol_lh, 
    #sphere=(0, 0, 0, 0.12)
    bem=fname_bem,
    #volume_label=volume_labels,
    subjects_dir=subjects_dir,
    sphere_units="m",
)

src_vol_rh = mne.setup_volume_source_space(
    subject='fsaverage',
    #mri = fname_aseg,
    mri = None, 
    #pos = 2.0, 
    pos = pos_vol_rh, 
    #sphere=(0, 0, 0, 0.12)
    bem=fname_bem,
    #volume_label=volume_labels,
    subjects_dir=subjects_dir,
    sphere_units="m",
)

src_vol = src_vol_lh + src_vol_rh
#ISSUE: src_vol must have a src_vol[0]["shape"] key and values for the Brain() class plots to plot them as volume 
# - The test volume src I created/loaded above, has the same "shape" for all the vol srcs in there
# - Now just extracting that and add to my src_vol 
src_vol[0]["shape"] = src_vol_test[0]["shape"]
src_vol[1]["shape"] = src_vol_test[0]["shape"]

#ISSUE: src_vol must have a src_vol[0]["src_mri_t"] key and values for the Brain() class to plot them as volume
# - The test volume src i created/loaded above, ahs the same transfomr for all the vol srcs in there 
# - Now just extracting that and add to my src_vol 
src_vol[0]["src_mri_t"] = src_vol_test[0]["src_mri_t"]
src_vol[1]["src_mri_t"] = src_vol_test[0]["src_mri_t"]


fwd_vol = mne.make_forward_solution(info, fname_trans, src_vol,fname_bem, mindist=5.0)
#the fwd_vol["src"][0]["shape"] is now present 
#the fwd_vol["src"][0]["src_mri_t"] is now present 

label_cer_lh = mne.Label(
    vertices=src_vol[0]['vertno'],
    pos=src_vol[0]['rr'],
    values=None, 
    hemi='lh',
    name='Left-Thalamus-Proper',
    subject='fsaverage'
)

label_cer_rh = mne.Label(
    vertices=src_vol[1]['vertno'],
    pos=src_vol[1]['rr'],
    values=None, 
    hemi='rh',
    name='Right-Thalamus-Proper',
    subject='fsaverage'
)
#Plot all vertices in cerebellum (full label)
#brain.add_foci(surf_labels.pos, coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) #in correct position 
Brain = mne.viz.get_brain_class()
brain = Brain(
    'fsaverage',
    hemi='lh',
    surf='white',
    alpha=0.5,
    background='black',
    cortex='low_contrast',
    units='m',
    subjects_dir=subjects_dir
)
#brain.add_foci(cer_lh_verts, coords_as_verts=False, color='green', alpha=0.4, scale_factor=0.2)
brain.add_foci(surf_labels.vertices, coords_as_verts=True, color='white', alpha=0.4, scale_factor=0.2)
brain.add_foci(v1_label_20.vertices, coords_as_verts=True, color='blue', alpha=1, scale_factor=0.2)
brain.add_foci(v1_label_10.vertices, coords_as_verts=True, color='darkred', alpha=1, scale_factor=0.2)
brain.add_foci(v1_com, coords_as_verts=True, color='red', alpha=1, scale_factor=0.3)

v1_com = mne.Label.center_of_mass(surf_labels, surf='white', subject=subject, subjects_dir=subjects_dir, restrict_vertices=True)
v1_label_10 = mne.label.select_sources(
                subject, 
                surf_labels,
                location='center', #uses center of mass (CoM) of provided label as seed 
                extent=10.0, 
                grow_outside=False,
                subjects_dir=subjects_dir,
                name = "V1",
                random_state=32
            )

v1_label_20 = mne.label.select_sources(
                subject, 
                surf_labels,
                location='center', #uses center of mass (CoM) of provided label as seed 
                extent=20.0, 
                grow_outside=False,
                subjects_dir=subjects_dir,
                name = "V1",
                random_state=42
            )



#### Figure out how to compute the center of a region in vol src 

#Find centroid 
com_med, com_mean = _center_of_mass(cer_lh_verts)

Brain = mne.viz.get_brain_class()
brain = Brain(
    'fsaverage',
    hemi='lh',
    surf='pial',
    alpha=0.5,
    background='white',
    cortex='low_contrast',
    units='m',
    subjects_dir=subjects_dir
)
# brain.add_foci(cer_lh_verts, coords_as_verts=False, color='red', hemi='lh', alpha=0.2, scale_factor=0.2)
# brain.add_foci(cer_lh_verts[com_med], coords_as_verts=False, color='blue', hemi='lh', alpha=0.8, scale_factor=0.5)
# brain.add_foci(cer_lh_verts[com_mean], coords_as_verts=False, color='green', hemi='lh', alpha=0.8, scale_factor=0.5)
brain.add_foci(([x.max(), y.max(), z.max()]), coords_as_verts=False, color='green', hemi='lh', alpha=0.8, scale_factor=0.5)
brain.add_foci(([x.min(), y.min(), z.min()]), coords_as_verts=False, color='blue', hemi='lh', alpha=0.8, scale_factor=0.5)
####


#Grow labels from seed - volume specific function 
new_cer_labels = grow_labels(
    src_vol, 
    subject,
    seeds=[50,50],
    extents=5.0,
    hemis=["lh","rh"],
    subjects_dir=subjects_dir,
    n_jobs=None,
    names=["Cerebellum-lh", "Cerebellum-rh"],
    colors=None,
)


Brain = mne.viz.get_brain_class()
brain = Brain(
    'fsaverage',
    hemi='both',
    surf='white',
    alpha=0.5,
    background='white',
    cortex='low_contrast',
    units='m',
    subjects_dir=subjects_dir
)
brain.add_foci(new_cer_labels[0].pos, coords_as_verts=False, color='red', hemi='lh', alpha=0.4, scale_factor=0.2)
brain.add_foci(new_cer_labels[1].pos, coords_as_verts=False, color='blue', hemi='rh', alpha=0.4, scale_factor=0.2)



#Simulate
def data_fun():
    #Generate random source time courses.
    rng = np.random.RandomState(42)

    return (
        50e-9 #50 nAm
        * np.sin(30.0 * times)
        * np.exp(-((times - 0.15 + 0.05 * rng.randn(1)) ** 2) / 0.01)
    )

tstep = 1.0 / info['sfreq']
times = np.arange(100, dtype=np.float64)*tstep

n_events = 100
events = np.zeros((n_events, 3), int)
events[:, 0] = 200 * np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

source_time_series = data_fun()

source_simulator = mne.simulation.SourceSimulator(fwd_vol['src'], tstep=tstep)
source_simulator.add_data(label_cer_lh, source_time_series, events)

stc = source_simulator.get_stc()


#### CHECK IF I CAN CREATE A VolSourceEstimate FROM THIS (discrete) SourceEstimate 
from mne.source_estimate import VolSourceEstimate

data = stc.data # (n_vertices, n_times)
vertices = stc.vertices #index of dipoles in source space (n_dipoles)
tmin = stc.tmin
tstep = stc.tstep
subject = stc.subject

stc_vol = VolSourceEstimate(data=data, vertices=vertices, tmin=tmin, tstep=tstep, subject=subject, verbose=None)


#Plotting the original stc (discrete)
mne.viz.plot_source_estimates(
    stc, subject=subject, subjects_dir=subjects_dir, surface='white', hemi='lh', src=fwd_vol['src'],
    alpha=0.2, 
)   

#Plotting the converted stc (VolSourceEstimate)
mne.viz.plot_source_estimates(
    stc_vol, subject=subject, subjects_dir=subjects_dir, surface='white', hemi='lh', src=fwd_vol['src'],
    alpha=0.2, 
)   

#Testing own modified function 
plot_source_estimate_discrete(stc_vol, subject=subject, subjects_dir=subjects_dir, surface="white", src=fwd_vol['src'])





surf_labels = mne.read_labels_from_annot(subject, regexp='occipital', subjects_dir=subjects_dir)[0] #lateraloccipital-lh


#Get positions of the volume regions to use as pos input to create an src 
#volume_labels = ["Left-Cerebellum-Cortex", "Left-Thalamus-Proper", "Left-Caudate", "Left-Hippocampus"] 
volume_labels = ["Left-Cerebellum-Cortex", "Right-Cerebellum-Cortex"]
#cer_lh_verts, thal_lh_verts, caud_lh_verts, hippo_lh_verts = get_vol_label_vertices(fname_aseg, volume_labels)
cer_lh_verts, cer_rh_verts = get_vol_label_vertices(fname_aseg, volume_labels)

Brain = mne.viz.get_brain_class()
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
#brain.add_foci(surf_labels.pos, coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) #in correct position 
brain.add_foci(cer_lh_verts, coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2)
# brain.add_foci(thal_lh_verts, coords_as_verts=False, color='green', alpha=0.4, scale_factor=0.2)
# brain.add_foci(caud_lh_verts, coords_as_verts=False, color='blue', alpha=0.4, scale_factor=0.2)
# brain.add_foci(hippo_lh_verts, coords_as_verts=False, color='orange', alpha=0.4, scale_factor=0.2)

src_vol = mne.read_source_spaces('/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/volume-5.0_mm-fsaverage-src.fif')
nn = np.concatenate([src_vol[0]['nn'], src_vol[0]['nn']])#Taken from an old volume src, the nn's are all identical for all volume srcs and all vertices 
nn_lh = nn[0:len(cer_lh_verts),] 
nn_rh = nn[0:len(cer_rh_verts),] 

pos_vol_lh = dict()
pos_vol_lh['rr'] = cer_lh_verts
pos_vol_lh['nn'] = nn_lh

pos_vol_rh = dict()
pos_vol_rh['rr'] = cer_rh_verts
pos_vol_rh['nn'] = nn_rh

### 
src_surf = mne.setup_source_space(subject, spacing='oct6', add_dist="patch", subjects_dir=subjects_dir) 
fwd_surf = mne.make_forward_solution(info, fname_trans, src_surf, fname_bem, mindist=5.0)

src_vol_lh = mne.setup_volume_source_space(
    subject='fsaverage',
    #mri = fname_aseg,
    mri = None, 
    #pos = 2.0, 
    pos = pos_vol_lh, 
    #sphere=(0, 0, 0, 0.12)
    bem=fname_bem,
    #volume_label=volume_labels,
    subjects_dir=subjects_dir,
    sphere_units="m",
)

src_vol_rh = mne.setup_volume_source_space(
    subject='fsaverage',
    #mri = fname_aseg,
    mri = None, 
    #pos = 2.0, 
    pos = pos_vol_rh, 
    #sphere=(0, 0, 0, 0.12)
    bem=fname_bem,
    #volume_label=volume_labels,
    subjects_dir=subjects_dir,
    sphere_units="m",
)

src_vol = src_vol_lh + src_vol_rh

cer_rr = src_vol[0]['rr'][src_vol[0]['inuse']]
thal_rr = src_vol[1]['rr'][src_vol[1]['inuse']]
caud_rr = src_vol[2]['rr'][src_vol[2]['inuse']]
hippo_rr = src_vol[3]['rr'][src_vol[3]['inuse']]

Brain = mne.viz.get_brain_class()
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
brain.add_foci(src_surf[0]['rr'][surf_labels.vertices], coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) # rr positions are correct!!
#brain.add_foci(fwd_surf['src'][0]['rr'][surf_labels.vertices], coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) # rr positions NOT correct 
# brain.add_foci(src_vol[0]['rr'], coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) # rr positions are correct!!
# brain.add_foci(thal_rr, coords_as_verts=False, color='green', alpha=0.4, scale_factor=0.2)
# brain.add_foci(caud_rr, coords_as_verts=False, color='blue', alpha=0.4, scale_factor=0.2)
# brain.add_foci(hippo_rr, coords_as_verts=False, color='orange', alpha=0.4, scale_factor=0.2)

#Generate fwd 
fname_trans = 'fsaverage'
fwd_vol = mne.make_forward_solution(info, fname_trans, src_vol,fname_bem, mindist=5.0)

#Compare n vertices in src and fwd 
src_vol #lh = 21354 vertices, 'rr' in MRI (RAS) coords 
fwd_vol['src'] #lh = 19190 vertices, 'rr' in head coords 

## When plotting "rr" from fwd['src'] they appear in the wrong place, because they are not in head coords instead of MRI (RAS)
Brain = mne.viz.get_brain_class()
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
brain.add_foci(fwd_vol['src'][0]['rr'], coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) 

#Create labels from these (giving them arbitrary vertex numbers, but correct pos)
# - Now just using 2 of the vertices in cerebellum to test 
# - The vert indices (vertno) must correspond to the same vertex in the src['vertno']
label_cer_lh = mne.Label(
    vertices=src_vol[0]['vertno'][10:12],
    pos=src_vol[0]['rr'][10:12],
    values=None, 
    hemi='lh',
    name='Left-Cerebellum-Cortex',
    subject='fsaverage'
)

label_cer_rh = mne.Label(
    vertices=src_vol[1]['vertno'][10:12],
    pos=src_vol[1]['rr'][10:12],
    values=None, 
    hemi='rh',
    name='Right-Cerebellum-Cortex',
    subject='fsaverage'
)

## Center of mass not working for these labels (the resulting vertex is not in the center)
label_cer_lh_com = label_cer_lh.center_of_mass(subject='fsaverage', subjects_dir=subjects_dir, restrict_vertices=True)
label_cer_rh_com = label_cer_rh.center_of_mass(subject='fsaverage', subjects_dir=subjects_dir, restrict_vertices=True)

Brain = mne.viz.get_brain_class()
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
brain.add_foci(label_cer_lh.pos, coords_as_verts=False, color='red', alpha=0.4, scale_factor=0.2) # rr positions are correct!!
#brain.add_foci(label_cer_lh.pos[label_cer_lh.vertices==label_cer_lh_com], coords_as_verts=False, color='red')
brain.add_foci(label_cer_rh.pos, coords_as_verts=False, color='green', alpha=0.4, scale_factor=0.2)
#brain.add_foci(label_cer_rh.pos[label_cer_rh.vertices==label_cer_rh_com], coords_as_verts=False, color='red')


### Testing if center_of_mass function works for volume labels 
vol_src_path = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/volume-5.0_mm-fsaverage-src.fif'
test_src = mne.read_source_spaces(vol_src_path)
labels = mne.get_volume_labels_from_src(test_src, subject=subject,subjects_dir=subjects_dir)
thal_lab = labels[0]
thal_com = thal_lab.center_of_mass(subject=subject, subjects_dir=subjects_dir, restrict_vertices=True)

Brain = mne.viz.get_brain_class()
brain = Brain(
    'fsaverage',
    hemi='lh',
    surf='white',
    alpha=0.5,
    background='black',
    cortex='low_contrast',
    units='m',
    subjects_dir=subjects_dir
)
brain.add_foci(thal_lab.pos, coords_as_verts=False, color='green', alpha=1, scale_factor=0.2) # rr positions are correct!!
brain.add_foci(v1_surf, coords_as_verts=True, color='red', alpha=1, scale_factor=0.2)

all_surf = src_surf[0]['vertno']
v1_surf = [w for w in all_surf if w in surf_labels.vertices] #n=155


fig = mne.viz.create_3d_figure(size=(600, 400))
# Plot the cortex
mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    trans=fname_trans,
    surfaces='white',
    coord_frame="mri",
    fig=fig,
)
# Show the three dipoles defined at each location in the source space
mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    trans=fname_trans,
    fwd=fwd_surf,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)
mne.viz.set_3d_view(figure=fig, azimuth=180, distance=1, focalpoint="auto")


#brain.add_foci(thal_lab.pos[thal_lab.vertices==thal_com], coords_as_verts=False, color='red', scale_factor=0.2)


#Check if I can simulate directly wtih this Label 
def data_fun():
    #Generate random source time courses.
    rng = np.random.RandomState(42)

    return (
        50e-9 #50 nAm
        * np.sin(30.0 * times)
        * np.exp(-((times - 0.15 + 0.05 * rng.randn(1)) ** 2) / 0.01)
    )

tstep = 1.0 / info['sfreq']
times = np.arange(100, dtype=np.float64)*tstep

n_events = 100
events = np.zeros((n_events, 3), int)
events[:, 0] = 200 * np.arange(n_events)  # Events sample.
events[:, 2] = 1  # All events have the sample id.

source_time_series = data_fun()

source_simulator = mne.simulation.SourceSimulator(fwd_vol['src'], tstep=tstep)
source_simulator.add_data(label_cer_lh, source_time_series, events)
#source_simulator = mne.simulation.SourceSimulator(fwd_surf['src'], tstep=tstep)
#source_simulator.add_data(surf_labels, source_time_series, events)

stc = source_simulator.get_stc()

#Testing the original function 
mne.viz.plot_source_estimates(
    stc, subject=subject, subjects_dir=subjects_dir, surface='white', hemi='lh', src=fwd_vol['src'],
    alpha=0.2, 
)   
#For surf = source is plotted correct (MNI coords are correct)
#For vol = source is not plotted correctly (MNI coords are incorrect)
#ORIGINAL PLOT_SOURCE_ESTIMATE() 
    #- For cerebellum label vertices 10 and 11 - shows L:10, MNI = -53.1, -24.0, -4.0 
#MY PLOT_SOURCE_ESTIMATE_DISCRETE()
plot_source_estimate_discrete(
    stc, subject=subject, subjects_dir=subjects_dir, surface='white', hemi='lh', src=fwd_vol['src'],
    alpha=0.2, 
)


###################################################################
#    CHECK WHAT TRANSFORMS THE PLOTTING STC FUNC DOES 
###################################################################

########## FROM plot_source_estimate() 
#Params inputted 
stc = source_simulator.get_stc()
subject = subject
subjects_dir = subjects_dir 
surface = 'white' #cannot be "inflated" with vol vertices 
hemi='lh'
colormap='auto'
time_label='auto'
smoothing_steps = 10
transparent=True
alpha=1.0 
time_viewer='auto'
figure=None
views='auto'
colorbar=True
clim='auto'
cortex='classic'
size=800
background='black'
foreground=None
initial_time=None
time_unit='s'
backend='auto'
spacing='oct6'
title=None
show_traces='auto'
src=fwd_vol['src']
volume_options=1.0
view_layout='vertical'
add_data_kwargs=None
brain_kwargs=None

from mne.source_estimate import _check_stc_src, _BaseSourceEstimate, _BaseVolSourceEstimate
from mne.utils.check import _validate_type, _check_subject, _check_option
from mne.utils.config import get_subjects_dir
from mne.viz.backends.renderer import _get_3d_backend, use_3d_backend, get_brain_class
from mne.viz._3d import _plot_stc, _check_views, _handle_time, _check_st_tv, _process_clim, _check_volume, _separate_map, _handle_time
import warnings
from mne.viz._brain._brain import _update_limits

_check_stc_src(stc, src) #all good 
_validate_type(stc, _BaseSourceEstimate, "stc", "source estimate") #all good 
subjects_dir = get_subjects_dir(subjects_dir=subjects_dir, raise_error=True) #returns posixpath (full path)
subject = _check_subject(stc.subject, subject) #all good 
plot_mpl = backend == 'matplotlib' #false 
if not plot_mpl: 
    if backend=='auto':
            backend = _get_3d_backend() #pyvistaqt 

kwargs = dict(
        subject=subject,
        surface=surface,
        hemi=hemi,
        colormap=colormap,
        time_label=time_label,
        smoothing_steps=smoothing_steps,
        subjects_dir=subjects_dir,
        views=views,
        clim=clim,
        figure=figure,
        initial_time=initial_time,
        time_unit=time_unit,
        background=background,
        time_viewer=time_viewer,
        colorbar=colorbar,
        transparent=transparent,
    )

with use_3d_backend(backend):
    _plot_stc(
        stc, 
        overlay_alpha=alpha, 
        brain_alpha=alpha, 
        vector_alpha=alpha, 
        cortex=cortex, 
        foreground=foreground,
        size=size,
        scale_factor=None, 
        show_traces=show_traces,
        src=src, 
        volume_options=volume_options,
        view_layout=view_layout, 
        add_data_kwargs=add_data_kwargs,
        brain_kwargs=brain_kwargs,
        title=title,
        **kwargs,
    )
#blue dot has MNI coords (which are in the wrong position)
#blue dot has "L=10" (probably the vertex number) - simualted vertex 10 and 11 from src_vol (they are in the label)

####### FROM _plot_stc()
#params inputted (+ all defined above in kwargs)
stc=stc
overlay_alpha=alpha
brain_alpha=alpha
vector_alpha=alpha 
cortex=cortex
foreground=foreground
size=size
scale_factor=None
show_traces=show_traces
src=src
volume_options=volume_options
view_layout=view_layout
add_data_kwargs=add_data_kwargs
brain_kwargs=brain_kwargs
title=title

vec = stc._data_ndim==3 #False 
subjects_dir = str(get_subjects_dir(subjects_dir=subjects_dir, raise_error=True)) #returns full path 
subject = _check_subject(stc.subject, subject) #all good 

backend = _get_3d_backend() #pyvistaqt 
del _get_3d_backend
Brain = get_brain_class()
views = _check_views(surface, views, hemi, stc, backend) # returns ['lateral']
_check_option("hemi", hemi, ['lh','rh','split','both']) #returns "lh"
_check_option("view_layout", view_layout, ("vertical","horizontal")) #returns "vertical"
time_label, times = _handle_time(time_label, time_unit, stc.times) #stc.times has len 19900
#times = array of shape (19900,)
show_traces, time_viewer = _check_st_tv(show_traces, time_viewer, times)
#show_traces = True
#time_viewer = True 

#Convert control points to locations in colormap (setting color limits based on data values)
use = stc.magnitude().data if vec else stc.data #returns array of shape (2, 19900)
mapdata = _process_clim(clim, colormap, transparent, use, allow_pos_lims=not vec)

volume = _check_volume(stc, src, surface, backend) #returns False (hence - assuming that it is a surface!)

_separate_map(mapdata)
colormap = mapdata['colormap']
diverging = "pos_lims" in mapdata['clim'] #True 
scale_pts = mapdata["clim"]["pos_lims" if diverging else "lims"] #takes the scale points (for colormap), based on the stc data 
transparent = mapdata["transparent"] #True 
del mapdata

if hemi in ["both", "split"]:
    hemis=["lh", "rh"]
else: 
    hemis=[hemi] 

if overlay_alpha is None:
    overlay_alpha=brain_alpha
if overlay_alpha==0: 
    smoothing_steps = 1 #disabling smoothing to save time 
#overlay_alpha = 1.0 
    
sub_info = subject if len(hemis) > 1 else f"{subject} - {hemis[0]}" #fsaverage - lh
title = title if title is not None else sub_info #fsaverage - lh 

kwargs = {
        "subject": subject,
        "hemi": hemi,
        "surf": surface,
        "title": title,
        "cortex": cortex,
        "size": size,
        "background": background,
        "foreground": foreground,
        "figure": figure,
        "subjects_dir": subjects_dir,
        "views": views,
        "alpha": brain_alpha,
    }

if brain_kwargs is not None: #False 
        kwargs.update(brain_kwargs)
kwargs["show"] = False
kwargs["view_layout"] = view_layout #vertical 

with warnings.catch_warnings(record=True):  # traits warnings
        brain = Brain(**kwargs) #create mne.viz.Brain obj 

if scale_factor is None: #True 
    #configure the glyphs scale directly 
    width = np.mean(
        [
            np.ptp(brain.geo[hemi].coords[:,1])
            for hemi in hemis 
            if hemi in brain.geo
        ]
    ) #np.float(167,38) 
    scale_factor = 0.025 * width / scale_pts[-1] #np.float64(91144777.34863009)

if transparent is None: #False 
    transparent = True

center = 0.0 if diverging else None #0.0 

kwargs = {
        "array": stc,
        "colormap": colormap,
        "smoothing_steps": smoothing_steps,
        "time": times,
        "time_label": time_label,
        "alpha": overlay_alpha,
        "colorbar": colorbar,
        "vector_alpha": vector_alpha,
        "scale_factor": scale_factor,
        "initial_time": initial_time,
        "transparent": transparent,
        "center": center,
        "fmin": scale_pts[0],
        "fmid": scale_pts[1],
        "fmax": scale_pts[2],
        "clim": clim,
        "src": src,
        "volume_options": volume_options,
        "verbose": None,
    }

if add_data_kwargs is not None: #False 
    kwargs.update(add_data_kwargs)

## Adding surface data 
for hemi in hemis: 
    if isinstance(stc, _BaseVolSourceEstimate): #no surf data (else for surf or mixed also add surf data)
        #false in our case with discrete 
        break 

    vertices = stc.vertices[0 if hemi == 'lh' else 1] #vertices that were present in the label used for sim 
    if len(vertices)==0:
        continue 
    use_kwargs = kwargs.copy()
    use_kwargs.update(hemi=hemi)
    with warnings.catch_warnings(record=True):
        brain.add_data(**use_kwargs)

# - FROM brain.add_data() 
#Params added in additon to the **use_kwargs inputted
time_label_size=None       
array = stc #first element in use_kwargs
time=None
fmin = kwargs['fmin']
fmid = kwargs['fmid']
fmax = kwargs['fmax']
center = kwargs['center']

if time_label_size is not None: 
    time_label_size = float(time_label_size)

hemi = brain._check_hemi(hemi, extras=["vol"]) #lh 
stc, array, vertices = brain._check_stc(hemi, array, vertices)
array = np.asarray(array)
vector_alpha = alpha if vector_alpha is None else vector_alpha #1.0 
brain._data["vector_alpha"] = vector_alpha
brain._data["scale_factor"] = scale_factor

#Create time array and add label if > 1D 
if array.ndim <=1: #False 
    time_idx=0
else: 
    #check time array 
    if time is None:
        time = np.arange(array.shape[-1]) # array range(0, 19900)
    else: 
        time = np.asarray(time)
        if time.shape != (array.shape[-1],):
            raise ValueError(
                f"time has shape {time.shape}, but need shape {(array.shape[-1],)} array.shape[-1]"
            )
    brain._data["time"] = time

    if brain._n_times is None: #True 
        brain._times = time 
    elif len(time) != self._n_times: 
        raise ValueError()
    elif not np.array_equal(time, self._times):
        raise ValueError()
    
    #initial time 
    if initial_time is None: #true 
        time_idx = 0
    else: 
        time_idx = brain._to_time_index(initial_time)

#Time label 
time_label, _ = _handle_time(time_label, "s", time) #function 
y_txt = 0.05 + 0.1 * bool(colorbar) #0.15000000000000002

if array.ndim == 3: #False 
    if array.shape[1] !=3: 
        raise ValueError()

fmin, fmid, fmax = _update_limits(fmin, fmid, fmax, center, array)
if colormap=="auto": #False 
    colormap = "mne" if center is not None else "hot"
if smoothing_steps is None: #False 
    smoothing_steps=7
elif smoothing_steps = "nearest": #False 
    smoothing_steps = -1 
elif isinstance(smoothing_steps, int):
    if smoothing_steps <0: #False 
        raise ValueError()
else:
    raise TypeError()

brain._data["stc"] = stc
brain._data["src"] = src
brain._data["smoothing_steps"] = smoothing_steps
brain._data["clim"] = clim
brain._data["time"] = time
brain._data["initial_time"] = initial_time
brain._data["time_label"] = time_label
brain._data["initial_time_idx"] = time_idx
brain._data["time_idx"] = time_idx
brain._data["transparent"] = transparent
#data spec for hemi 
brain._data[hemi] = dict()
brain._data[hemi]["glyph_dataset"] = None
brain._data[hemi]["glyph_mapper"] = None
brain._data[hemi]["glyph_actor"] = None
brain._data[hemi]["array"] = array
brain._data[hemi]["vertices"] = vertices
brain._data["alpha"] = alpha
brain._data["colormap"] = colormap
brain._data["center"] = center
brain._data["fmin"] = fmin
brain._data["fmid"] = fmid
brain._data["fmax"] = fmax
brain._update_colormap_range()

# 1) add the surfaces first 
actor = None
for _ in brain._iter_views(hemi):
    if hemi in ("lh", "rh"): 
        actor = brain._layered_meshes[hemi]._actor
        print('- Adding normal actor')
    else: 
        src_vol = src[2:] if src.kind=="mixed" else src 
        actor, _ = brain._add_volume_data(hemi, src_vol, volume_options) #unsure when this would be activated (gives error )
        print('- Adding add_volume_data actor') 
assert actor is not None #should ahve added one 
brain._add_actor("data", actor)
### The actor has a "center" param with what looks like MNI coords, also position coords with (0, 0, 0)
# - Maybe just the center of the brain? 
# - Think this is the blue dot (has a color param of lightblue, a scale, and pickabel=True and visible=True)


# 2) update time and smoothing parameters 
# set_data_smoothing calls "_update_current_time_idx" for us, which will set _current_time 
brain.set_time_interpolation(brain.time_interpolation)
brain.set_data_smoothing(brain._data["smoothing_steps"])

# 3) add the other actors 
if colorbar is True: 
    #bottom left by default 
    colorbar = (brain._subplot_shape[0] -1, 0)
for ri, ci, v in brain._iter_views(hemi): #iterates though hemis per view, here just one view (lateral)
    print(ri) #0 
    print(ci) #0 
    print(v) #lateral 
    #Add the time label to the bottommost view 
    do = (ri, ci) == colorbar
    if not brain._time_label_added and time_label is not None and do: #True 
        time_actor = brain._renderer.text2d(
            x_window=0.95, 
            y_window=y_txt, 
            color=brain._fg_color, 
            size=time_label_size, 
            text=time_label(brain._current_time),
            justification="right",
        )
        brain._data["time_actor"] = time_actor
        brain._time_label_added = True
    
    if colorbar and brain._scalar_bar is None and do: #True 
        kwargs = dict(
            source=actor, 
            nlabels=8, 
            color=brain._fg_color,
            bgcolor=brain._brain_color[:3]
        )



################################################################################
################################################################################


################################################################################
#   TESTING WHAT THE SIMUALTE_STC() (called by the .get_stc()) is doing

### get_stc() 
start_sample = None
stop_sample = None
start_sample = source_simulator.first_samp
stop_sample = start_sample + source_simulator.n_times-1
n_samples = stop_sample - start_sample + 1 # 19900 

#Initialize the stc_data array to span all possible samples 
stc_data = np.zeros((len(source_simulator._labels), n_samples)) # (100, 19900)

#Select only the events that fall within the span 
ind = np.where(np.logical_and(
    source_simulator._last_samples >= start_sample, source_simulator._events[:, 0] <= stop_sample
))[0] #len = 100 

#Loop only over the items that are in the time span 
subset_waveforms = [source_simulator._waveforms[i] for i in ind] # len=100 
for i, (waveform, event) in enumerate(zip(subset_waveforms, source_simulator._events[ind])):
    #we retrieve the first and last sample of each waveform 
    #according to teh corresponding event 
    wf_start = event[0]
    wf_stop = source_simulator._last_samples[ind[i]]

    #Recover the indices of the event that should be in the chunk 
    waveform_ind = np.isin(
        np.arange(wf_start, wf_stop + 1),
        np.arange(start_sample, stop_sample + 1)
    ) # = all of the samples in event, if start and stop samples are None (inputs)

    #Recover the indices that correspond to the overlap 
    stc_ind = np.isin(
        np.arange(start_sample, stop_sample+1),
        np.arange(wf_start, wf_stop+1),
    ) #bool, of len 19900 

    #add the resulting waveform chunk to the corresponding label 
    stc_data[ind[i]][stc_ind] += waveform[waveform_ind]

#NOTE 
    #stc_data now has shape (100, 19900) = (n_events in label, n timesamples of full source time course (including all events))
    #nothing has been done to the vertices/positions 
    #only waveforms are extracting for the right indices and added to an array of labels, timesamples 
    #label.pos vertex coords do not match those in source_simulator._src['rr'] because the inherent src is in 
    #head coords (from fwd) and the label.pos are in MRI (RAS) coords (from raw src)

start_sample -= source_simulator.first_samp #STC sample ref is 0 

stc = mne.simulation.source.simulate_stc(
    source_simulator._src, #the source space (in this case in head coords, if simulator is initiated with fwd['src'])
    source_simulator._labels, #the labels (in this case in MRI (RAS) coords, if created with rr pos from raw src)
    stc_data, # array, shape (n_labels, n_times) #if we have 100 events for one label, n_labels will be 100 
    start_sample*source_simulator._tstep, #tmin (the beginning of the timeseries)
    source_simulator._tstep, #tstep (1/sampling frequency)
    value_fun=None, #function to pply to the label values to obtain the waveform scaling ofr each vertex in the label. If None (default), uniform scaling is used (same waveform is applied to all vertices in label)
    allow_overlap=False, #allow overlapping labels or not, default is False
)

# return stc


### simulate_stc() 
src = source_simulator._src
labels = source_simulator._labels
tmin = start_sample*source_simulator._tstep #0.0
tstep = source_simulator._tstep #0.0067
value_fun=None
allow_overlap=False

vertno = [[], []]
stc_data_extended = [[], []] 
hemi_to_ind = {"lh": 0, "rh": 1}

for i, label in enumerate(labels):
    hemi_ind = hemi_to_ind[label.hemi]
    src_sel = np.intersect1d(src[hemi_ind]['vertno'], label.vertices) ##return the common elements in the two arrays (src.vertices and label.vertices)
    if len(src_sel)==0: #if no vertex number matches 
        #code that computes neartest vertices based on rr positions in src 
        continue
    if value_fun is not None: 
        #code that scales the vertex values 
        continue
    else: 
        data = np.tile(stc_data[i], (len(src_sel), 1)) # array (2, 19900) (n_vertices, n_samples)
    if allow_overlap: 
        #code that deals with them 
        continue
    #Extend the existing list instead of appendign it so that we can index its elements 
    vertno[hemi_ind].extend(src_sel)
    stc_data_extended[hemi_ind].extend(np.atleast_2d(data))

stc_data_extended # len = 2 (one per hemi), each of those has len=200 (100 per source vertex?), each of these is an array of shape (n_samples, ) (19900, )

vertno = [np.array(v) for v in vertno] #list of 2 (one per hemi), each is an array with shape 200 (vert1, vert2 * 100)
#iterating through the vertices in label n events times (10, 11, 10, 11, 10, 11.... n=200)
if not allow_overlap: 
    for v, hemi in zip(vertno, ("left", "right")):
        d=len(v) - len(np.unique(v))
        if d>0: 
            raise RuntimeError(
                "labels had overlap in hemi, they must be non-overlapping"
            )
        else: 
            print("All good")

#The data is in the order left, right 
data = list()
for i in range(2):
    if len(stc_data_extended[i]) != 0: 
        stc_data_extended[i] = np.vstack(stc_data_extended[i]) #reshaping to (200, 19900)
        #Ordder the indices of each hemisphere 
        idx = np.argsort(vertno[i]) #indices that sorts them to be ordered as 10*100, 11*100 (10 and 11 beging the vertex numbers simualted from)
        data.append(stc_data_extended[i][idx])
        #data is now a list of 2 (one per hemi), 
        #each element being an array of (n_source_vertices*n_events, n_samples) (200, 19900)
        vertno[i] = vertno[i][idx] #order vertno in increasing order

stc = mne.source_estimate.SourceEstimate(
    np.concatenate(data), 
    vertices=vertno, 
    tmin=tmin, 
    tstep=tstep, 
    subject=src._subject
)



################################################################################




################################################################################
#   TESTING WHAT THE SIMUALTE_RAW() is doing 
from mne._fiff.pick import pick_types, pick_channels
from mne.utils.config import _get_stim_channel
from mne.simulation.raw import _check_head_pos, _check_stc_iterable, _log_ch, _SimForwards, _stc_data_event
from mne.source_space._source_space import _ensure_src
from mne.forward.forward import restrict_forward_to_stc
from mne._ola import _Interp2
from mne.io.array._array import RawArray

stc=source_simulator
bem=None #can be None if fwd is provided
forward=fwd_vol
mindist=1.0
interp='cos2'
first_samp=0
max_iter=10000
trans=None #if trans is None, an identity transform will be used - also test with trans file from fsaverage as input to simulate_raw
head_pos=None #head movements can be optinally simulated using teh head_pos parameter 
n_jobs=None
use_cps=True

len(pick_types(info, meg=False, stim=True))==0 #False 
event_ch = pick_channels(info["ch_names"], _get_stim_channel(None, info))[0]

if forward is not None: 
    ch_types = info.get_channel_types(unique=True)
    src=forward['src']

dev_head_ts, offsets = _check_head_pos(head_pos, info, first_samp, None)
#computes a trans of shape (4,4)

src = _ensure_src(src, verbose=False)

#Extract necessary info 
meeg_picks = pick_types(info, meg=True, eeg=True, exclude=[])

stc_enum, stc_counted, verts = _check_stc_iterable(stc, info) #takin vertices (numbers) from the source_simulator obj 

if forward is not None: 
    forward = restrict_forward_to_stc(forward, verts) #reducing fwd to only include the two vertices (dipoles) simualted in source simulator
    src = forward['src'] #same for src - now only have the vertices simulated in source_simulator 

#array used to store results 
raw_datas = list()
_log_ch("Event information", info, event_ch)
n=1
get_fwd = _SimForwards(
    dev_head_ts, 
    offsets, 
    info, 
    trans, 
    src, 
    bem, 
    mindist, 
    n_jobs, 
    meeg_picks, 
    forward, 
    use_cps, 
) 
#_SimForwards object 

interper = _Interp2(offsets, get_fwd, interp)

this_start = 0
for n in range(max_iter):
    if isinstance(stc_counted[1], list | tuple): #true, stc_counted[1] has len=2 (each is a SourceEstimate obj with empty values and len data = simulated time series)
        this_n = stc_counted[1][0].data.shape[1] #1000 data points
    this_stop = this_start + this_n #=1000 
    n_doing = this_stop - this_start #=1000
    this_data = np.zeros((len(info['ch_names']), n_doing)) # (376, 1000) 
    raw_datas.append(this_data)
    #Stim channel 
    fwd, fi = interper.feed(this_stop- this_start) 
    # fwd = array (366, 2, 1000) (n_channels, n_dipoles, n_timesamples)
    # fi = array (1000, )
    fi = fi[0]
    stc_data, stim_data, _ = _stc_data_event(
        stc_counted, fi, info['sfreq'], get_fwd.src, None if n==0 else verts
    )
    # stc_data = array (2, 1000)
    # stim_data = array (1000, )

    if event_ch is not None:
        this_data[event_ch, :] = stim_data[:n_doing]
    this_data[meeg_picks] = np.einsum("svt, vt->st", fwd, stc_data)
    # - above line sums the matrices fwd (366, 2, 1000) (svt) with stc_data (2, 1000) vt, into a channels*time matrix (st = 366, 1000)
    try: 
        stc_counted = next(stc_enum)
    except StopIteration: 
        break 
    del fwd 
raw_data = np.concatenate(raw_datas, axis=-1)
raw = RawArray(raw_data, info, first_samp=first_samp)
raw.set_annotations(raw.annotations)

################################################################################
fname_cov = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/MNE-sample-data/MEG/sample/sample_audvis-cov.fif'
picks = mne.pick_types(info, meg=True, exclude="bads")
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd_vol)
raw = raw.pick(picks=["meg", "stim"], exclude="bads")
noise_cov = mne.read_cov(fname_cov)
mne.simulation.add_noise(raw, cov=noise_cov, iir_filter=None, random_state=42)
mne.simulation.add_eog(raw, random_state=42)
mne.simulation.add_ecg(raw, random_state=42)

stc = source_simulator.get_stc()

mne.viz.plot_source_estimates(
        stc, subject=subject, subjects_dir=subjects_dir, surface='white', hemi='both', src=fwd_vol['src'],
        alpha=0.2, 
    )       

src_sel = np.intersect1d(fwd_vol['src'][0]["vertno"], label_cer_lh.vertices)



#Find com of labels 
cer_lh_com = label_cer_lh.center_of_mass(subject='fsaverage', restrict_vertices=True, subjects_dir=subjects_dir, surf='sphere')
cer_rh_com = label_cer_rh.center_of_mass(subject='fsaverage', restrict_vertices=True, subjects_dir=subjects_dir, surf='sphere')


#Using select_sources() or grow_labels() cannot be used to define labels for volume, 
#as they are growing regions on a cortical surface mesh, not on a 3D volume 
#To "grow" a label aourd a peak in a 3D volume, you must define the source space as volumetric, find the 
#voxel index of the maximum and identify its neighbors within the source space






src_vol = mne.setup_volume_source_space(
    subject='fsaverage',
    mri = fname_aseg,
    pos = 5.0, 
    #sphere=(0, 0, 0, 0.12)
    #bem=fname_bem,
    volume_label=volume_labels,
    subjects_dir=subjects_dir,
    sphere_units="m",
)
#cer = 397
#thal = 68
#caudate = 30
#hippo = 39 

src_vol = mne.setup_volume_source_space(
    subject='fsaverage',
    mri = fname_aseg,
    pos = 5.0, 
    #sphere=(0, 0, 0, 0.12)
    bem=fname_bem,
    volume_label=volume_labels,
    subjects_dir=subjects_dir,
    sphere_units="m",
)
#cer=518 
#thal=68
#caudate=30
#hippo=39 




#####
from mne.surface import read_surface, mesh_dist
from mne.parallel import parallel_func
fname_vol_src = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/simulations/volume-5.0_mm-fsaverage-src.fif'
subject='fsaverage'
subjects_dir = '/Users/au553087/Library/CloudStorage/OneDrive-Aarhusuniversitet/Work/RCB/simulation_study/simulation_cortical_omission/data/freesurfer/subjects'
vol_src = mne.read_source_spaces(fname_vol_src)
labels = mne.get_volume_labels_from_src(vol_src, subject='fsaverage',subjects_dir=subjects_dir)

### BELOW: using code from grow_labels()
selected_label = [l for l in labels if l.name=='Thalamus-Proper-lh'][0]
# selected_label.vertices do NOT plto in the correct position on surface mesh 
# selected_label.pos DO plot in the correct position on surface mesh 

names = np.array('Thalamus-Proper-lh')
hemi='lh'
surface = 'white'
overlap = True
#load the surfaces and create the distance graphs 
tris, vert, dist = {}, {}, {}
surf_fname = os.path.join(subjects_dir, subject, 'surf', f'{hemi}.{surface}')
vert[hemi], tris[hemi] = read_surface(surf_fname)
dist[hemi] = mesh_dist(tris[hemi], vert[hemi])




if overlap: 
    #create the patches 
    parallel, my_grow_labels, n_jobs = parallel_func(__grow_labels, n_jobs) #_grow labels must be defined  """ 