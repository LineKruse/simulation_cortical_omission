"""
TO BE DONE:
- add option to use different helmet templates (e.g. beta2)
    Current challenge is that we have a orientation for each sensor described with a single vector (shape 3,) but MNE expects a 3x3 orientation matrix.
    Lacking the excel file that we have for the alpha1 helmet
- check that we are happy with the coverage of the sensors. We don't have that nice frontal coverage
- add dual and triaxial OPMs (currently only simulating single axis)
- figure out why that one sensor has a weird orientation

"""

from pathlib import Path
import pickle
import numpy as np
import mne
from mne.utils._bunch import NamedInt

import pyvista as pv

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from helmet_templates import load_helmet_template, TemplateBase


class OPMSensorLayout(TemplateBase):
    def __init__(self, labels, depth_mm, helmet_template, n_axis, coil_type:NamedInt = NamedInt("FieldLine OPM sensor Gen1 size = 2.00   mm", 8101), sensor_depth_offset_mm=52):
        self.depth_mm = np.asarray(depth_mm)
        self.helmet = helmet_template
        self.coil_type = coil_type
        self.offset = sensor_depth_offset_mm
        self.n_axis=n_axis

        chan_pos = self._compute_positions(labels)
        chan_ori = self._compute_orientations(labels)

        super().__init__(labels, self.helmet.unit)

        self.chan_pos = chan_pos
        self.chan_ori = chan_ori

        
    
    def _compute_positions(self, labels): #len_sleeve:float = 75/1000, offset:float = 13/1000
        template_ori = self.helmet.get_chs_ori(labels)
        template_pos = self.helmet.get_chs_pos(labels)
        
        # Create a new list to store the updated positions
        transformed_pos = []
        
        # Move template pos by measurement length in template ori direction
        for pos, ori, depth in zip(template_pos, template_ori, self.depth_mm):
            updated_depth = (52-depth)/1000
            x = (pos[0] - (updated_depth* ori[2,0]))
            y = (pos[1] - (updated_depth* ori[2,1]))
            z = pos[2] + (updated_depth * ori[2,2])

            transformed_pos.append([x, y, z])
        
        #add same sensors again (to allow 2 orientations) FIXME: for now using same orientations, must be updated when you have a good solution
        if self.n_axis==2: 
            for pos, ori, depth in zip(template_pos, template_ori, self.depth_mm):
                updated_depth = (52-depth)/1000
                x = (pos[0] - (updated_depth* ori[2,0]))
                y = (pos[1] - (updated_depth* ori[2,1]))
                z = pos[2] + (updated_depth * ori[2,2])

                transformed_pos.append([x, y, z])
            
        return np.array(transformed_pos)

    def _compute_orientations(self, labels):

        template_ori = self.helmet.get_chs_ori(labels)        
        
        if n_meas_axis==1: 
            transformed_ori = template_ori
        
        elif n_meas_axis==2: 
            transformed_ori = []

            #Append rotated ori for all sensor positions (first half)
            for idx in range(0, int(len(template_ori)/2)): 
                transformed_ori.append(template_ori[idx])

            #Append rotated ori for all sensor positions (second half)
            for idx in range(0, int(len(template_ori)/2)): 
        
                #Get original coordiantes and resphape into 3x3 (rep x, y and z directions)
                orig_coords = template_ori[idx] #testing on MEG 0123
                #R = np.array(orig_coords).reshape((3, 3),axis=1)
                R = orig_coords.copy() #already in 3x3 shape

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
                #new_coords = R_new.flatten().tolist()
                transformed_ori.append(R_new)

        return transformed_ori
            


def add_sensor_layout(mne_object, sensor_layout: OPMSensorLayout):
    """
    Updates channel positions and orientations for MNE object based on a sensor layout.
    Args:
        mne_object: MNE object, for example Raw.
        sensor_layout: A layout object containing channel positions, orientations, labels and coil type.
    """

    for pos, ori, label in zip(
        sensor_layout.chan_pos, sensor_layout.chan_ori, sensor_layout.labels
    ):
        idx = next(
            (
                idx
                for idx, ch in enumerate(mne_object.info["chs"])
                if ch["ch_name"] == label
            ),
            None,
        )
        if idx is None:
            print(f"Warning: Channel {label} not found in MNE object")
            continue

        mne_object.info["chs"][idx]["loc"][:3] = pos
        mne_object.info["chs"][idx]["loc"][3:] = ori.flatten()
        mne_object.info["chs"][idx]["coil_type"] = sensor_layout.coil_type
        mne_object.info["chs"][idx]["kind"] = mne.io.constants.FIFF.FIFFV_MEG_CH
        mne_object.info["chs"][idx]["unit"] = mne.io.constants.FIFF.FIFF_UNIT_T


def inflate_mesh(mesh: pv.PolyData, distance: float):
    mesh = mesh.copy()
    mesh.compute_normals(point_normals=True, inplace=True)
    mesh.points = mesh.points + distance * mesh.point_normals
    return mesh

if __name__ == "__main__":
    path = Path(__file__).parents[2]

    helmet = "alpha1"
    n_meas_axis = 1

    if helmet == "alpha1":
        # approximate translation and rotation to roughly align the template with the head surface (fsaverage). From here the sensors are projected to the head surface
        tx, ty, tz, rx, ry, rz = 0.0, -0.016, -0.043000000000000003, 19.5, 1.5, 0.0
    else:
        raise ValueError(f"Helmet {helmet} not supported yet")

    if n_meas_axis == 1:
        print("Using single axis OPMs")
        n_meas_axis_str = "single_axis"
    elif n_meas_axis == 2: 
        print("Using dual axis OPMs")
        n_meas_axis_str = "dual_axis"
    else:
        raise ValueError("Only single axis OPMs supported for now")

    # path to the template file
    template_path = path / "data" / "OPM" / "template" / f"FL_{helmet}_helmet.pkl"
    helmet_template = load_helmet_template(template_path)


    data_path = path / "data" / "MNE-sample-data" / "MEG" / "sample"
    subjects_dir = path / "data" / "freesurfer" / "subjects"
    bem_fname = subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    raw_fname = data_path / 'sample_audvis_filt-0-40_raw.fif'

    head_mri_trans = mne.Transform('head', 'mri')
    raw = mne.io.read_raw_fif(raw_fname, preload=False)
    raw.pick(["meg"])

    # Load BEM surfaces
    surf = mne.read_bem_surfaces(bem_fname)

    # find the one with "id"==4 (head surface)
    head_surf = None
    for s in surf:
        if s['id'] == 4:
            head_surf = s
            break

    if head_surf is None:
        raise ValueError("Head surface not found")

    print(f"Surface {head_surf['id']}: {head_surf['np']} points, {head_surf['ntri']} triangles")

    # to account for the fact that sensors won't be exactly on the scalp surface but a few mm above, we inflate the mesh by 4mm
    scalp_mesh = pv.PolyData(head_surf["rr"], faces=np.hstack([np.full((len(head_surf["tris"]), 1), 3), head_surf["tris"]]))
    head_surf = inflate_mesh(scalp_mesh, distance=4/1000)

    labels = helmet_template.labels*2 if n_meas_axis==2 else helmet_template.labels
    sensor_layout = OPMSensorLayout(
        labels=labels,
        depth_mm=[50]*len(helmet_template.labels),
        helmet_template=helmet_template,   
        n_axis=n_meas_axis
        # delete this line, but useful for plotting as orientation of OPMs (default coil types) are difficult to see on alignment plot
        #coil_type=NamedInt("SQ20950N", 3024)
        ) 
    #print(sensor_layout.labels)

    # rename to match the raw info channel names
    #sensor_layout.labels = [raw.ch_names[i] for i in range(len(helmet_template.labels))]
    sensor_layout.labels = [raw.ch_names[i] for i in range(len(labels))]

    add_sensor_layout(raw, sensor_layout)


    raw.pick(sensor_layout.labels)


    # -----------------------------
    # 1. head -> MRI transform
    # -----------------------------
    def head_dev_transform(tx, ty, tz, rx, ry, rz):
        t = np.array([tx, ty, tz])
        r = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()

        T = np.eye(4)
        T[:3, :3] = r
        T[:3, 3] = t
        return T


    head_mri_t = mne.transforms.Transform(
        'head', 'mri',
        head_dev_transform(tx, ty, tz, rx, ry, rz)
    )

    head_mri_t.save(fname=path / "data" / "OPM" / f"fsaverage_OPM_head_mri-trans.fif", overwrite=True)


    # -----------------------------
    # 2. MEG device -> head
    # -----------------------------
    dev_head_t = raw.info['dev_head_t']['trans']

    picks = mne.pick_types(raw.info, meg=True)

    sensor_pos_dev = np.array([
        raw.info['chs'][p]['loc'][:3]
        for p in picks
    ])

    sensor_pos_head = mne.transforms.apply_trans(
        dev_head_t,
        sensor_pos_dev
    )


    # -----------------------------
    # 3. head -> MRI
    # -----------------------------
    sensor_pos_mri = mne.transforms.apply_trans(
        head_mri_t['trans'],
        sensor_pos_head
    )


    # -----------------------------
    # 4. project onto head surface (MRI space)
    # -----------------------------
    head_vertices = np.asarray(head_surf.points)

    tree = cKDTree(head_vertices)
    _, idx = tree.query(sensor_pos_mri)

    projected_mri = head_vertices[idx]


    # -----------------------------
    # 5. MRI -> head
    # -----------------------------
    mri_head_t = np.linalg.inv(head_mri_t['trans'])

    projected_head = mne.transforms.apply_trans(
        mri_head_t,
        projected_mri
    )


    # -----------------------------
    # 6. write back
    # -----------------------------
    for i, p in enumerate(picks):
        raw.info['chs'][p]['loc'][:3] = projected_head[i]

    # turn head dev t to identiy
    raw.info['dev_head_t']['trans'] = np.eye(4)
    mne.viz.plot_alignment(raw.info, trans=head_mri_t, subjects_dir=subjects_dir, surfaces=["head", "brain"], coord_frame="head", subject="fsaverage", meg="sensors")

    # save the info for later use
    raw.info.save(path / "data" / "OPM" / f"fsaverage_OPM_{helmet}_{n_meas_axis_str}-info.fif", overwrite=True)
    # with open(template_path.parents[1] / f"fsaverage_OPM_{helmet}_{n_meas_axis_str}-info.fif", "wb") as f:
    #     pickle.dump(raw.info, f)
