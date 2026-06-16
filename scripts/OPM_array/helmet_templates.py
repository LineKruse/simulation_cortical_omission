import pickle
import numpy as np
import pandas as pd
from pathlib import Path

class TemplateBase:
    def __init__(self, labels, unit):
        self.labels = list(labels)
        self.unit = unit
        self._index = {label: i for i, label in enumerate(self.labels)}

    def _select(self, data, labels):
        if labels is None:
            return np.asarray(data)

        if isinstance(labels, str):
            labels = [labels]

        idx = [self._index[label] for label in labels if label in self._index]
        return np.asarray([data[i] for i in idx])

class HelmetTemplate(TemplateBase):
    def __init__(self, chan_ori, chan_pos, labels, unit):
        super().__init__(labels, unit)

        self.chan_ori = np.asarray(chan_ori)
        self.chan_pos = np.asarray(chan_pos)


        if len(self.chan_pos) != len(self.labels):
            raise ValueError("chan_pos and labels mismatch")

        if len(self.chan_ori) != len(self.labels):
            raise ValueError("chan_ori and labels mismatch")
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._index = {label: i for i, label in enumerate(self.labels)}
    def get_chs_pos(self, labels=None):
        return self._select(self.chan_pos, labels)

    def get_chs_ori(self, labels=None):
        return self._select(self.chan_ori, labels)

    def get_fid_pos(self, labels=None):
        return self._select(self.fid_pos, labels)

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "HelmetTemplate":
            return HelmetTemplate
        return super().find_class(module, name)

def load_helmet_template(file_path):
    with open(file_path, 'rb') as f:
        unpickler = CustomUnpickler(f)
        helmet_template = unpickler.load()
    return helmet_template




def generate_FL_helmet_template(path: Path):
    """
    Very important that all the depth measurements in the Alpha 1 Adjustable Helmet Sensor locations.xlsx is set to 52 when loading in.
    Otherwise the remaining functions, e.g. when creating the OPMSensorLayout based on the depth measurements will be wrong
    """


    df = pd.read_excel(path / "alpha1.xlsx")

    # Create a list to hold the orientation matrices
    orientation_matrices = []

    # Loop through each row and create a 3x3 matrix
    for index, row in df.iterrows():
        matrix = np.array([
            [row["ex_i"], row["ex_j"], row["ex_k"]],
            [row["ey_i"], row["ey_j"], row["ey_k"]],
            [row["ez_i"], row["ez_j"], row["ez_k"]]
        ])
        orientation_matrices.append(matrix)

    # Convert the list to a numpy array if you want a full 3D array
    orientation_matrices = np.array(orientation_matrices)


    positions = []
    # Loop through each row and create a 3x3 matrix
    for index, row in df.iterrows():
        vector = np.array([
            row["sensor_x"], row["sensor_y"], row["sensor_z"]
            #row["x cell"], row["y cell"], row["z cell"]
        ])
        positions.append(vector)

    positions = np.array(positions)

    FL_template = HelmetTemplate(
        chan_pos=positions,
        chan_ori=orientation_matrices,
        labels=[f"FL{i}" for i in range(1, len(positions)+1)],
        unit="m")

    with open(path / "FL_alpha1_helmet.pkl", 'wb') as file:
        pickle.dump(FL_template, file)



if __name__ == "__main__":
    path=Path(__file__).parent / "template"
    generate_FL_helmet_template(path)
