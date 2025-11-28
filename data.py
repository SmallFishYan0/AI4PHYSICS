import uproot
import numpy as np
import os
import re


def get_bin_index(edges, value):
    return np.searchsorted(edges, value, side="right")


def filter_values_3d(values):
    '''对训练数据进行滤波'''
    values_out = values.copy()
    nx, ny, nz = values.shape

    for j in range(ny):
        h = values[:, j, :]
        h_out = h.copy()
        for i in range(nx):
            for k in range(nz):
                if h_out[i, k] == 0:
                    up = h_out[i, k-1] if k-1 >= 0 else h_out[i, k]
                    dn = h_out[i, k+1] if k+1 < nz else h_out[i, k]
                    if up + dn == up or up + dn == dn:
                        h_out[i, k] = up + dn
                    else:
                        h_out[i, k] = 0.5 * (up + dn)

        h_out2 = h_out.copy()
        for i in range(1, nx-1):
            for k in range(1, nz-1):
                h_out2[i, k] = (
                    h_out[i, k-1] +
                    h_out[i, k+1] +
                    2 * h_out[i, k]
                ) / 4.0

        values_out[:, j, :] = h_out2

    return values_out


def process_efficiency_data(filepath):
    m = re.search(r'(\d+)MeV', os.path.basename(filepath))
    momentum = int(m.group(1)) if m else 0
    f = uproot.open(filepath)
    h = f['h3']
    values = h.values()
    values = filter_values_3d(values)
    N = values.shape[0]
    x_axis = h.axes[0].edges()[:-1]
    y_axis = h.axes[1].edges()[1:4]
    z_axis = h.axes[2].edges()[:-1]
    pos_features = []
    labels = []
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                binX = get_bin_index(x_axis, x)
                binY = get_bin_index(y_axis, y)
                binZ = get_bin_index(z_axis, z)
                labels.append(values[binX-1, binY-1, binZ-1])
                x = int(x)
                y = int(y)
                z = int(z)
                vec = np.zeros(61, dtype=np.float32)
                positions = [x + i * z for i in range(y)]
                for p in positions:
                    if 0 <= p < 61:
                        vec[p] = 1.0
                full_vec = np.concatenate([vec, [momentum]])
                pos_features.append(full_vec)
    return pos_features, labels


def find_efficiency(one_layer_path, pos_features):
    f = uproot.open(one_layer_path)
    h_one_layer = f['h_out']
    values_one_layer = h_one_layer.values()
    x_axis_one_layer = h_one_layer.axes[0].edges()
    x_axis_one_layer = np.array([int(i) + 25 for i in x_axis_one_layer][:-1])
    y_axis_one_layer = np.array(h_one_layer.axes[1].edges()[:-1])
    efficiency_features = []
    for feature in pos_features:
        m = int(feature[-1])
        positions = np.where(feature[:-1] == 1.0)[0]
        feature[:-1] = values_one_layer[x_axis_one_layer == m].squeeze()
        filtered_feature = np.zeros_like(feature[:-1])
        filtered_feature[positions] = feature[:-1][positions]
        efficiency_features.append(np.concatenate([filtered_feature, [m]]))

    return efficiency_features



def load_efficiency_data():
    efficiency_filepath = ["efficiency_500MeV.root", "efficiency_600MeV.root", "efficiency_700MeV.root", "efficiency_800MeV.root", "efficiency_1000MeV.root", "efficiency_1200MeV.root", "efficiency_1500MeV.root"]
    one_layer_path = "raw_data/oneLayer_data/efficiency_filtered.root"
    features = []
    labels = []
    for file in efficiency_filepath:
        feature, label = process_efficiency_data("raw_data/" + file)
        feature = find_efficiency(one_layer_path, feature)
        features.extend(feature)
        labels.extend(label)

    return np.array(features), np.array(labels)


