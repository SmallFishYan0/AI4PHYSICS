import uproot
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset


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
    '''构造单个文件的训练数据（不包含单层效率特征）'''
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
    datas = []
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                binX = get_bin_index(x_axis, x)
                binY = get_bin_index(y_axis, y)
                binZ = get_bin_index(z_axis, z)
                x, y, z = int(x), int(y), int(z)
                positions = np.arange(x, x + y * z, z)
                if any(positions >= 61):
                    continue
                datas.append({"pos_feature": positions, "momentum": momentum,
                             "label": values[binX-1, binY-1, binZ-1]})
    return datas


def find_efficiency(one_layer_path, datas):
    '''查询给定探测器结构的单层效率特征'''
    f = uproot.open(one_layer_path)
    h_one_layer = f['h_out']
    values_one_layer = h_one_layer.values()
    x_axis_one_layer = h_one_layer.axes[0].edges()
    x_axis_one_layer = np.array([int(i) + 25 for i in x_axis_one_layer][:-1])
    y_axis_one_layer = np.array(h_one_layer.axes[1].edges()[:-1])

    final_datas = []
    for data in datas:
        m = data["momentum"]
        pos_feature = data["pos_feature"]
        efficiency_feature = values_one_layer[x_axis_one_layer == m].squeeze()
        efficiency_feature = efficiency_feature[pos_feature]
        final_datas.append({
            "pos_feature": data["pos_feature"], 
            "efficiency_feature": efficiency_feature, 
            "momentum": data["momentum"], 
            "label": data["label"]
        })

    return final_datas


def load_efficiency_data(file_paths: list):
    one_layer_path = "raw_data/oneLayer_data/efficiency_filtered.root"

    datas = []
    for file in file_paths:
        if "efficiency" in file:
            datas_tmp = process_efficiency_data(os.path.join("raw_data", file))
            datas_tmp = find_efficiency(one_layer_path, datas_tmp)
            datas.extend(datas_tmp)

    return datas


class DetectorDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        # 创建一个61维的mask，标记哪些位置有探测器
        pos_mask = torch.zeros(61, dtype=torch.bool)
        pos_mask[item['pos_feature']] = True
        
        # 创建效率特征向量，没有探测器的位置填充0
        efficiency_vec = torch.zeros(61, dtype=torch.float32)
        efficiency_vec[item['pos_feature']] = torch.tensor(
            item['efficiency_feature'], dtype=torch.float32
        )
        
        # 动量归一化到 [0, 1] 范围
        momentum = torch.tensor([item['momentum'] / 1500.0], dtype=torch.float32)
        
        label = torch.tensor([item['label']], dtype=torch.float32)
        
        return {
            'pos_mask': pos_mask,
            'efficiency': efficiency_vec,
            'momentum': momentum,
            'label': label
        }
