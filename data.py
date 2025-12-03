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


def process_MeV_data(filepath):
    '''从efficency_xxxMeV.root文件中提取训练数据：结构、动量和真值'''
    m = re.search(r'(\d+)MeV', os.path.basename(filepath))
    momentum = int(m.group(1)) if m else 0
    f = uproot.open(filepath)
    h = f['h3']
    values = h.values()
    values = filter_values_3d(values)
    N = values.shape[0]
    x_axis = h.axes[0].edges()[:-1]
    y_axis = h.axes[1].edges()[:-1]
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


def process_inout_data(filepath, inner=True):
    '''从inner/outer文件中提取训练数据：结构、动量和真值'''
    f = uproot.open(filepath)
    h = f["colz"]
    values = h.values()
    x_axis = h.axes[0].edges()
    x_axis = (x_axis[:-1] + x_axis[1:]) / 2
    y_axis = h.axes[1].edges()
    y_axis = y_axis[:-1]
    datas = []
    for momentum in x_axis:
        for y in y_axis:
            binX = get_bin_index(x_axis, momentum)
            binY = get_bin_index(y_axis, y)
            if inner:
                positions = np.arange(0, int(y))
            else:
                positions = np.arange(int(y), 61)
            if any(positions >= 61):
                continue
            datas.append({"pos_feature": positions, "momentum": momentum,
                            "label": values[binX-1, binY-1]})
    return datas


def find_single_efficiency(one_layer_path, pos_feature, momentum):
    '''给定探测器结构和动量，查询单层效率特征'''
    f = uproot.open(one_layer_path)
    h_one_layer = f['h_out']
    values_one_layer = h_one_layer.values()
    x_axis_one_layer = h_one_layer.axes[0].edges()
    x_axis_one_layer = np.array(x_axis_one_layer)
    x_axis_one_layer = (x_axis_one_layer[:-1] + x_axis_one_layer[1:]) / 2
    y_axis_one_layer = np.array(h_one_layer.axes[1].edges()[:-1])

    efficiency_feature = values_one_layer[x_axis_one_layer == momentum].squeeze()
    efficiency_feature = efficiency_feature[pos_feature]
    return efficiency_feature


def load_efficiency_data(file_path: list, onelayer_data_dir: str):
    '''提取给定root文件中的训练数据'''
    one_layer_path = os.path.join(onelayer_data_dir, "efficiency_filtered.root")

    if "MeV" in file_path:
        datas = process_MeV_data(file_path)
    else:
        if "inner" in file_path:
            datas = process_inout_data(file_path, inner=True)
        else:
            datas = process_inout_data(file_path, inner=False)

    for data in datas:
        pos_feature = data['pos_feature']
        momentum = data['momentum']
        data['efficiency_feature'] = find_single_efficiency(
            one_layer_path, pos_feature, momentum
        )
        
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