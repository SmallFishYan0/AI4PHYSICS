from data import load_efficiency_data
import numpy as np

efficiency_filepaths = ["efficiency_1500MeV.root", "efficiency_1200MeV.root", "efficiency_1000MeV.root", "efficiency_800MeV.root", "efficiency_700MeV.root", "efficiency_600MeV.root", "efficiency_500MeV.root"]

# input: list of efficiency file paths
# output: mae mse and baseline_data with baseline efficiency added
def calculate_baseline_efficiency(efficiency_filepaths):
    baseline_data = []
    for file in efficiency_filepaths:
        datas = load_efficiency_data(file_paths=[file])
        for data in datas:
            data['baseline_efficiency'] = 1 - np.prod(1 - data['efficiency_feature'])
        baseline_data.append(datas)
        count = 0
        mae = 0
        mse = 0
        for m_data in baseline_data:
            for data in m_data:
                mae += abs(data['baseline_efficiency'] - data['label'])
                mse += (data['baseline_efficiency'] - data['label'])**2
                count += 1

    
    return mae / count, mse / count, baseline_data


