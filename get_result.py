import numpy as np
from glob import glob
np.set_printoptions(suppress=True)

# txt_list = glob('/HOME/HOME/Caixiang/GauHuman/result/*.txt')
# txt_list = glob('result/monocap_w_o_gaussion_rot_scale.txt')
#txt_list = glob('result/monocap_temp.txt')
txt_list = glob('result/temp.txt')
print(len(txt_list))

info = True
for txt_path in txt_list:
    try:
        print("==========")
        print(txt_path)
        # try:
        method_max = []
        file = open(txt_path, 'r')
        data = file.read()
        data = [i for i in data.split('\n')]
        metrics_np = []
        data_max = []
        for i in data:
            if len(i)<=1:continue
            # if len(i)<9:
            if "my_" in i:
                if len(metrics_np)>0:
                    data = np.array(metrics_np)
                    row_with_max_last_column = data[data[:, -1].argmax()]
                    if info:
                        print(row_with_max_last_column)
                    data_max.append(row_with_max_last_column)
                    metrics_np = []
            else:
                metrics = i.split(' ')
                for i in metrics:
                    if len(i)<2:metrics.remove(i)
                metrics = [float(i) for i in metrics]
                metrics_np.append(metrics)

        data = np.array(metrics_np)
        row_with_max_last_column = data[data[:, -1].argmin()]
        if info:
            print(row_with_max_last_column)
        data_max.append(row_with_max_last_column)

        data_max = np.array(data_max)
        max = np.sum(data_max,axis=0)/data_max.shape[0]
        method_max.append(max)
        print(max)
    except:
        print('jump over ',txt_path)
        
        
# info = True
# for txt_path in txt_list:
#     try:
#         print("==========")
#         print(txt_path)
#         # try:
#         method_max = []
#         file = open(txt_path, 'r')
#         data = file.read()
#         data = [i for i in data.split('\n')]
#         metrics_np = []
#         data_max = []
#         for i in data:
#             if len(i)<=1:continue
#             # if len(i)<9:
#             if "my_" in i:
#                 if len(metrics_np)>0:
#                     data = np.array(metrics_np)
#                     row_with_max_last_column = data[data[:, 1].argmax()]
#                     if info:
#                         print(row_with_max_last_column)
#                     data_max.append(row_with_max_last_column)
#                     metrics_np = []
#             else:
#                 metrics = i.split(' ')
#                 for i in metrics:
#                     if len(i)<2:metrics.remove(i)
#                 metrics = [float(i) for i in metrics]
#                 metrics_np.append(metrics)

#         data = np.array(metrics_np)
#         row_with_max_last_column = data[data[:, 1].argmax()]
#         if info:
#             print(row_with_max_last_column)
#         data_max.append(row_with_max_last_column)

#         data_max = np.array(data_max)
#         max = np.sum(data_max,axis=0)/data_max.shape[0]
#         method_max.append(max)
#         print(max)
#     except:
#         print('jump over ',txt_path)

# # result/monocap_w_o_gaussion_rot_scale.txt
