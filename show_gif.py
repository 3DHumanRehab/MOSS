import imageio
import glob


size = 0
# for name in ['377','386','387','392','393','394']:
#     path = glob.glob(f'/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_temp/test/ours_2200/gt/*.png')
#     # path = glob.glob(f'/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_temp/test/ours_3400/gt/*.png')
#     path.sort()
#     print(path)
#     with imageio.get_writer(uri=f'{name}_GT.gif', mode='I', duration=1/48) as writer:
#         for i in path:
#             writer.append_data(imageio.imread(i))



name_list = ['olek_images0812',"lan_images620_1300", "marc_images35000_36200","vlad_images1011"]
# name_list = ["marc_images35000_36200"]
log_name_list = ['monocap_temp']
# iteration_list_list = [[2200,3400,3200,2700]]
iteration_list_list = [3600,2000,2000,2200]

for name,iter in zip(name_list,iteration_list_list):
    path = glob.glob(f'/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_monocap_temp/test/ours_{iter}/gt/*.png')
    # path = glob.glob(f'/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_temp/test/ours_3400/renders/*.png')
    path.sort()
    print(path)
    with imageio.get_writer(uri=f'{name}_GT.gif', mode='I', duration=1/48) as writer:
        for i in path:
            writer.append_data(imageio.imread(i))



# import imageio
# import glob

# path = glob.glob('/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_377_best_2/test/ours_2700/renders/*.png')

# size = 100
# for name in ['377','386','387','392','393','394']:
#     path = f'cp -r /home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_temp/test/ours_2200/renders {name}'
#     print(path)



# name_list = ['olek_images0812',"lan_images620_1300", "marc_images35000_36200","vlad_images1011"]
# # name_list = ["marc_images35000_36200"]
# log_name_list = ['monocap_temp']
# # iteration_list_list = [[2200,3400,3200,2700]]
# iteration_list_list = [3600,2000,2000,2200]

# for name,iter in zip(name_list,iteration_list_list):
#     path =  f'cp -r /home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_monocap_temp/test/ours_{iter}/renders {name}'
#     print(path)


