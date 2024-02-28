import imageio
import glob

# path = glob.glob('/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_377_best_2/test/ours_2700/renders/*.png')

for name in ['377','386','387','392','393','394']:
    path = glob.glob(f'/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_temp/test/ours_2200/renders/*.png')
    # path = glob.glob(f'/home/zjlab1/workspace/Caixiang/GauHuman_ablation/output/zju_mocap_refine/my_{name}_temp/test/ours_3400/renders/*.png')
    path.sort()
    print(path)
    with imageio.get_writer(uri=f'{name}.gif', mode='I', duration=1/48) as writer:
        for i in path:
            writer.append_data(imageio.imread(i))
