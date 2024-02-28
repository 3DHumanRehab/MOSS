import os
import cv2
import glob
import tqdm
import matplotlib.pyplot as plt
# 获取当前工作目录
Path = '/home/zjlab1/workspace/Caixiang/GauHuman_ablation/methods/'
# log_name_list = ['Gauhuman','InstantNVR','neuralbody','deform','Ours']
# log_name_list = ['Ours','Gauhuman','InstantNVR','AnimateNeRF','NeuralBody']
# log_name_list = ['Ours','Gauhuman','InstantNVR']
log_name_list = ['Ours','Gauhuman','InstantNVR']


# dataset = ['377','386','387','392','393','394']
dataset = ['marc']

for data in dataset:
    save_path = f'./vision/{data}_compare/'
    get_gt = True
    pred_list =[]
    for name in log_name_list: 
        current_directory = Path + name
        print("========")
        print(name)
        for item in os.listdir(current_directory):
            item_path = os.path.join(current_directory, item)
            if os.path.isdir(item_path) and data in item:
                item_path = os.path.join(current_directory, item)
                if "my" in  item_path:  # Ours,Gauhuman
                    print(name)
                    item_path = os.path.join(item_path, 'test')
                    for item in os.listdir(item_path):
                        item_path  = os.path.join(item_path,item)
                        item_path  = os.path.join(item_path,'renders/*.png')
                        if get_gt: 
                            temp = glob.glob(item_path.replace('renders','gt'))
                            temp.sort()
                            pred_list.append(temp)
                            get_gt = False
                        img_list = glob.glob(item_path)
                        img_list.sort()
                        pred_list.append(img_list)
                        print(len(img_list))

                elif 'aninerf' in item_path or 'inb' in item_path:
                    print(name)
                    item_path  = os.path.join(item_path,'comparison/*.png')
                    img_list = glob.glob(item_path)
                    img_list.sort()
                    img_list = [i for i in img_list if 'gt' not in i]
                    print(len(img_list))
                    pred_list.append(img_list)

                elif "xyzc" in  item_path:  # Ours,Gauhuman
                    print(name)
                    # import pdb
                    # pdb.set_trace()
                    # item_path = os.path.join(item_path, f'xyzc_{data}')
                    for item in os.listdir(item_path):
                        item_path  = os.path.join(item_path,'frame_0000/*.png')
                        img_list = glob.glob(item_path)
                        img_list.sort()
                        pred_list.append(img_list[3:])
                        print(len(img_list))

    txt_length = len(pred_list)
    temp_log_name_list = ['GT']+log_name_list
    for n in tqdm.tqdm(range(31,374)):
        bound_mask = pred_list[1][n].replace('renders','depth')
        bound_mask = cv2.imread(bound_mask)
        img = cv2.cvtColor(bound_mask, cv2.COLOR_BGR2GRAY)
        _, bound_mask = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(bound_mask)
        images = [cv2.imread(pred_list[i][n])[y-5:y + h+25, x-5:x + w+25,:][:,:,::-1] for i in range(0, txt_length)]
        # images = [cv2.imread(pred_list[i][n])[250:-205, 250:-205,:][:,:,::-1] for i in range(0, txt_length)]
        # images = [cv2.imread(pred_list[i][n])[:,:,::-1] for i in range(0, txt_length)]
        # import pdb
        # pdb.set_trace()
        # imgs = np.concatenate(images,axis=1)

        fig, axs = plt.subplots(1, len(pred_list), figsize=(15, 15))
        idx = 0
        for ax,log_name in zip(axs,temp_log_name_list):
            ax.imshow(images[idx])
            ax.axis('off')  # Hide axes
            ax.set_title(log_name.replace('monocap_w_o','w/o'), fontsize=10)  # Number below the image
            idx +=1
        # plt.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
        # if n==3:
        #     break
        # cv2.imwrite(save_path+str(n)+'.jpg',imgs)
        plt.savefig(save_path+str(n)+'.pdf',dpi=900)
        # plt.savefig(save_path+str(n)+'.png',dpi=300)
        # break 
    break


