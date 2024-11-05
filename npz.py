import glob
import cv2
import numpy as np
import os

from scipy.ndimage import zoom


def npz(im, la, s):
    images_path = im
    labels_path = la

    images = os.listdir(images_path)
    labels = os.listdir(labels_path)
    for (img_name, lab_name) in zip(images, labels):
        image_path = os.path.join(images_path, img_name)
        label_path = os.path.join(labels_path, lab_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		# 标签由三通道转换为单通道
        label = cv2.imread(label_path, flags=0)

        x, y, _ = image.shape
        image = zoom(image, (256 / x, 256 / y, 1), order=3)  # why not 3?
        label = zoom(label, (256 / x, 256 / y), order=0)

        label[label < 76] = 0
        label[label == 76]= 2
        label[label > 76]= 1
        # 保存npz文件
        np.savez(s + img_name[:-4]+".npz",image=image,label=label)

# npz('./img_datas_RITEyes/train/image/', './img_datas_RITEyes/train/label_gray/', './data/RITEyes/train_npz/')
# npz('./img_datas/val/image/', './img_datas/val/label/', './data/REFUGE/val_npz/')
npz('./Independent_test_data/test/image/', './Independent_test_data/test/label_gray/', './data/Independent_test_data/test_vol_h5/')


# def npz():
#     #原图像路径
#     path = r'G:\dataset\Segmentation\LungSegmentation\npz\images\*.png'
#     #项目中存放训练所用的npz文件路径
#     path2 = r'G:\dataset\Unet\TransUnet-ori\data\Synapse\train_npz\\'
#     for i,img_path in enumerate(glob.glob(path)):
#     	#读入图像
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#         #读入标签
#         label_path = img_path.replace('images','labels')
#         label = cv2.imread(label_path,flags=0)
#         #将非目标像素设置为0
#         label[label!=255]=0
#         #将目标像素设置为1
#         label[label==255]=1
# 		#保存npz
#         np.savez(path2+str(i),image=image,label=label)
#         print('------------',i)
#
#     # 加载npz文件
#     # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
#     # image, label = data['image'], data['label']
#
#     print('ok')
