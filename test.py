"""

测试脚本
"""
import os
import copy
import collections
from time import time
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology
# from models.model1207.UNet import UNet
# from models.new_work_new.Cross_UNet import Cross_UNet
# from models.unet.UNet import UNet
# from models.model1207.ResUNet import ResUNet
# from models.model1207.VNet import VNet
# from models.model1207.Att_UNet import Att_UNet
# from models.model1207.SkipDenseNet3D import SkipDenseNet3D
# from models.model1207.Att_UNet import Att_UNet
# from models.model1207.Cross_UNet1209 import Cross_UNet3
# from models.model1207.Cross_UNet1211 import Cross_UNet4
# from models.model1207.Dense_UNet import Dense_UNet
# from models.model1207.ResUNet import ResUNet
# from models.model1207.Cross_UNet0310 import Cross_UNet7
# from models.model1207.UNet import UNet
# from models.model1207.Cross_UNet0312 import Cross_UNet9
# from models.model1207.Cross_LDU_UNet_sum0314_2 import Cross_LDU_UNet_sum
# from models.model1207.Cross_UNet0313_sum import Cross_UNet102_sum
# from models.model1207.Cross_UNet0312 import Cross_UNet9
# from models.model1207.DeepLabv3plus_DRLCTUNet import DeepLabv3plus_DRLCTUnet
# from models.model1207.Cross_DeepLabv3plus_DRLCTUNet import Cross_DeepLabv3plus_DRLCTUnet
# from models.model1207.DeepLabv3plus import DeepLabv3plus
# from models.model1207.DeepLabv3plus_UNet import DeepLabv3plus_UNet
# from models.model1207.DeepLabv3plus_ResNet3D import DeepLabv3plus_ResNet3D
# from models.model1207.UNet import UNet
# from models.model1207.DeepLabv3plus_DRLCTUNet import DeepLabv3plus_DRLCTUnet
# from models.model1207.UNet import UNet

from models.model1207.DeepLabv3plus_DRUNet_MS_concat import DeepLabv3plus_DRUNet_MS_concat
# from models.my_models.LCT_DR_UNet_new import LCT_DR_UNet_new
from models.model1207.LCT_DR_UNet_new import LCT_DR_UNet_new
from utils.metrics import Metirc
import parameter_test as para
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# 为了计算dice_global定义的两个变量
dice_intersection = 0.0
dice_union = 0.0

file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间

# 定义评价指标
coronary_score = collections.OrderedDict()
coronary_score['dice'] = []
coronary_score['jacard'] = []
coronary_score['voe'] = []
coronary_score['rvd'] = []
coronary_score['fnr'] = []
coronary_score['fpr'] = []
coronary_score['assd'] = []
coronary_score['rmsd'] = []
coronary_score['msd'] = []
coronary_score['recall'] = []
coronary_score['precision'] = []

# 定义网络并加载参数
# net = torch.nn.DataParallel(DeepLabv3plus_DRUNet_MS_concat(training=False)).cuda()#local
net = torch.nn.DataParallel(LCT_DR_UNet_new(training=False)).cuda()#local
# net = UNet(training=False).cuda()#服务器
#---------------1----------------
# net = torch.load(para.coronary_module_path, map_location='cuda:2')
#--------------------------------------------------
# net.load_state_dict(torch.load(para.coronary_module_path))
#---------------2----------------
state_dict = torch.load(para.coronary_module_path, map_location=torch.device('cuda'))#map_location='cuda:1'

new_state_dict = collections.OrderedDict()
for k,v in state_dict.items():
    name = 'module.' + k
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)#Missing...error
# net.load_state_dict(new_state_dict, strict=False)

net.eval()
#-------------------------------
#----------------3---------------

#-------------------------------

print(para.model_name)
for file_index, file in enumerate(os.listdir(para.test_coronary_ct_path)):

    start = time()

    file_name.append(file)

    # 将CT读入内存
    ct = sitk.ReadImage(os.path.join(para.test_coronary_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    # print(ct_array.shape) #(75, 512, 512)

    origin_shape = ct_array.shape

    # 将灰度值在阈值之外的截断掉
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # min max 归一化
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 500
    # ct_array = ct_array / 200

    # 对CT使用双三次算法进行插值，插值之后的array依然是int16
    ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3) #(,256,256)#ct_array = ndimage.zoom(ct_array, (1, para.down_scale, para.down_scale), order=3) #(,256,256)

    # 对slice过少的数据使用padding填充，如果切片数量小于48，要对切片就行边界填充，为(48,256,256)
    too_small = False
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        temp = np.ones((para.size, int(512 * para.down_scale), int(512 * para.down_scale))) * para.lower  # (48,256,256)
        temp[0: depth] = ct_array
        ct_array = temp
        too_small = True

    #  将原始CT影像分割成长度为48的一系列的块，如0~47, 48~95, 96~143, .....
    start_slice = 0
    end_slice = start_slice + para.size - 1 # 47
    # 用来统计原始CT影像中的每一个像素点被预测了几次
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
    # 用来存储每个像素点的预测值
    probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)


    with torch.no_grad():
        while end_slice < ct_array.shape[0]:

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()  # [0,48]
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)  # shape变为: (1, 1, 48, 256, 256)

            outputs = net(ct_tensor)
            #outputs = outputs[3]
            # print(outputs.shape) #torch.Size([1, 1, 48, 512, 512])

            count[start_slice: end_slice + 1] += 1
            # print(count)
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())
            # print(probability_map.shape) #(75, 512, 512)
            # exit()

            # 由于显存不足，这里直接保留ndarray数据，并在保存之后直接销毁计算图
            del outputs

            # 滑动窗口取样预测,
            start_slice += para.stride  # 12,24,36
            end_slice = start_slice + para.size - 1  # 59,71,83


        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            # outputs = outputs[3]

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs

        pred_seg = np.zeros_like(probability_map)  # 创建同shape的全0矩阵
        pred_seg[probability_map >= (para.threshold * count)] = 1

        if too_small:
            temp = np.zeros((depth, 512, 512), dtype=np.float32)
            temp += pred_seg[0: depth]
            pred_seg = temp

    # 将金标准读入内存
    seg = sitk.ReadImage(os.path.join(para.test_coronary_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
    pred_seg = pred_seg.astype(np.uint8)
    coronary_seg = copy.deepcopy(pred_seg)
    coronary_seg = measure.label(coronary_seg, 4)  # 把输入的整数数组进行连通域标记
    props = measure.regionprops(coronary_seg)  # 测量标记图像区域的属性
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index
    coronary_seg[coronary_seg != max_index] = 0
    coronary_seg[coronary_seg == max_index] = 1
    coronary_seg = coronary_seg.astype(np.bool)
    morphology.remove_small_holes(coronary_seg, para.maximum_hole, connectivity=2, in_place=True) # 去除孔洞connectivity=2

    coronary_seg = pred_seg.astype(np.uint8)
    # 计算分割评价指标
    coronary_metric = Metirc(seg_array, coronary_seg, ct.GetSpacing())

    coronary_score['dice'].append(coronary_metric.get_dice_coefficient()[0])
    coronary_score['jacard'].append(coronary_metric.get_jaccard_index())
    coronary_score['voe'].append(coronary_metric.get_VOE())
    coronary_score['rvd'].append(coronary_metric.get_RVD())
    coronary_score['fnr'].append(coronary_metric.get_FNR())
    coronary_score['fpr'].append(coronary_metric.get_FPR())
    coronary_score['assd'].append(coronary_metric.get_ASSD())
    coronary_score['rmsd'].append(coronary_metric.get_RMSD())
    coronary_score['msd'].append(coronary_metric.get_MSD())
    coronary_score['recall'].append(coronary_metric.get_recall())
    coronary_score['precision'].append(coronary_metric.get_precision())

    dice_intersection += coronary_metric.get_dice_coefficient()[1]
    dice_union += coronary_metric.get_dice_coefficient()[2]

    # 将预测的结果保存为nii数据
    pred_seg = sitk.GetImageFromArray(coronary_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    if not os.path.exists(para.pred_coronary_path):
        os.makedirs(para.pred_coronary_path)
    sitk.WriteImage(pred_seg, os.path.join(para.pred_coronary_path, file.replace('volume', 'pred')))

    speed = time() - start
    time_pre_case.append(speed)

    print(file_index, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')


# 将评价指标写入到exel中
coronary_data = pd.DataFrame(coronary_score, index=file_name)
coronary_data['time'] = time_pre_case

coronary_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(coronary_data.columns))
coronary_statistics.loc['mean'] = coronary_data.mean()
coronary_statistics.loc['std'] = coronary_data.std()
coronary_statistics.loc['min'] = coronary_data.min()
coronary_statistics.loc['max'] = coronary_data.max()

writer = pd.ExcelWriter('./result_pred_xlsx/bs3result_coronary_pred_'+para.model_name+'.xlsx')
coronary_data.to_excel(writer, 'coronary')
coronary_statistics.to_excel(writer, 'coronary_statistics')
writer.save()

# 打印dice global
print('dice global:', dice_intersection / dice_union)
