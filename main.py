# from models.model1207.ResUNet import inita,ResUNet
# from models.model1207.DR_UNet import DR_UNet
# from models.model1207.unetr import UNETR
# from models.model1207.vnet import VNet
# from models.model1207.Att_UNet import Att_UNet
# from models.model1207.LCT_UNet import LCT_UNet
# from models.model1207.Att_UNet import Att_UNet
# from models.model1207.SkipDenseNet3D import SkipDenseNet3D
# from models.model1207.Cross_UNet1208 import inita,Cross_UNet
# from loss.loss_function import FocalLoss
# from models.model1207.UNETR import UNETR
# from utils.trainer_cross import trainer
# from models.model1207.Dense_UNet import Dense_UNet
# from models.model1207.LCT_DR_UNet_new import LCT_DR_UNet_new
# from models.model1207.Cross_LDU_UNet_sum0314_2 import inita,Cross_LDU_UNet_sum
# from models.model1207.Cross_UNet0313_2 import Cross_UNet102
# from models.model1207.UNet import UNet
# from models.model1207.Cross_LDU_UNet_cat0314 import Cross_LDU_UNet_cat
# from models.model1207.Cross_UNet0313_sum import inita,Cross_UNet102_sum
# from models.model1207.Cross_UNet0312 import inita,Cross_UNet9
# from models.model1207.Cross_UNet0313_cat import inita,Cross_UNet102_cat
# from models.model1207.DeepLabv3plus import inita,DeepLabv3plus
# from models.model1207.DeepLabv3plus_ResNet3D import DeepLabv3plus_ResNet3D
# from models.model1207.DeepLabv3plus_UNet import DeepLabv3plus_UNet
# from models.model1207.Cross_DeepLabv3plus_UNet import inita,Cross_DeepLabv3plus_UNet
# from models.model1207.DeepLabv3plus_DRLCTUNet import DeepLabv3plus_DRLCTUnet
# from models.model1207.Cross_DeepLabv3plus_DRLCTUNet3 import Cross_DeepLabv3plus_DRLCTUnet
# from models.model1207.UNet import UNet
# from models.model1207.UNet import inita,UNet
# from models.model1207.DeepLabv3 import inita,DeepLabv3
# from models.model1207.UNetPlusPlus import inita,UNetPlusPlus
# from models.model1207.DeepLabv3plus_UNet import DeepLabv3plus_UNet
# from models.model1207.DeepLabv3plus_DRLCTUNet_add import inita,DeepLabv3plus_DRLCTUNet_add
# from models.model1207.DeepLabv3plus_UNet_MS import inita,DeepLabv3plus_UNet_MS
# from models.other_models.DRUNet_yxf import inita, DRUNet_yxf
# from models.other_models.KiUNet_min import  KiUNet_min
# from models.other_models.SegNet import SegNet
# from models.other_models.DRUNet_yxf import DRUNet_yxf
# from models.model1207.DeepLabv3plus_DRUNet_MS import inita,DeepLabv3plus_DRUNet_MS
# from models.model1207.DeepLabv3plus_DRUNet_MS import inita,DeepLabv3plus_DRUNet_MS
# from models.model1207.DeepLabv3plus_LCTUNet import inita,DeepLabv3plus_LCTUNet
# from models.model1207.DeepLabv3plus_DRUNet import inita,DeepLabv3plus_DRUNet
# from models.model1207.DeepLabv3plus_LCTUNet_concat import  inita,DeepLabv3plus_LCTUNet_concat
# from models.model1207.DeepLabv3plus_UNet_MS_concat import inita,DeepLabv3plus_UNet_MS_concat
# from models.model1207.DeepLabv3plus_DRUNet_MS_concat import inita,DeepLabv3plus_DRUNet_MS_concat
# from models.model1207.DeepLabv3plus_DRUNet_MS2_concat import inita, DeepLabv3plus_DRUNet_MS2_concat
from models.model1207.LCT_DR_UNet_new import inita,LCT_DR_UNet_new
# from models.model1207.UNet import inita, UNet
from utils.trainer import trainer
import torch
from dataset.dataset import Dataset
from torch.utils.data import Dataset as dataset, DataLoader
from loss.Tversky import TverskyLoss, DiceLoss, FocalLoss
import parameter as para
import torch.backends.cudnn as cudnn
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


if __name__ == '__main__':
    device = para.device
    cudnn.benchmark = para.cudnn_benchmark
    # SEED = 3047  # 1022#2022
    # torch.cuda.manual_seed_all(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    # model = torch.nn.DataParallel(ResUNet(training=True)).cuda()#.to(device)#改进失败
    model = LCT_DR_UNet_new(training=True).to(device)#LCT_new_DR_UNet_new(training=True).to(device)
    model.apply(inita)
    Resume = para.Resume
    coronary_module_path=para.coronary_module_path
    # model =UNet(training=True).to(device)
    # model.apply(init)
    # model = UNETR(img_shape=(16, 512,512), input_dim=1, output_dim=1, embed_dim=768, patch_size=16, num_heads=16, dropout=0.1, training=True).to(device)

    learning_rate = para.learning_rate
    weight_decay = para.weight_decay
    max_epochs = para.max_epochs
    batch_size = para.batch_size
    learning_rate_decay = para.learning_rate_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, para.learning_rate_decay)
    loss = DiceLoss().to(device)
    # lossb = InfoNCELoss().to(device)
    lossb = TverskyLoss().to(device)
    # loss = TV_dice_loss().to(device)
    # loss = AsymmetricUnifiedFocalLoss().to(device)
    # loss =FocalLoss().to(device)

    dataset_path = para.dataset_coronary_path
    train_ds = Dataset(dataset_path, model='train')
    train_dl = DataLoader(train_ds, para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)
    val_ds = Dataset(dataset_path, model='val')
    val_dl = DataLoader(val_ds, para.batch_size, shuffle=True, num_workers=para.num_workers, pin_memory=para.pin_memory)

    alpha = para.alpha
    checkpoint_dir = './checkpoints/lits17_test/bs3/' + para.model_name
    comments = 'lits17_test_'
    verbose_train = 1
    # verbose_val = 25
    ckpt_frequency = 3
    ckpt_alpha = 40
    ch_in = para.ch_in

    # trainer = trainer(model, optimizer, max_epochs, lr_scheduler, loss, comments, train_dl, val_dl, alpha, ckpt_frequency, ckpt_alpha, verbose_train, verbose_val, checkpoints_dir, device)
    #带lossb
    trainer = trainer(model,Resume, coronary_module_path,optimizer, max_epochs, lr_scheduler, loss,lossb,comments, train_dl, val_dl, alpha, ckpt_frequency, ckpt_alpha, verbose_train, checkpoint_dir, ch_in, device)
    # trainer = trainer(model, Resume, coronary_module_path, optimizer, max_epochs, lr_scheduler, loss, comments,train_dl, val_dl, alpha, ckpt_frequency, ckpt_alpha, verbose_train, checkpoints_dir, ch_in,device)
    trainer.train()

