import os
import sys
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

sys.path.append("D:/PythonProject/CRNN/src")
from src.dataset import MyDataSet
from src.models import  NUM_CLASSES, CRNN, CRNN_Better


# 配置参数
parser = argparse.ArgumentParser()

# 数据及结果路径配置
parser.add_argument("--trainRoot", default="E:/OCRData/MiniData", help="path to dataset",)
parser.add_argument("--valRoot", default="E:/OCRData/ch4_training_word_images_gt",help="path to dataset",)
parser.add_argument("--pretrained", default="", help="path to pretrained model (to continue training)")
parser.add_argument("--expr_dir", default="expr", help="Where to store samples and models")

# 数据集参数配置
parser.add_argument("--imgH", type=int, default=32, help="the height of the input image to network")
parser.add_argument("--imgW", type=int, default=100, help="the width of the input image to network")

# 硬件参数及超参数配置
parser.add_argument("--cuda", action="store_true", help="enables cuda")
parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
parser.add_argument("--nepoch", type=int, default=100, help="number of epochs to train for")
parser.add_argument("--saveInterval", type=int, default=5, help="number of epochs to save model")
parser.add_argument("--batchSize", type=int, default=32, help="input batch size")
parser.add_argument("--workers", type=int, default=1, help="number of data loading workers")
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')

# 优化器参数配置
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
opt = parser.parse_args()

# 创建输出文件夹
if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

# 设置随机种子
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, opt):
    epoch_loss = 0.0
    
    print('Epoch: {}/{} ...'.format(epoch + 1, opt.nepoch))
    for batch, (image, input_len, target, target_len) in enumerate(data_loader):
        image = image.unsqueeze(1) # 添加通道维度
        
        # print("image shape: {}".format(image.shape))   # batch_sizie*height*wdith -> batch_size*channels*height*wdith
        # print("target shape: {}".format(target.shape))  # batch_size -> batch_size*cls
        
        image = image.to(device)
        target = target.to(device)
        outputs = model(image)  # [B,N,C]
        outputs = outputs.permute([1, 0, 2])  # [N,B,C] input_legnth*batch_size*number_of_classes
        outputs = torch.log_softmax(outputs, dim=2) # 此处的log_softmax不要放在模型里，否则loss会为负

        # 修改输入张量，每个元素的长度等于模型输出序列的长度
        input_len = torch.IntTensor([outputs.shape[0]] * opt.batchSize)

        # print("outputs shape: {}".format(outputs.shape))
        # print("input_len: {}".format(input_len.shape))
        # print("target_len: {}".format(target_len.shape))
        loss = criterion(outputs, target, input_len, target_len)

        # CTCloss参考
        # https://blog.csdn.net/qq_41915623/article/details/125753277
        # https://vimsky.com/examples/usage/python-torch.nn.CTCLoss-pt.html
        
        # 梯度更新
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 当前轮的loss
        epoch_loss += loss.item() * image.size(0)
        # print('batch: {}/{} batch loss: {:03f}'.format(batch, epoch + 1, epoch_loss))

        if np.isnan(loss.item()):
            print(target, input_len, target_len)

    epoch_loss = epoch_loss / len(data_loader.dataset)
    # 打印日志,保存权重
    print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, opt.nepoch, epoch_loss))
    return epoch_loss

def train():
    # 选择设备
    device = torch.device("cuda" if opt.cuda == "cuda" and torch.cuda.is_available() else "cpu")

    # dataloader
    dataset = MyDataSet(opt.trainRoot, mode="train")
    data_loader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=True)

    # 模型
    # model = CRNN(num_classes=NUM_CLASSES)
    model = CRNN_Better(32, 1, NUM_CLASSES, 256)
    criterion = torch.nn.CTCLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    # 存在预训练模型则进行加载
    if opt.pretrained != "":
        print('loading pretrained model from %s' % opt.pretrained)
        model.load_state_dict(torch.load(opt.pretrained))
    else:
        # 模型权重初始化
        model.apply(weights_init)

    # setup optimizer 从头训练
    if opt.adam:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(model.parameters())
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)


    # 学习率衰减
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                               gamma=0.65)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=[2, 4, 6, 8, 10],
    #                                            gamma=0.65)

    # train
    model.train()

    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为字符串
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S\n")
    log_file = open(os.path.join(opt.expr_dir, "log.txt"), "a+")
    log_file.write(time_str)

    for epoch in range(opt.nepoch):
        # 训练
        loss = train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, opt)

        scheduler.step()
        # 保存训练日志
        log_file.write("Epoch: {}/{} loss: {:03f}\n".format(epoch + 1, opt.nepoch, loss))
        log_file.flush()

        # do checkpointing
        if epoch % opt.saveInterval == 0:
            torch.save(
                model.state_dict(), "{0}/netCRNN.pth".format(opt.expr_dir))
    
    log_file.close()   
    

if __name__ == "__main__":
    print("opt.trainRoot: {}".format(opt.trainRoot))
    train()
    
