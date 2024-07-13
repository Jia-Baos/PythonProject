import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class MyDataSet(Dataset):
    """
    dataset_dir
    ----Annotations
    --------000000.xml
    --------000001.xml
                ~
    ----Images
    --------000000.jpg
    --------000001.jpg
                ~
    ----ImageSets
    --------val.txt
    --------text.txt
    --------train.txt
                ~
    ----Labels
    --------000000.txt
    --------000001.txt
                ~
    """
    def __init__(self, dataset_dir, mode="train", trans=None):
        self.data_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, "Images")
        self.label_dir = os.path.join(dataset_dir, "Labels")
        self.imagesets_dir = os.path.join(dataset_dir, "ImageSets")
        self.mode = mode

        # img_list存的只是图片的名字
        self.img_list = []
        if mode == "train":
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        elif mode == "val":
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        else:
            img_list_txt = os.path.join(self.imagesets_dir, mode + '.txt')
            with open(img_list_txt, 'r') as f:
                for line in f.readlines():
                    self.img_list.append(line.strip('\n'))

        self.trans = trans

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.img_list[item] + '.png')
        image = Image.open(img_path)
        label_file = os.path.join(self.label_dir, self.img_list[item] + '.txt')
        f = open(label_file, 'r')
        label = f.readline()
        label = int(label[0])
        if self.trans is None:
            trans = transforms.Compose([
                transforms.Resize((28, 28)),
                # convert a PIL image to tensor(H * W * C) in range[0, 255]
                # to a torch.Tensor(C * H * W) in the range[0.0, 1.0]
                transforms.ToTensor()
            ])
        else:
            trans = self.trans

        image = trans(image)
        label = torch.tensor(label)
        return image, label


if __name__ == '__main__':
    data_dir = "D:\\PythonProject\\LeNet\\newdata"
    dataset = MyDataSet(data_dir)
    # DataLoader要求输入图片大小一致
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0, drop_last=False)
    for data in dataloader:
        # imgs：
        # 四维张量[N, C, H, W]
        # N：每个batch图片的数量
        # C：通道数（channels），即in_channels
        # H：图片的高
        # W：图片的宽
        imgs, targets = data
        print(imgs.shape)
        print(targets)
