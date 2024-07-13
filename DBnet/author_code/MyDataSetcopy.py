import os
import cv2
import numpy as np
from Myutils import shrink, make_threshold_map
from torch.utils.data import Dataset, DataLoader

data_path = "/data/train_data"


# 数据读取
class MyDataset(Dataset):
    def __init__(self, base_path=data_path):
        imgs = []
        labels = []
        img_dir = os.path.join(base_path, 'image')
        label_dir = os.path.join(base_path, 'label')

        img_list = os.listdir(img_dir)
        for i in range(len(img_list)):
            img_path = os.path.join(img_dir, img_list[i])
            label_path = os.path.join(label_dir, img_list[i].replace('png', 'txt'))

            imgs.append(img_path)
            labels.append(label_path)

        self.imgs_path = imgs
        self.labels_path = labels

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label_path = self.labels_path[index]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        with open(label_path, encoding='utf-8') as f:
            # 列表，txt文件中的一行构成其中的一个元素
            data = f.readlines()

        # 宽高调整为32的倍数
        height, width, _ = img.shape
        new_height = int(height / 32) * 32
        scale_y = new_height / height
        new_width = int(width / 32) * 32
        scale_x = new_width / width
        img = cv2.resize(img, (new_width, new_height))

        gt_boxes_all = []
        # 把列表中的ground truth框的坐标提取出来
        for i in range(len(data)):
            # 以空格为间隔将坐标分割出来
            gt_data = data[i].strip().split()
            x_list = []
            x_list.append(int(int(gt_data[0]) * scale_x))
            x_list.append(int(int(gt_data[2]) * scale_x))
            x_list.append(int(int(gt_data[4]) * scale_x))
            x_list.append(int(int(gt_data[6]) * scale_x))
            y_list = []
            y_list.append(int(int(gt_data[1]) * scale_y))
            y_list.append(int(int(gt_data[3]) * scale_y))
            y_list.append(int(int(gt_data[5]) * scale_y))
            y_list.append(int(int(gt_data[7]) * scale_y))

            gt_boxes = []
            # 将一个点的横、纵坐标封装成一个数组
            for j in range(len(x_list)):
                gt_boxes.append([x_list[j], y_list[j]])
            gt_boxes = np.array(gt_boxes, dtype=np.int32)
            # dimension = 3
            # round truth box, ground truth box's point, ground truth box's point's coordination
            gt_boxes_all.append(gt_boxes)

        probability_map = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
        # 遍历每一个候选框
        for gt_boxes in gt_boxes_all:
            poly = shrink(gt_boxes, 0.4)
            # 填充任意形状的图形，这里还要再次恢复ploy的维度（3）
            cv2.fillPoly(probability_map, [poly], (1.0))

        threshold_map = make_threshold_map(img=img, text_polys=gt_boxes_all)

        img = np.transpose(img, (2, 0, 1))
        img = np.array(img / 255, dtype=np.float32)
        probability_map = np.expand_dims(probability_map, axis=0)
        threshold_map = np.expand_dims(threshold_map, axis=0)
        return img, probability_map, threshold_map

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    for real_data in dataloader:
        real_img, real_probability_map, real_threshold_map = real_data
        print(real_img.size())
        print(real_probability_map.size())
        print(real_threshold_map.size())
