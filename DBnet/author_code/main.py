import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from MyNet import MyModel
from MyDataSet import MyDataSet
from Myutils import shrink_out


def train(model, train_loader, optimizer, epoch):
    model.train()
    all_step = 0
    for i in range(epoch):
        for step, data in enumerate(train_loader):
            all_step = all_step + 1
            img, shrink_label, threshold_label = data[0].to(torch.device('cuda')), data[1].to(torch.device('cuda')), data[2].to(torch.device('cuda'))
            shrink_pre, threshold_pre, binary_pre = model(img)
            loss_shrink_map = nn.BCELoss()(shrink_pre, shrink_label)
            loss_threshold_map = nn.L1Loss()(threshold_pre, threshold_label)
            loss_binary_map = nn.BCELoss()(binary_pre, shrink_label)
            loss = loss_shrink_map + loss_binary_map + 10 * loss_threshold_map
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                print('epoch:', i + 1, 'step:', step + 1, 'loss:', loss)
    torch.save(model.state_dict(), './model/DBnet_pytorch.pth')


def inference(model, test_loader):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            # img = data[0].to(torch.device('cuda'))
            img = data[0]
            shrink_pre, threshold_pre, binary_pre = model(img)
            # 通过索引batch获取图片
            img = img.cpu().numpy()[0]
            # 重新排列RGB通道
            img = np.transpose(img, (1, 2, 0))
            img = np.array(img * 255, dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pre = binary_pre.cpu().numpy()[0][0]
            pre[pre > 0.5] = 1
            pre[pre < 1] = 0
            pre = (pre * 255).astype(np.uint8)
            contours, _ = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            pre_boxes = []
            for i in range(len(contours)):
                contour = contours[i].squeeze(1)
                contour_perimeter = cv2.arcLength(contour, True)
                # 过小的可能是噪点，删除
                if contour_perimeter > 10:
                    bounding_box = cv2.minAreaRect(contour)
                    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
                    if points[1][1] > points[0][1]:
                        index_1, index_4 = 0, 1
                    else:
                        index_1, index_4 = 1, 0
                    if points[3][1] > points[2][1]:
                        index_2, index_3 = 2, 3
                    else:
                        index_2, index_3 = 3, 2
                    points = [points[index_1], points[index_2], points[index_3], points[index_4]]
                    points = np.array(points)
                    box = shrink_out(points, rate=2.0)
                    bounding_box2 = cv2.minAreaRect(box)
                    points = sorted(list(cv2.boxPoints(bounding_box2)), key=lambda x: x[0])
                    if points[1][1] > points[0][1]:
                        index_1, index_4 = 0, 1
                    else:
                        index_1, index_4 = 1, 0
                    if points[3][1] > points[2][1]:
                        index_2, index_3 = 2, 3
                    else:
                        index_2, index_3 = 3, 2
                    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
                    box = np.array(box).astype(np.int32)
                    pre_boxes.append(box)
            for i in range(len(pre_boxes)):
                box = pre_boxes[i]
                for j in range(len(box)):
                    cv2.line(img, (box[j][0], box[j][1]), (box[(j + 1) % 4][0], box[(j + 1) % 4][1]), (0, 0, 255), 2)
            cv2.imwrite('../result/result.jpg', img)


if __name__ == '__main__':
    # train
    # model = Model().to(torch.device('cuda'))
    # optimizer = optim.Adam(model.parameters())
    # train_data = MyDataset(base_path='./data/train_data')
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    # train(model, train_loader, optimizer, 10)

    # inference
    model = Model()
    # model = Model().to(torch.device('cuda'))
    # model.load_state_dict(torch.load('./model/DBnet_pytorch.pth'))
    test_data = MyDataset(base_path='../data/test_data1')
    test_loader = DataLoader(dataset=test_data, batch_size=1)
    inference(model, test_loader)
