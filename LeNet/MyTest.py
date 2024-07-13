import torch
from MyLeNet import MyNet
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToPILImage
from torchvision.transforms import transforms

# 将数据转化为tensor
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用模型
model = MyNet().to(device)

model.load_state_dict(torch.load("D:\\PythonProject\\LeNet\checkpoints\\best_model.pt"))

# 获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把tonsor转化为图片，方便可视化
show = ToPILImage()

# 进入验证
for i in range(2):
    x, y = test_dataset[i][0], train_dataset[i][0]
    # print(test_dataset[i][0])
    # print(train_dataset[i][0])
    show(x).show()
    img = torch.unsqueeze(x, dim=0)
    # print(img.size())
    img = img.to(device)
    with torch.no_grad():
        output = torch.squeeze(model(img), dim=0)
        # print(output)
        predict = classes[torch.argmax(output)]
        print(f'Predicted: {predict}')
        print("\n")
