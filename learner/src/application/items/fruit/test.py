import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import warnings

# 过滤警告信息
warnings.filterwarnings("ignore")

# 自定义加载图片处理函数
def load_image_with_transparency(image_path):
    image = Image.open(image_path)
    if image.mode == 'P':  # 检查是否为调色板图像
        image = image.convert('RGBA')
    if image.mode == 'RGBA':  # 如果是RGBA格式，则转换为RGB格式
        image = image.convert('RGB')
    return image

# 自定义数据集类
class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = load_image_with_transparency(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

# 自定义填充函数
def pad_to_square(img):
    w, h = img.size
    max_side = max(w, h)
    padding = (max_side - w, max_side - h)
    # 填充图像，使其变成正方形
    return transforms.functional.pad(img, (padding[0] // 2, padding[1] // 2, (padding[0] + 1) // 2, (padding[1] + 1) // 2))

# 确保图像为 RGB 格式
def convert_to_rgb(img):
    return img.convert('RGB')

# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Resize(128),  # 调整图像短边为 128 像素，保持长宽比
    transforms.Lambda(pad_to_square),  # 填充长边到正方形
    transforms.Resize((128, 128)),  # 最终调整图像为 128x128
    transforms.Lambda(convert_to_rgb),  # 确保图像为 RGB 格式
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.Lambda(pad_to_square),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomCrop(128, padding=4),  # 随机裁剪
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义改进后的模型
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # 设置数据集文件夹路径
    train_data_dir = 'G:/data/images/train'
    test_data_dir = 'G:/data/images/test'

    # 读取文件夹名称生成映射字典
    try:
        class_to_fruit = {i: folder for i, folder in enumerate(sorted(os.listdir(train_data_dir)))}
    except FileNotFoundError:
        print(f"路径 {train_data_dir} 不存在，请确认路径是否正确。")
        exit()

    # 打印生成的映射字典
    print("生成的映射字典:")
    for class_id, fruit_name in class_to_fruit.items():
        print(f"类别 {class_id}: {fruit_name}")

    # 生成训练集和测试集的数据集对象
    train_dataset = CustomImageFolder(root=train_data_dir, transform=train_transform)
    test_dataset = CustomImageFolder(root=test_data_dir, transform=transform)

    print("训练数据集的长度为：{}".format(len(train_dataset)))
    print("测试数据集的长度为：{}".format(len(test_dataset)))

    # 利用 DataLoader 来加载数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 获取类别数量
    num_classes = len(class_to_fruit)

    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = ImprovedCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 模型测试
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        print(f'测试集上的准确率: {100 * correct / total}%')

    # 打印实际标签与预测标签对应的水果名称
    print("实际标签与预测标签对应的水果名称:")
    for actual, pred in zip(all_labels, all_preds):
        actual_fruit = class_to_fruit[actual]
        predicted_fruit = class_to_fruit[pred]
        print(f"实际: {actual_fruit}, 预测: {predicted_fruit}")

if __name__ == '__main__':
    main()
