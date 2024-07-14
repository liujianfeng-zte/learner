from PIL import Image
from torchvision import datasets, transforms
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

def preprocess_image(size=128):
    transform = transforms.Compose([
        transforms.Resize(size),  # 调整图像短边为 128 像素，保持长宽比
        transforms.Lambda(pad_to_square),  # 填充长边到正方形
        transforms.Resize((size, size)),  # 最终调整图像为 128x128
        transforms.Lambda(convert_to_rgb),  # 确保图像为 RGB 格式
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])
    return transform

# 自定义加载图片处理函数
def load_image_with_transparency(image_path):
    image = Image.open(image_path)
    if image.mode == 'P':  # 检查是否为调色板图像
        image = image.convert('RGBA')
    if image.mode == 'RGBA':  # 如果是RGBA格式，则转换为RGB格式
        image = image.convert('RGB')
    return image

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

def augment_and_preprocess_image_for_training(size=128):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.Lambda(pad_to_square),
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomCrop(size, padding=4),  # 随机裁剪
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform

