import os
import json
import warnings
import torch
from torch.utils.data import DataLoader
from src.domain.service.fruit.model import ImprovedCNN
from src.domain.service.fruit.train_model import train_model
from src.domain.service.fruit.test_model import test_model
from src.infrastructure.deep_learn.image_process import augment_and_preprocess_image_for_training, preprocess_image, CustomImageFolder
from src.infrastructure.utils.map_utils import get_keys_by_value

warnings.filterwarnings("ignore")

class FruitDeepLearner:
    def __init__(self, mode='train'):
        self.train_class_to_fruit = None
        self.test_class_to_fruit = {}
        self.data_loader = None
        self.num_epochs = 50
        self.learning_rate = 0.0001
        self.image_size = 64
        self.mode = mode
        self.train_data_dir = 'G:/data/images/train'
        self.test_data_dir = 'G:/data/images/test'
        self.model_path = f'fruit_{self.image_size}_{self.num_epochs}_{self.learning_rate}.pth'

    def prepare_data(self):
        if self.mode == 'train':
            self.train_class_to_fruit = {i: folder for i, folder in enumerate(sorted(os.listdir(self.train_data_dir)))}

            print("生成的映射字典（训练集）:")
            for class_id, fruit_name in self.train_class_to_fruit.items():
                print(f"类别 {class_id}: {fruit_name}")

            transform = augment_and_preprocess_image_for_training(self.image_size)
            dataset = CustomImageFolder(root=self.train_data_dir, transform=transform)
            print("训练数据集的长度为：{}".format(len(dataset)))
            self.data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
            num_classes = len(self.train_class_to_fruit)
            self.model = ImprovedCNN(num_classes=num_classes).cuda()

        elif self.mode == 'test':
            self.train_class_to_fruit = {i: folder for i, folder in enumerate(sorted(os.listdir(self.train_data_dir)))}
            self.test_class_to_fruit = {i: folder for i, folder in enumerate(sorted(os.listdir(self.test_data_dir)))}

            print("生成的映射字典（测试集）:")
            for class_id, fruit_name in self.test_class_to_fruit.items():
                print(f"类别 {class_id}: {fruit_name}")

            transform = preprocess_image(self.image_size)
            dataset = CustomImageFolder(root=self.test_data_dir, transform=transform)
            print("测试数据集的长度为：{}".format(len(dataset)))
            self.data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        self.prepare_data()
        train_model(self.model, self.data_loader, self.num_epochs, self.learning_rate)
        torch.save(self.model.state_dict(), self.model_path)

    def test(self):
        accuracy_num = 0
        self.prepare_data()
        self.model = ImprovedCNN(num_classes=len(self.train_class_to_fruit)).cuda()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        all_labels, all_preds = test_model(self.model, self.data_loader)

        # 打印实际标签和预测标签，并检查标签映射关系
        print("实际标签与预测标签:")
        for actual, pred in zip(all_labels, all_preds):
            print(f"实际标签: {actual}, 预测标签: {pred}")

        print("实际标签与预测标签对应的水果名称:")
        for actual, pred in zip(all_labels, all_preds):
            actual_fruit = self.test_class_to_fruit.get(actual, "未知类别")
            predicted_fruit = self.train_class_to_fruit.get(pred, "未知类别")
            print(f"实际: {actual_fruit}, 预测: {predicted_fruit}")
            if(actual_fruit == predicted_fruit):
                accuracy_num += 1
        accuracy_rate = accuracy_num / len(all_labels)
        # 精确到小数点后四位
        accuracy_rate = round(accuracy_rate, 4)
        print(f'测试集上的准确率: {accuracy_rate * 100}%')

if __name__ == '__main__':
    mode = input("请输入模式（train/test）：").strip()
    fruit = FruitDeepLearner(mode=mode)
    if mode == 'train':
        fruit.train()
    elif mode == 'test':
        fruit.test()
    else:
        print("无效的模式，请输入 'train' 或 'test'")
