import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern
import os
import glob

CLASSES = [
        "ginseng",
        "Leech",
        "JujubaeFructus",
        "LiliiBulbus",
        "CoptidisRhizoma",
        "MumeFructus",
        "MagnoliaBark",
        "Oyster",
        "Seahorse",
        "Luohanguo",
        "GlycyrrhizaUralensis",
        "Sanqi",
        "TetrapanacisMedulla",
        "CoicisSemen",
        "LyciiFructus",
        "TruestarAnise",
        "ClamShell",
        "Chuanxiong",
        "Garlic",
        "GinkgoBiloba",
        "ChrysanthemiFlos",
        "AtractylodesMacrocephala",
        "JuglandisSemen",
        "TallGastrodiae",
        "TrionycisCarapax",
        "AngelicaRoot",
        "Hawthorn",
        "CrociStigma",
        "SerpentisPeriostracum",
        "EucommiaBark",
        "ImperataeRhizoma",
        "LoniceraJaponica",
        "Zhizi",
        "Scorpion",
        "HouttuyniaeHerba",
        "EupolyphagaSinensis",
        "OroxylumIndicum",
        "CurcumaLonga",
        "NelumbinisPlumula",
        "ArecaeSemen",
        "Scolopendra",
        "MoriFructus",
        "FritillariaeCirrhosaeBulbus",
        "DioscoreaeRhizoma",
        "CicadaePeriostracum",
        "PiperCubeba",
        "BupleuriRadix",
        "AntelopeHom",
        "Pangdahai",
        "NelumbinisSemen",
        ]

# 1. CBAM注意力模块实现
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(concat)
        return self.sigmoid(att)

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)  # 通道注意力
        x = x * self.sa(x)  # 空间注意力
        return x

# 2. 注意力增强的ResNet模块
class AttentionBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(AttentionBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        # 添加CBAM注意力模块
        self.cbam = CBAM(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        # 应用注意力机制
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 3. LBP特征提取器
class LBPFeatureExtractor(nn.Module):
    def __init__(self, radius=1, n_points=8):
        super(LBPFeatureExtractor, self).__init__()
        self.radius = radius
        self.n_points = n_points
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # 在GPU上计算LBP特征
        lbp_features = []
        for i in range(x.size(0)):
            img = x[i].squeeze().cpu().numpy()
            lbp_img = local_binary_pattern(img, self.n_points, self.radius, method='uniform')
            lbp_img = torch.from_numpy(lbp_img).unsqueeze(0).to(x.device)
            lbp_features.append(lbp_img)
        
        x = torch.stack(lbp_features, dim=0)
        x = x.unsqueeze(1).float()  # 添加通道维度
        
        # LBP特征处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        
        return x.view(x.size(0), -1)

# 4. 完整的注意力增强ResNet模型（带LBP特征融合）
class AttentionResNetWithLBP(nn.Module):
    def __init__(self, block, layers, num_classes=60, zero_init_residual=False):
        super(AttentionResNetWithLBP, self).__init__()
        self.inplanes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 自适应池化
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LBP特征提取器
        self.lbp_extractor = LBPFeatureExtractor()
        
        # 分类器（融合深度特征和LBP特征）
        self.fc1 = nn.Linear(512 * block.expansion + 128, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 深度特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_avg_pool(x)
        deep_features = x.view(x.size(0), -1)
        
        # LBP特征提取（使用原始输入的灰度图）
        with torch.no_grad():
            gray_x = 0.2989 * x[:, 0, :, :] + 0.5870 * x[:, 1, :, :] + 0.1140 * x[:, 2, :, :]
            gray_x = gray_x.unsqueeze(1)  # 添加通道维度
        lbp_features = self.lbp_extractor(gray_x)
        
        # 特征融合
        combined_features = torch.cat((deep_features, lbp_features), dim=1)
        
        # 分类
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 5. 数据增强和数据集类
class HerbalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        self.transform = transform
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_path in glob.glob(os.path.join(cls_dir, '*.jpg')):
                self.images.append((img_path, self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 6. 训练函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=150, device='cuda'):
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return model

# 7. 主函数
def main():
    # 数据增强
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 创建数据集
    data_dir = './datasets/images'  
    image_datasets = {
        'CLASSES': CLASSES,
        'train': HerbalDataset(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': HerbalDataset(os.path.join(data_dir, 'val'), data_transforms['val'])
    }
    
    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=4)
    }
    
    # 初始化模型
    model = AttentionResNetWithLBP(AttentionBottleneck, [3, 4, 6, 3], num_classes=50)
    
    # 使用预训练权重（ImageNet）
    pretrained_dict = torch.load('resnet50.pth')  # 加载预训练权重
    model_dict = model.state_dict()
    
    # 过滤出匹配的权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # 更新模型权重
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=150, device=device)
    
    # 保存模型
    torch.save(model.state_dict(), 'attention_resnet_lbp.pth')

if __name__ == '__main__':
    main()
