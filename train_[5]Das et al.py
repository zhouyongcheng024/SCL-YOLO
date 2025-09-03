import torch
import yaml
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

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

# 1. 数据准备与预处理
class MedicinalLeafDataset:
    def __init__(self, raw_data_path, output_path, img_size=640):
        """
        :param raw_data_path: 原始数据路径（包含images和labels）
        :param output_path: 处理后的YOLO格式数据集路径
        :param img_size: 图像缩放尺寸
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.img_size = img_size
        self.classes = []  # 存储类别名称
        
    def _load_classes(self):
        """从标签文件中提取所有类别"""
        label_files = glob(os.path.join(self.raw_data_path, 'labels', '*.txt'))
        all_classes = set()
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    all_classes.add(class_id)
        
        self.classes = sorted(list(all_classes))
        return self.classes
    
    def _resize_image(self, img_path, save_path):
        """等比例缩放图像并保持宽高比"""
        img = Image.open(img_path)
        w, h = img.size
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # 创建新图像（填充灰色背景）
        new_img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        new_img.paste(img, ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2))
        new_img.save(save_path)
        return scale, (self.img_size - new_w) // 2, (self.img_size - new_h) // 2
    
    def _adjust_annotations(self, label_path, save_path, scale, pad_x, pad_y):
        """调整标注坐标"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # 调整坐标
            x_center = x_center * scale + pad_x
            y_center = y_center * scale + pad_y
            width = width * scale
            height = height * scale
            
            # 归一化
            x_center /= self.img_size
            y_center /= self.img_size
            width /= self.img_size
            height /= self.img_size
            
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        with open(save_path, 'w') as f:
            f.writelines(new_lines)
    
    def prepare_dataset(self, test_size=0.3):
        """准备YOLO格式数据集"""
        # 创建目录结构
        os.makedirs(os.path.join(self.output_path, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'labels', 'val'), exist_ok=True)
        
        # 加载类别
        self._load_classes()
        
        # 获取所有图像文件
        image_files = glob(os.path.join(self.raw_data_path, 'images', '*.jpg'))
        image_files += glob(os.path.join(self.raw_data_path, 'images', '*.png'))
        
        # 划分训练集和验证集 (70/30)
        train_files, val_files = train_test_split(
            image_files, test_size=test_size, random_state=42
        )
        
        print(f"Preparing training set ({len(train_files)} images)...")
        for img_path in tqdm(train_files):
            base_name = os.path.basename(img_path)
            label_path = os.path.join(self.raw_data_path, 'labels', 
                                    os.path.splitext(base_name)[0] + '.txt')
            
            # 处理图像
            save_img_path = os.path.join(self.output_path, 'images', 'train', base_name)
            scale, pad_x, pad_y = self._resize_image(img_path, save_img_path)
            
            # 处理标注
            if os.path.exists(label_path):
                save_label_path = os.path.join(self.output_path, 'labels', 'train', 
                                             os.path.splitext(base_name)[0] + '.txt')
                self._adjust_annotations(label_path, save_label_path, scale, pad_x, pad_y)
        
        print(f"Preparing validation set ({len(val_files)} images)...")
        for img_path in tqdm(val_files):
            base_name = os.path.basename(img_path)
            label_path = os.path.join(self.raw_data_path, 'labels', 
                                    os.path.splitext(base_name)[0] + '.txt')
            
            # 处理图像
            save_img_path = os.path.join(self.output_path, 'images', 'val', base_name)
            scale, pad_x, pad_y = self._resize_image(img_path, save_img_path)
            
            # 处理标注
            if os.path.exists(label_path):
                save_label_path = os.path.join(self.output_path, 'labels', 'val', 
                                             os.path.splitext(base_name)[0] + '.txt')
                self._adjust_annotations(label_path, save_label_path, scale, pad_x, pad_y)
        
        # 创建data.yaml配置文件
        data_yaml = {
            'train': os.path.join(self.output_path, 'images', 'train'),
            'val': os.path.join(self.output_path, 'images', 'val'),
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(os.path.join(self.output_path, 'data.yaml'), 'w') as f:
            yaml.dump(data_yaml, f)
        
        return data_yaml

# 2. YOLOv7模型配置
class YOLOv7Config:
    def __init__(self, model_size='base', num_classes=30):
        """
        :param model_size: 模型大小 ('tiny', 'base', 'large', 'x')
        :param num_classes: 类别数量
        """
        self.model_size = model_size
        self.num_classes = num_classes
        
    def get_config(self):
        """获取YOLOv7配置字典"""
        # 基础配置
        config = {
            'nc': self.num_classes,
            'depth_multiple': 1.0,
            'width_multiple': 1.0,
            'anchors': [
                [12,16, 19,36, 40,28],      # P3/8
                [36,75, 76,55, 72,146],     # P4/16
                [142,110, 192,243, 459,401] # P5/32
            ],
            'backbone': [
                [-1, 1, 'Conv', [32, 3, 1]],  # 0
                [-1, 1, 'Conv', [64, 3, 2]],  # 1-P1/2
                [-1, 1, 'Conv', [64, 3, 1]],
                [-1, 1, 'Conv', [128, 3, 2]], # 3-P2/4
                [-1, 1, 'C3', [128]],
                [-1, 1, 'Conv', [256, 3, 2]], # 5-P3/8
                [-1, 1, 'C3', [256]],
                [-1, 1, 'Conv', [512, 3, 2]], # 7-P4/16
                [-1, 1, 'C3', [512]],
                [-1, 1, 'Conv', [1024, 3, 2]],# 9-P5/32
                [-1, 1, 'C3', [1024]],
                [-1, 1, 'SPPF', [1024, 5]],
            ],
            'head': [
                [-1, 1, 'Conv', [512, 1, 1]],
                [-1, 1, 'Upsample', [None, 2, 'nearest']],
                [[-1, 8], 1, 'Concat', [1]],
                [-1, 1, 'C3', [512, False]],
                [-1, 1, 'Conv', [256, 1, 1]],
                [-1, 1, 'Upsample', [None, 2, 'nearest']],
                [[-1, 6], 1, 'Concat', [1]],
                [-1, 1, 'C3', [256, False]],
                [-1, 1, 'Conv', [256, 3, 2]],
                [[-1, 4], 1, 'Concat', [1]],
                [-1, 1, 'C3', [512, False]],
                [-1, 1, 'Conv', [512, 3, 2]],
                [[-1, 2], 1, 'Concat', [1]],
                [-1, 1, 'C3', [1024, False]],
                [[15, 18, 21], 1, 'Detect', [self.num_classes, self.get_anchors()]],
            ]
        }
        
        # 根据模型大小调整配置
        if self.model_size == 'tiny':
            config['depth_multiple'] = 0.33
            config['width_multiple'] = 0.25
        elif self.model_size == 'large':
            config['depth_multiple'] = 1.0
            config['width_multiple'] = 1.0
        elif self.model_size == 'x':
            config['depth_multiple'] = 1.33
            config['width_multiple'] = 1.25
        
        return config
    
    def get_anchors(self):
        """获取锚框配置"""
        return [
            [12,16, 19,36, 40,28],      # P3/8
            [36,75, 76,55, 72,146],     # P4/16
            [142,110, 192,243, 459,401] # P5/32
        ]
    
    def save_config(self, save_path):
        """保存YOLO配置到YAML文件"""
        config = self.get_config()
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
        return save_path

# 3. YOLOv7模型训练
class YOLOv7Trainer:
    def __init__(self, data_yaml, model_config, device='cuda', img_size=640):
        """
        :param data_yaml: 数据配置文件路径
        :param model_config: 模型配置文件路径
        :param device: 训练设备
        :param img_size: 图像尺寸
        """
        self.data_yaml = data_yaml
        self.model_config = model_config
        self.device = device
        self.img_size = img_size
        
    def train(self, epochs=20, batch_size=16, weights=None):
        """训练YOLOv7模型"""
        # 克隆YOLOv7官方仓库
        if not os.path.exists('yolov7'):
            os.system('git clone https://github.com/WongKinYiu/yolov7.git')
        os.chdir('yolov7')
        
        # 安装依赖
        os.system('pip install -r requirements.txt')
        
        # 训练命令
        cmd = f'python train.py \
                --img {self.img_size} \
                --batch {batch_size} \
                --epochs {epochs} \
                --data {self.data_yaml} \
                --cfg {self.model_config} \
                --device {self.device} \
                --name medicinal_leaf \
                --hyp data/hyp.scratch.p5.yaml'
        
        if weights:
            cmd += f' --weights {weights}'
        else:
            cmd += ' --weights ""'
        
        # 执行训练
        os.system(cmd)
        
        # 返回最佳模型路径
        best_model = 'runs/train/medicinal_leaf/weights/best.pt'
        return best_model

# 4. 模型评估
class YOLOv7Evaluator:
    def __init__(self, model_path, data_yaml, device='cuda', img_size=640):
        """
        :param model_path: 模型路径
        :param data_yaml: 数据配置文件路径
        :param device: 评估设备
        :param img_size: 图像尺寸
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.device = device
        self.img_size = img_size
        
    def evaluate(self):
        """评估模型性能"""
        # 确保在yolov7目录
        if not os.path.exists('yolov7'):
            raise FileNotFoundError("YOLOv7 directory not found")
        os.chdir('yolov7')
        
        # 评估命令
        cmd = f'python test.py \
                --weights {self.model_path} \
                --data {self.data_yaml} \
                --img {self.img_size} \
                --device {self.device} \
                --task test \
                --name medicinal_leaf_eval'
        
        os.system(cmd)
        
        # 解析结果
        results_file = 'runs/test/medicinal_leaf_eval/results.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results
        return None

# 5. 主函数
def main():
    # 配置参数
    RAW_DATA_PATH = "./datasets/images"  # 原始数据集路径
    OUTPUT_PATH = "./datasets/handle_images"    # 处理后的数据集路径
    MODEL_SIZE = 'base'  # 模型大小 (tiny, base, large, x)
    NUM_CLASSES = 50     # 类别数量
    EPOCHS = 100          # 训练轮数（论文设置）
    BATCH_SIZE = 32      # 批大小
    IMG_SIZE = 640       # 图像尺寸
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. 准备数据集
    print("Preparing dataset...")
    dataset = MedicinalLeafDataset(RAW_DATA_PATH, OUTPUT_PATH, IMG_SIZE)
    data_yaml_path = dataset.prepare_dataset(test_size=0.3)
    
    # 2. 创建模型配置
    print("Creating model configuration...")
    model_config = YOLOv7Config(model_size=MODEL_SIZE, num_classes=NUM_CLASSES)
    model_config_path = model_config.save_config(os.path.join(OUTPUT_PATH, 'yolov7.yaml'))
    
    # 3. 训练模型
    print("Training YOLOv7 model...")
    trainer = YOLOv7Trainer(
        data_yaml=os.path.join(OUTPUT_PATH, 'data.yaml'),
        model_config=model_config_path,
        device=device,
        img_size=IMG_SIZE
    )
    best_model = trainer.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    print(f"Training completed. Best model saved at: {best_model}")
    
    # 4. 评估模型
    print("Evaluating model...")
    evaluator = YOLOv7Evaluator(
        model_path=best_model,
        data_yaml=os.path.join(OUTPUT_PATH, 'data.yaml'),
        device=device,
        img_size=IMG_SIZE
    )
    results = evaluator.evaluate()
    
    if results:
        print("\nEvaluation Results:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"mAP@0.5: {results['mAP_0.5']:.4f}")
        print(f"mAP@0.5:0.95: {results['mAP_0.5:0.95']:.4f}")
    
    # 5. 导出模型
    print("Exporting model to ONNX...")
    os.system(f'python export.py --weights {best_model} --img-size {IMG_SIZE} --include onnx')
    onnx_model = best_model.replace('.pt', '.onnx')
    print(f"Model exported to: {onnx_model}")

if __name__ == "__main__":
    main()
