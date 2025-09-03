import os
import shutil
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

# 1. 数据集准备与预处理
class HerbDatasetPreparer:
    def __init__(self, raw_data_path, output_path, classes):
        """
        :param raw_data_path: 原始图像路径 (包含不同子文件夹)
        :param output_path: 处理后的YOLO格式数据集路径
        :param classes: 类别列表 ['bay leaf', 'rosemary', 'mint']
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.classes = classes
        self.class_to_id = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        # 创建YOLO需要的目录结构
        self.image_dirs = {
            'train': os.path.join(output_path, 'images', 'train'),
            'val': os.path.join(output_path, 'images', 'val'),
            'test': os.path.join(output_path, 'images', 'test')
        }
        self.label_dirs = {
            'train': os.path.join(output_path, 'labels', 'train'),
            'val': os.path.join(output_path, 'labels', 'val'),
            'test': os.path.join(output_path, 'labels', 'test')
        }
        
        for dir_path in list(self.image_dirs.values()) + list(self.label_dirs.values()):
            os.makedirs(dir_path, exist_ok=True)

    def _filter_blurred_images(self, image_paths, threshold=100):
        """过滤模糊图像（论文中提到的预处理步骤）"""
        keep_paths = []
        for img_path in tqdm(image_paths, desc="Filtering blurred images"):
            img = Image.open(img_path).convert('L')  # 转换为灰度图
            laplacian_var = np.var(np.array(img))
            if laplacian_var > threshold:
                keep_paths.append(img_path)
        return keep_paths

    def prepare_dataset(self, val_ratio=0.1, test_ratio=0.1):
        """准备YOLO格式数据集"""
        # 收集所有图像路径
        all_images = []
        for root, _, files in os.walk(self.raw_data_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(root, file))
        
        # 过滤模糊图像（论文中提到的数据清洗）
        all_images = self._filter_blurred_images(all_images)
        
        # 划分数据集（论文中90%训练，10%验证）
        train_val, test_images = train_test_split(
            all_images, test_size=test_ratio, random_state=42
        )
        train_images, val_images = train_test_split(
            train_val, test_size=val_ratio/(1-test_ratio), random_state=42
        )
        
        # 创建数据集分割
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        # 创建YOLO数据配置文件
        self._create_data_yaml()
        
        return splits

    def _create_data_yaml(self):
        """创建YOLO数据配置文件"""
        data = {
            'path': self.output_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.classes,
            'nc': len(self.classes)
        }
        
        with open(os.path.join(self.output_path, 'herbs.yaml'), 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        return os.path.join(self.output_path, 'herbs.yaml')

# 2. YOLOv8模型训练器
class HerbDetectorTrainer:
    def __init__(self, data_yaml, model_size='m', device='cuda'):
        """
        :param data_yaml: YOLO数据配置文件路径
        :param model_size: 模型大小 (n, s, m, l, x)
        :param device: 训练设备
        """
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.device = device
        self.model = YOLO(f'yolov8{model_size}.pt')  # 加载预训练模型

    def train(self, epochs=300, imgsz=640, batch=16, patience=50, **kwargs):
        """训练模型（论文中使用300个epoch）"""
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=self.device,
            project='herb_detection',
            name=f'yolov8_{self.model_size}',
            **kwargs
        )
        return results

    def evaluate(self, model_path, split='val'):
        """评估模型性能（论文中使用的指标）"""
        model = YOLO(model_path)
        metrics = model.val(
            data=self.data_yaml,
            split=split,
            imgsz=640,
            conf=0.25,
            iou=0.6,
            device=self.device
        )
        
        # 提取关键指标（论文中关注的指标）
        results = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
            'f1': metrics.box.mf1,
        }
        
        # 打印每个类别的指标
        for idx, cls_name in enumerate(metrics.names.values()):
            results[f'{cls_name}_precision'] = metrics.box.p[idx]
            results[f'{cls_name}_recall'] = metrics.box.r[idx]
        
        return results

    def export(self, model_path, format='onnx'):
        """导出模型为部署格式"""
        model = YOLO(model_path)
        model.export(format=format, imgsz=640, simplify=True)
        return model_path.replace('.pt', f'.{format}')

# 3. 训练流程整合
def main():
    # === 配置参数（论文中的设置） ===
    RAW_DATA_PATH = "./datasets/images"  # 原始图像路径
    OUTPUT_PATH = "./datasets/handle_images"       # 处理后的数据集路径
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
    
    # 训练参数（论文中使用300个epoch）
    EPOCHS = 300
    MODEL_SIZE = 'm'  # 中等大小模型（平衡精度和速度）
    DEVICE = 'cuda'   # 使用GPU加速
    
    # === 1. 准备数据集 ===
    print("Preparing dataset...")
    preparer = HerbDatasetPreparer(RAW_DATA_PATH, OUTPUT_PATH, CLASSES)
    preparer.prepare_dataset(val_ratio=0.1, test_ratio=0.1)
    data_yaml = os.path.join(OUTPUT_PATH, 'herbs.yaml')
    
    # === 2. 训练模型 ===
    print("Initializing trainer...")
    trainer = HerbDetectorTrainer(data_yaml, model_size=MODEL_SIZE, device=DEVICE)
    
    print("Starting training...")
    train_results = trainer.train(
        epochs=EPOCHS,
        batch=16,              # 根据GPU内存调整
        imgsz=640,             # 图像尺寸
        lr0=0.01,              # 初始学习率
        lrf=0.01,              # 最终学习率
        momentum=0.937,        # 动量
        weight_decay=0.0005,   # 权重衰减
        warmup_epochs=3,       # 热身epoch
        warmup_momentum=0.8,   # 热身动量
        box=7.5,               # 框损失权重
        cls=0.5,               # 分类损失权重
        dfl=1.5,               # dfl损失权重
        fl_gamma=0.0,          # 焦点损失gamma
        label_smoothing=0.0,   # 标签平滑
        patience=50            # 早停耐心值
    )
    
    # === 3. 评估模型 ===
    print("Evaluating model...")
    best_model_path = os.path.join('herb_detection', f'yolov8_{MODEL_SIZE}', 'weights', 'best.pt')
    metrics = trainer.evaluate(best_model_path)
    
    # 打印关键指标（论文中关注的指标）
    print("\n=== Evaluation Results ===")
    print(f"mAP50: {metrics['mAP50']:.3f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    
    for cls in CLASSES:
        print(f"\n{cls} Performance:")
        print(f"  Precision: {metrics[f'{cls}_precision']:.3f}")
        print(f"  Recall: {metrics[f'{cls}_recall']:.3f}")
    
    # === 4. 导出模型 ===
    print("Exporting model...")
    exported_model = trainer.export(best_model_path, format='onnx')
    print(f"Exported model to: {exported_model}")

if __name__ == "__main__":
    main()
