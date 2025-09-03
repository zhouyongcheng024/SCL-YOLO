import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.feature import local_binary_pattern
from skimage import color, filters
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import cv2
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# 1. MSCO-based Segmentation Module
class MSCO_Segmenter:
    def __init__(self, radius=1, n_points=8, max_iter=50, num_coyotes=10):
        self.radius = radius
        self.n_points = n_points
        self.max_iter = max_iter
        self.num_coyotes = num_coyotes
        self.alpha = 0.5  # Weight factor for fitness-distance balance

    def _entropy(self, hist):
        prob = hist / hist.sum()
        return -np.sum(prob * np.log(prob + 1e-10))

    def segment(self, image):
        # Convert to grayscale
        gray = color.rgb2gray(image)
        
        # Initialize coyotes (threshold candidates)
        thresholds = np.random.uniform(0.1, 0.9, self.num_coyotes)
        best_threshold = 0.5
        best_entropy = -np.inf
        
        # Coyote optimization
        for _ in range(self.max_iter):
            entropies = []
            for t in thresholds:
                # Binarize image
                binary = (gray > t).astype(np.uint8)
                
                # Calculate entropy
                hist = np.histogram(binary, bins=2, range=(0, 1))[0]
                entropy_val = self._entropy(hist)
                entropies.append(entropy_val)
                
                # Update best solution
                if entropy_val > best_entropy:
                    best_entropy = entropy_val
                    best_threshold = t
            
            # Update coyotes positions (fitness-distance balance)
            mean_threshold = np.mean(thresholds)
            for i in range(self.num_coyotes):
                r1, r2 = np.random.choice(self.num_coyotes, 2, replace=False)
                social_tendency = self.alpha * mean_threshold + (1 - self.alpha) * thresholds[i]
                thresholds[i] = 0.5 * (thresholds[r1] + social_tendency) + np.random.normal(0, 0.1)
        
        # Apply best threshold
        segmented = (gray > best_threshold).astype(np.uint8) * 255
        return segmented, best_threshold

# 2. ICVSO-based Feature Selector
class ICVSO_FeatureSelector:
    def __init__(self, num_features=20):
        self.num_features = num_features
        self.selected_indices = None

    def _extract_features(self, image):
        """Extract color, texture and shape features"""
        features = []
        
        # Color features (HSV moments)
        hsv = color.rgb2hsv(image)
        for i in range(3):
            channel = hsv[:, :, i]
            features.extend([np.mean(channel), np.std(channel), np.median(channel)])
        
        # Texture features (LBP)
        gray = color.rgb2gray(image)
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
        features.extend(hist / hist.sum())
        
        # Shape features (after segmentation)
        segmented = filters.sobel(gray)
        contours, _ = cv2.findContours(
            (segmented > 0.1).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            features.extend([area, perimeter, area/perimeter if perimeter > 0 else 0])
        else:
            features.extend([0, 0, 0])
        
        return np.array(features)

    def fit(self, X, y):
        # Select top K features using mutual information
        selector = SelectKBest(mutual_info_classif, k=self.num_features)
        selector.fit(X, y)
        self.selected_indices = selector.get_support(indices=True)
        return self

    def transform(self, X):
        return X[:, self.selected_indices]

# 3. FDB-DNN Classifier
class FDB_DNN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_layers=[256, 128]):
        super(FDB_DNN, self).__init__()
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def fdb_score(self, outputs, targets, w=0.5):
        """Calculate Fitness-Distance Balance scores"""
        # Fitness: Cross entropy loss
        criterion = nn.CrossEntropyLoss(reduction='none')
        fitness = criterion(outputs, targets)
        norm_fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-10)
        
        # Distance: Euclidean distance to decision boundary
        probs = torch.softmax(outputs, dim=1)
        true_probs = probs[torch.arange(len(targets)), targets]
        distance = true_probs - (1 - true_probs)  # Simplified distance metric
        norm_distance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-10)
        
        # FDB score
        scores = w * norm_fitness + (1 - w) * norm_distance
        return scores

# 4. Custom Dataset with Integrated Processing
class HerbalDataset(Dataset):
    def __init__(self, root_dir, segmenter, selector, transform=None, num_features=20):
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        self.segmenter = segmenter
        self.selector = selector
        self.transform = transform
        self.num_features = num_features
        
        # Collect all image paths
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img_path in glob.glob(os.path.join(cls_dir, '*.jpg')):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls])
        
        # Pre-calculate features
        self.features = []
        print("Extracting features from dataset...")
        for img_path in tqdm(self.image_paths):
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            
            # Apply MSCO segmentation
            segmented, _ = self.segmenter.segment(img_array)
            
            # Extract features
            features = self.selector._extract_features(img_array)
            self.features.append(features)
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Fit feature selector
        self.selector.fit(self.features, self.labels)
        self.features = self.selector.transform(self.features)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        features = self.features[idx].astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(features), torch.tensor(label)

# 5. Training Function with FDB Mechanism
def train_fdb_dnn(model, dataloader, optimizer, device, epochs=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate FDB scores
            fdb_scores = model.fdb_score(outputs, labels)
            
            # Weighted loss based on FDB scores
            loss = torch.mean(fdb_scores * criterion(outputs, labels))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    return model

# 6. Main Training Pipeline
def main():
    # Configuration
    DATA_DIR = "./datasets/images"  
    NUM_CLASSES = 50  # Update based on your dataset
    CLASSES = CLASSES
    NUM_FEATURES = 20
    BATCH_SIZE = 32
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize modules
    segmenter = MSCO_Segmenter()
    selector = ICVSO_FeatureSelector(num_features=NUM_FEATURES)
    
    # Create datasets
    train_dataset = HerbalDataset(
        os.path.join(DATA_DIR, "train"),
        segmenter,
        selector,
        num_features=NUM_FEATURES
    )
    
    test_dataset = HerbalDataset(
        os.path.join(DATA_DIR, "test"),
        segmenter,
        selector,
        num_features=NUM_FEATURES
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_size = train_dataset.features.shape[1]
    model = FDB_DNN(input_size, NUM_CLASSES, hidden_layers=[256, 128]).to(DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Train model
    trained_model = train_fdb_dnn(model, train_loader, optimizer, DEVICE, epochs=EPOCHS)
    
    # Save model
    torch.save(trained_model.state_dict(), "fdb_dnn_herbal_classifier.pth")
    print("Model saved successfully.")
    
    # Evaluate on test set
    trained_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = trained_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
