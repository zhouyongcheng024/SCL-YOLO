import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import pandas as pd
import os
import json

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

# 1. 数据加载与预处理
class TCMDataLoader:
    def __init__(self, prescription_path, kg_path):
        """
        :param prescription_path: 处方数据路径
        :param kg_path: 知识图谱路径
        """
        self.prescription_path = prescription_path
        self.kg_path = kg_path
        self.symptom2id = {}
        self.herb2id = {}
        self.id2herb = {}
        self.meridial2id = {}
        self.flavor2id = {}
        
    def load_data(self):
        # 加载处方数据
        prescriptions = pd.read_csv(self.prescription_path)
        print(f"Loaded {len(prescriptions)} prescriptions")
        
        # 创建症状和草药映射
        all_symptoms = set()
        all_herbs = set()
        for idx, row in prescriptions.iterrows():
            symptoms = set(row['symptoms'].split(','))
            herbs = set(row['herbs'].split(','))
            all_symptoms |= symptoms
            all_herbs |= herbs
        
        self.symptom2id = {s: i for i, s in enumerate(sorted(all_symptoms))}
        self.herb2id = {h: i for i, h in enumerate(sorted(all_herbs))}
        self.id2herb = {i: h for h, i in self.herb2id.items()}
        
        # 加载知识图谱
        with open(self.kg_path, 'r') as f:
            kg_data = json.load(f)
        
        # 创建属性映射
        all_meridials = set()
        all_flavors = set()
        for herb, attributes in kg_data.items():
            if herb in self.herb2id:
                all_meridials.update(attributes.get('meridial', []))
                all_flavors.update(attributes.get('flavor', []))
        
        self.meridial2id = {m: i for i, m in enumerate(sorted(all_meridials))}
        self.flavor2id = {f: i for i, f in enumerate(sorted(all_flavors))}
        
        return prescriptions, kg_data
    
    def build_graphs(self, prescriptions, kg_data, s_s_threshold=5, h_h_threshold=40):
        """
        构建三个图：S-H图、S-S图、H-H图
        :param s_s_threshold: S-S图共现阈值
        :param h_h_threshold: H-H图共现阈值
        """
        # 1. 构建S-H图（症状-草药二分图）
        sh_edges = []
        for _, row in prescriptions.iterrows():
            symptoms = row['symptoms'].split(',')
            herbs = row['herbs'].split(',')
            for s in symptoms:
                for h in herbs:
                    sh_edges.append((self.symptom2id[s], self.herb2id[h]))
        
        # 转换为PyG格式
        sh_edge_index = torch.tensor(list(zip(*sh_edges)), dtype=torch.long)
        sh_graph = Data(edge_index=sh_edge_index)
        
        # 2. 构建S-S图（症状协同图）
        symptom_cooccur = defaultdict(int)
        for _, row in prescriptions.iterrows():
            symptoms = list(row['symptoms'].split(','))
            for i in range(len(symptoms)):
                for j in range(i+1, len(symptoms)):
                    s1, s2 = sorted([symptoms[i], symptoms[j]])
                    symptom_cooccur[(s1, s2)] += 1
        
        # 过滤低于阈值的边
        ss_edges = []
        for (s1, s2), count in symptom_cooccur.items():
            if count >= s_s_threshold:
                ss_edges.append((self.symptom2id[s1], self.symptom2id[s2]))
        
        ss_edge_index = torch.tensor(list(zip(*ss_edges)), dtype=torch.long)
        ss_graph = Data(edge_index=ss_edge_index)
        
        # 3. 构建H-H图（草药协同图）
        herb_cooccur = defaultdict(int)
        for _, row in prescriptions.iterrows():
            herbs = list(row['herbs'].split(','))
            for i in range(len(herbs)):
                for j in range(i+1, len(herbs)):
                    h1, h2 = sorted([herbs[i], herbs[j]])
                    herb_cooccur[(h1, h2)] += 1
        
        # 过滤低于阈值的边
        hh_edges = []
        for (h1, h2), count in herb_cooccur.items():
            if count >= h_h_threshold:
                hh_edges.append((self.herb2id[h1], self.herb2id[h2]))
        
        hh_edge_index = torch.tensor(list(zip(*hh_edges)), dtype=torch.long)
        hh_graph = Data(edge_index=hh_edge_index)
        
        return sh_graph, ss_graph, hh_graph
    
    def create_herb_attributes(self, kg_data):
        """
        创建草药属性矩阵
        :return: 草药属性矩阵 (num_herbs, num_attributes)
        """
        num_herbs = len(self.herb2id)
        num_meridials = len(self.meridial2id)
        num_flavors = len(self.flavor2id)
        
        # 属性矩阵：经络分布 + 味道
        attribute_matrix = torch.zeros((num_herbs, num_meridials + num_flavors))
        
        for herb, herb_id in self.herb2id.items():
            if herb in kg_data:
                attributes = kg_data[herb]
                # 经络分布
                for m in attributes.get('meridial', []):
                    if m in self.meridial2id:
                        attribute_matrix[herb_id, self.meridial2id[m]] = 1
                # 味道
                for f in attributes.get('flavor', []):
                    if f in self.flavor2id:
                        attribute_matrix[herb_id, num_meridials + self.flavor2id[f]] = 1
        
        return attribute_matrix

# 2. 图卷积网络模块
class S_MUGCN(nn.Module):
    """症状-草药图上的症状GCN"""
    def __init__(self, in_dim, out_dim, num_layers=2):
        super(S_MUGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else out_dim, out_dim))
        
    def forward(self, x, edge_index):
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.tanh(x)
            layer_outputs.append(x)
        return layer_outputs

class H_MUGCN(nn.Module):
    """症状-草药图上的草药GCN"""
    def __init__(self, in_dim, out_dim, num_layers=2):
        super(H_MUGCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(in_dim if i == 0 else out_dim, out_dim))
        
    def forward(self, x, edge_index):
        layer_outputs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.tanh(x)
            layer_outputs.append(x)
        return layer_outputs

class S_GCN(nn.Module):
    """症状-症状图上的GCN"""
    def __init__(self, in_dim, out_dim):
        super(S_GCN, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        return F.tanh(self.conv(x, edge_index))

class H_KG_GCN(nn.Module):
    """草药-草药图上的知识增强GCN"""
    def __init__(self, in_dim, out_dim):
        super(H_KG_GCN, self).__init__()
        self.conv = GCNConv(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        return F.tanh(self.conv(x, edge_index))

# 3. 多层信息融合模块
class MultiLayerFusion(nn.Module):
    """多层信息融合机制"""
    def __init__(self, in_dim, out_dim):
        super(MultiLayerFusion, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh()
        )
    
    def forward(self, layer_outputs):
        # 平均各层输出
        fused = torch.stack(layer_outputs, dim=0).mean(dim=0)
        # 添加自身信息
        fused = fused + layer_outputs[0]
        # MLP变换
        return self.mlp(fused)

# 4. 完整的KDHR模型
class KDHR(nn.Module):
    def __init__(self, num_symptoms, num_herbs, attribute_dim, 
                 emb_dim=64, hidden_dim=256, num_layers=2):
        super(KDHR, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 初始化嵌入
        self.symptom_emb = nn.Embedding(num_symptoms, emb_dim)
        self.herb_emb = nn.Embedding(num_herbs, emb_dim)
        
        # 图卷积网络
        self.s_mugcn = S_MUGCN(emb_dim, emb_dim, num_layers)
        self.h_mugcn = H_MUGCN(emb_dim, emb_dim, num_layers)
        self.s_gcn = S_GCN(emb_dim, emb_dim)
        self.h_kg_gcn = H_KG_GCN(emb_dim + attribute_dim, emb_dim)
        
        # 多层信息融合
        self.s_fusion = MultiLayerFusion(emb_dim, emb_dim)
        self.h_fusion = MultiLayerFusion(emb_dim, emb_dim)
        
        # 症状集综合嵌入
        self.symptom_set_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        
        # 输出层
        self.output_layer = nn.Linear(emb_dim, 1)
        
    def forward(self, symptom_set, herb_attributes, 
                sh_graph, ss_graph, hh_graph):
        """
        :param symptom_set: 症状集张量 (batch_size, num_symptoms)
        :param herb_attributes: 草药属性矩阵 (num_herbs, attribute_dim)
        :param sh_graph: S-H图数据
        :param ss_graph: S-S图数据
        :param hh_graph: H-H图数据
        :return: 草药推荐概率 (batch_size, num_herbs)
        """
        # 1. 获取初始嵌入
        s_emb = self.symptom_emb(torch.arange(len(self.symptom_emb.weight)).to(symptom_set.device))
        h_emb = self.herb_emb(torch.arange(len(self.herb_emb.weight)).to(symptom_set.device))
        
        # 2. S-H图嵌入
        s_sh_layers = self.s_mugcn(s_emb, sh_graph.edge_index)
        h_sh_layers = self.h_mugcn(h_emb, sh_graph.edge_index)
        
        # 多层信息融合
        s_sh = self.s_fusion(s_sh_layers)
        h_sh = self.h_fusion(h_sh_layers)
        
        # 3. S-S图嵌入
        s_ss = self.s_gcn(s_emb, ss_graph.edge_index)
        
        # 4. H-H图嵌入（知识增强）
        # 连接草药嵌入和属性
        h_with_attr = torch.cat([h_emb, herb_attributes], dim=1)
        h_hh = self.h_kg_gcn(h_with_attr, hh_graph.edge_index)
        
        # 5. 组合不同图的嵌入（使用SUM操作）
        s_combined = s_sh + s_ss
        h_combined = h_sh + h_hh
        
        # 6. 症状集综合嵌入
        # 症状集one-hot编码与嵌入交互
        symptom_set_emb = torch.matmul(symptom_set, s_combined)  # (batch_size, emb_dim)
        symptom_set_emb = self.symptom_set_mlp(symptom_set_emb)
        
        # 7. 草药推荐概率
        # 计算症状集嵌入与所有草药的相似度
        logits = torch.matmul(symptom_set_emb, h_combined.t())  # (batch_size, num_herbs)
        probs = torch.sigmoid(logits)
        
        return probs

# 5. 数据准备函数
def prepare_batch(prescriptions, symptom2id, herb2id, device):
    """准备批数据"""
    batch_symptom_sets = []
    batch_herb_sets = []
    
    for _, row in prescriptions.iterrows():
        # 症状集one-hot编码
        symptoms = row['symptoms'].split(',')
        symptom_vec = torch.zeros(len(symptom2id))
        for s in symptoms:
            symptom_vec[symptom2id[s]] = 1
        batch_symptom_sets.append(symptom_vec)
        
        # 草药集one-hot编码
        herbs = row['herbs'].split(',')
        herb_vec = torch.zeros(len(herb2id))
        for h in herbs:
            herb_vec[herb2id[h]] = 1
        batch_herb_sets.append(herb_vec)
    
    return (
        torch.stack(batch_symptom_sets).to(device),
        torch.stack(batch_herb_sets).to(device)
    )

# 6. 训练函数
def train_model(model, data_loader, kg_data, sh_graph, ss_graph, hh_graph, 
                herb_attributes, device, epochs=200, batch_size=512, lr=0.0003):
    # 划分数据集
    train_data, val_data, test_data = np.split(
        data_loader.prescriptions.sample(frac=1), 
        [int(0.6*len(data_loader.prescriptions)), int(0.8*len(data_loader.prescriptions))]
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    best_val_f1 = 0
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = int(np.ceil(len(train_data) / batch_size))
        
        for i in range(num_batches):
            batch_data = train_data.iloc[i*batch_size:(i+1)*batch_size]
            symptom_set, herb_set = prepare_batch(
                batch_data, data_loader.symptom2id, data_loader.herb2id, device
            )
            
            optimizer.zero_grad()
            
            # 前向传播
            herb_probs = model(
                symptom_set, herb_attributes, 
                sh_graph, ss_graph, hh_graph
            )
            
            # 计算损失
            loss = criterion(herb_probs, herb_set)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 验证
        val_metrics = evaluate_model(model, val_data, data_loader, herb_attributes, 
                                     sh_graph, ss_graph, hh_graph, device, top_k=5)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/num_batches:.4f} | "
              f"Val P@5: {val_metrics['precision']:.4f} | R@5: {val_metrics['recall']:.4f} | "
              f"F1@5: {val_metrics['f1']:.4f}")
        
        # 早停机制
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_kdhr_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 7:
                print("Early stopping triggered")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load("best_kdhr_model.pth"))
    
    # 测试
    test_metrics = evaluate_model(model, test_data, data_loader, herb_attributes, 
                                  sh_graph, ss_graph, hh_graph, device, top_k=5)
    
    print("\nTest Results:")
    print(f"Precision@5: {test_metrics['precision']:.4f}")
    print(f"Recall@5: {test_metrics['recall']:.4f}")
    print(f"F1-Score@5: {test_metrics['f1']:.4f}")
    
    return model

# 7. 评估函数
def evaluate_model(model, data, data_loader, herb_attributes, 
                   sh_graph, ss_graph, hh_graph, device, top_k=5):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        symptom_set, herb_set = prepare_batch(
            data, data_loader.symptom2id, data_loader.herb2id, device
        )
        
        herb_probs = model(
            symptom_set, herb_attributes, 
            sh_graph, ss_graph, hh_graph
        )
        
        # 获取top-k预测
        _, topk_indices = torch.topk(herb_probs, k=top_k, dim=1)
        preds = torch.zeros_like(herb_probs)
        for i in range(len(preds)):
            preds[i, topk_indices[i]] = 1
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(herb_set.cpu().numpy())
    
    # 展平结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算指标
    precision = precision_score(all_targets, all_preds, average='micro')
    recall = recall_score(all_targets, all_preds, average='micro')
    f1 = f1_score(all_targets, all_preds, average='micro')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 8. 主函数
def main():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    CLASSES = CLASSES
    # 加载数据
    data_loader = TCMDataLoader(
        prescription_path="./herb_data.csv",
        kg_path="./herb_data.json"
    )
    prescriptions, kg_data = data_loader.load_data()
    
    # 构建图
    sh_graph, ss_graph, hh_graph = data_loader.build_graphs(
        prescriptions, kg_data, s_s_threshold=5, h_h_threshold=40
    )
    sh_graph = sh_graph.to(device)
    ss_graph = ss_graph.to(device)
    hh_graph = hh_graph.to(device)
    
    # 创建草药属性矩阵
    herb_attributes = data_loader.create_herb_attributes(kg_data).to(device)
    
    # 初始化模型
    model = KDHR(
        num_symptoms=len(data_loader.symptom2id),
        num_herbs=len(data_loader.herb2id),
        attribute_dim=herb_attributes.size(1),
        emb_dim=64,
        hidden_dim=256,
        num_layers=2
    ).to(device)
    
    # 训练模型
    trained_model = train_model(
        model, data_loader, kg_data, sh_graph, ss_graph, hh_graph, 
        herb_attributes, device, epochs=200, batch_size=512, lr=0.0003
    )
    
    # 保存模型
    torch.save(trained_model.state_dict(), "kdhr_final_model.pth")

if __name__ == "__main__":
    main()
