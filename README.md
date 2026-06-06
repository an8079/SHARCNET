# SHARCNet (HyperDNE-RC²)

**深度融合超图语义对比与Ricci曲率增强的蛋白质网络嵌入框架**

SHARCNet (HyperDNE-RC²) 是一个基于深度网络嵌入的蛋白质-蛋白质相互作用 (PPI) 网络分析框架。它融合了图卷积网络 (GCN)、超图神经网络 (HGNN)、Ricci曲率图增强以及多种对比学习策略，用于生成高质量的蛋白质嵌入表示。

## 核心特性

- **双编码器架构**: GCN 编码原始PPI图拓扑 + HGNN 编码基于团簇的超图高阶结构
- **Ricci曲率增强 (RCGA)**: 利用 Ollivier-Ricci 曲率移除噪声边，构建增强视图
- **多重对比学习**:
  - TCL (拓扑对比损失): 原始图 vs Ricci增强图
  - HSCL (超图语义对比损失): 基于软聚类的正/负样本对
  - 视图对齐损失: 超图视图与RC增强视图对齐
- **结构与特征重构**: 双解码器确保嵌入保留原始信息
- **多种特征源支持**:
  - **ESM-2 蛋白质语言模型特征**: 使用 `facebook/esm2_t33_650M_UR50D` 生成节点初始特征
  - **InterPro 域特征**: 使用 InterPro 功能域标注作为节点特征 (DPFunc 数据集)
- **自动模型下载**: 支持 HuggingFace 和 ModelScope 双通道自动下载 ESM 模型

## 项目结构

```
sharc/
├── code/                      # 源代码
│   ├── main.py                # PPI数据集训练入口
│   ├── main_dpfunc.py         # DPFunc 功能预测数据集训练入口
│   ├── roc.py                 # 带ROC/PR曲线输出的训练入口
│   ├── model.py               # HyperDNE-RC² 模型定义
│   ├── dataset.py             # PPI数据集加载与ESM特征生成
│   ├── dataset_dpfunc.py      # DPFunc数据集加载 (InterPro域特征)
│   ├── parser.py              # 命令行参数定义
│   ├── utils.py               # 工具函数 (损失函数、图操作等)
│   └── sensitivity_analysis.py # 超参数敏感性分析脚本
├── data/                      # 数据集目录
│   ├── c_elegans/             # C. elegans PPI网络
│   │   ├── edge_list.csv      # 边列表 (source, target)
│   │   └── protein_seq.tsv    # 蛋白质序列
│   ├── HuRI/                  # 人类相互作用组
│   ├── yeast/                 # 酵母PPI网络
│   ├── hy/                    # 自定义数据集
│   └── data_dpfunc/           # 蛋白质功能预测数据集
│       ├── id_map.pkl         # 蛋白质ID映射 (60K+)
│       ├── all_protein_interpros.pkl  # InterPro域标注
│       ├── inter_idx.pkl      # InterPro域索引
│       └── bp/cc/mf_*.txt     # GO功能注释 (BP/CC/MF)
├── result/                    # 评估结果输出
├── requirements.txt           # Python依赖
└── README.md
```

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/SHARCNet.git
cd SHARCNet
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- PyTorch >= 2.0
- Transformers >= 4.30
- ModelScope >= 1.9 (用于国内镜像下载)
- NetworkX >= 3.0
- scikit-learn >= 1.3
- GraphRicciCurvature >= 0.5.3

### 3. ESM 模型下载 (首次运行自动完成)

首次运行时，程序会自动下载 ESM-2 蛋白质语言模型 (~2.5GB):
- **国内用户**: 如果 HuggingFace 不可用，程序会自动切换到 ModelScope 镜像下载
- **手动下载**: 也可以预先下载模型到本地，然后通过 `--esm_model_name` 参数指定本地路径

```bash
# 方式1: 自动下载 (默认)
# 程序首次运行会自动下载 facebook/esm2_t33_650M_UR50D

# 方式2: 手动指定本地模型路径
python main.py --esm_model_name /path/to/local/esm2_t33_650M_UR50D
```

## 使用方法

### 基本运行

```bash
cd code
python main.py --dataset_name c_elegans
```

### 使用 ROC 曲线输出

```bash
cd code
python roc.py --dataset_name HuRI
```

### 超参数敏感性分析

```bash
cd code
python sensitivity_analysis.py --base_data_path ../data
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset_name` | `c_elegans` | 数据集名称 (c_elegans, HuRI, yeast, hy) |
| `--data_path` | `../data` | 数据集根目录 (相对于 code/) |
| `--epochs` | `10` | 训练轮数 |
| `--learning_rate` | `1e-3` | 学习率 |
| `--embedding_dim` | `128` | 蛋白质嵌入维度 |
| `--esm_model_name` | `facebook/esm2_t33_650M_UR50D` | ESM模型ID或本地路径 |
| `--device` | 自动检测 | 计算设备 (cuda / cpu) |
| `--use_ricci_augmentation` | `True` | 是否使用Ricci曲率增强 |
| `--link_pred_n_trials` | `5` | 链接预测试验次数 |

完整参数列表请运行 `python main.py --help`。

### DPFunc 功能预测数据集

`data_dpfunc` 是一个大规模蛋白质功能预测数据集，包含 **60,254 个蛋白质**和 **26,203 个 InterPro 功能域**，覆盖三个 GO 命名空间:

| 命名空间 | 含义 | 训练蛋白数 | 验证蛋白数 | 测试蛋白数 |
|----------|------|-----------|-----------|-----------|
| `bp` | Biological Process | 47,140 | 731 | 1,312 |
| `cc` | Cellular Component | 41,539 | 633 | 1,005 |
| `mf` | Molecular Function | 33,339 | 422 | 702 |

**特征**: 使用 InterPro 域二值向量 (26,203 维) 通过 TruncatedSVD 降维到 1,280 维，替代 ESM 序列特征。

**图构建**: 在 SVD 特征上使用 k-NN (k=10, cosine) 构建蛋白质相似图。

```bash
# DPFunc 数据集训练 (默认 BP 命名空间)
python main_dpfunc.py --dataset_name data_dpfunc

# 指定 CC 或 MF 命名空间
python main_dpfunc.py --dataset_name data_dpfunc --dpfunc_namespace cc
python main_dpfunc.py --dataset_name data_dpfunc --dpfunc_namespace mf

# 调整图构建参数
python main_dpfunc.py --dataset_name data_dpfunc \
    --dpfunc_knn_k 15 \
    --dpfunc_svd_dim 1024 \
    --min_clique_size 3
```

**DPFunc 专用参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dpfunc_knn_k` | `10` | k-NN 相似图每个节点的邻居数 |
| `--dpfunc_svd_dim` | `1280` | InterPro 特征 SVD 降维目标维度 |
| `--dpfunc_namespace` | `bp` | GO 功能注释命名空间 (bp/cc/mf) |
| `--min_clique_size` | `2` | DPFunc 默认使用 2 (每边即超边) |

### 使用自定义数据集

在 `data/` 下创建新目录，包含以下文件:
1. `edge_list.csv` — PPI网络边列表，包含 `source` 和 `target` 列
2. `protein_seq.tsv` — 蛋白质序列表，包含 `Entry` (或 `VEuPathDB`) 和 `Sequence` 列

```bash
python main.py --dataset_name your_dataset_name
```

## 评估指标

模型通过链接预测任务评估，报告以下指标 (5次试验的均值 ± 标准差):
- **AUC** (ROC曲线下面积)
- **AUPR** (PR曲线下面积)
- **F1-Score**
- **Accuracy**

## 数据集

### PPI 网络数据集

| 数据集 | 物种 | 节点数 | 边数 |
|--------|------|--------|------|
| c_elegans | 秀丽隐杆线虫 | ~3,500 | ~8,000 |
| HuRI | 人类 | ~8,000 | ~50,000 |
| yeast | 酿酒酵母 | ~5,000 | ~30,000 |

### DPFunc 功能预测数据集

| 属性 | 值 |
|------|-----|
| 蛋白质数 | 60,254 |
| InterPro 域数 | 26,203 |
| GO 命名空间 | BP, CC, MF |
| 特征源 | InterPro 域 + SVD |
| 图类型 | k-NN 相似图 (cosine) |

## 引用

如果您使用了本框架，请引用相关工作。

## License

本项目采用 MIT License 开源。详见 [LICENSE](LICENSE)。
