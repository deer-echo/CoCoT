# Qwen2-VL 数据生成系统

基于 Qwen2-VL 模型的视觉问答数据生成系统，用于生成包含边界框(bbox)和推理链的高质量训练数据。

## 📁 项目结构

```
Generate_Data_Qwen2-VL/
├── 🔧 核心脚本
│   ├── generate_bbox_one_agent_qwen.py    # 主要脚本：生成bbox数据
│   └── generate_relation_cycle.py         # 主要脚本：生成推理链数据

├── 📖 文档和配置
│   ├── README.md                          # 📖 本文档
│   ├── requirements.txt                   # pip依赖包列表
│   ├── current_requirements.txt           # 当前环境完整包列表
│   ├── environment.yml                    # conda环境配置
│   ├── Dockerfile                         # Docker镜像配置
│   └── docker-compose.yml                # Docker编排配置
├── 🤖 模型文件
│   └── Qwen2-VL-7B-Instruct/             # Qwen2-VL模型文件
├── 📊 数据集
│   ├── dataset_with_GT/                   # 原始数据集（带Ground Truth）
│   │   ├── Docvqa/DocVQA_complex_4plus.json                       # DocVQA数据集
│   │   ├── GQA/GQA_merged_complex_6plus.json                      # GQA数据集
│   │   ├── InfoVQA/InfoVQA_complex_4plus_parallel.json            # InfoVQA数据集
│   │   ├── TextVQA/TextVQA_complex_3plus_parallel.json           # TextVQA数据集
│   │   ├── VQAv2/VQA_v2_train_merged.json                        # VQAv2数据集
│   │   └── Visual7W/Visual7W_complex_3plus_parallel.json        # Visual7W数据集
│   └── playground/                        # 数据存储目录
│       └── data/                         # 各种中间和最终数据
│           └── cot/                      # 图像数据按数据集分类
│              ├── docvqa/ffbf0023_4.png...              # DocVQA图像
│              ├── gqa/1.jpg...                            # GQA图像
│              ├── textvqa/0a0bc91825468c45.jpg             # TextVQA图像
│              ├── coco/COCO_train2014_000000000009.jpg...       # COCO图像(VQAv2)
│              ├── v7w/v7w_1.jpg...                 # Visual7W图像
│              └── infographicsvqa/10002.jpeg...    # InfoVQA图像
│          
├── 📦 生成结果
│   ├── images_bbox/                       # 生成的bbox数据
│   │   ├── DocVQA_complex_one_agent.json
│   │   ├── GQA_complex_one_agent.json
│   │   ├── InfoVQA_complex_one_agent.json
│   │   ├── TextVQA_complex_one_agent.json
│   │   ├── VQAv2_complex_one_agent.json
│   │   └── Visual7W_complex_one_agent.json
│   └── reasoning_chains/                  # 生成的推理链数据
│       ├── DocVQA_complex_reasoning_chains_one_agent.json
│       ├── GQA_complex_reasoning_chains_one_agent.json
│       ├── InfoVQA_complex_reasoning_chains_one_agent.json
│       ├── TextVQA_complex_reasoning_chains_one_agent.json
│       ├── VQAv2_complex_reasoning_chains_one_agent.json
│       └── Visual7W_complex_reasoning_chains_one_agent.json
└── 🗂️ 其他文件
    ├── __pycache__/                       # Python缓存文件
    └── *.log                              # 运行日志文件
```

## 📋 文件功能说明

### 🔧 核心脚本
- **`generate_bbox_one_agent_qwen.py`**: 使用Qwen2-VL生成边界框数据，支持4层生成策略
- **`generate_relation_cycle.py`**: 基于bbox数据构建推理链，支持单步和多步推理

### 📖 配置文件
- **`requirements.txt`**: 精心整理的pip依赖包列表，包含版本固定
- **`environment.yml`**: conda环境配置文件，支持一键创建环境
- **`Dockerfile`**: Docker镜像配置，支持容器化部署
- **`docker-compose.yml`**: Docker编排配置，简化容器使用

### 📊 数据目录
- **`dataset_with_GT/`**: 原始数据集，包含问题、答案和图像路径
- **`playground/data/cot/`**: 图像文件，按数据集分类存储
- **`images_bbox/`**: 生成的bbox数据，包含边界框坐标和描述
- **`reasoning_chains/`**: 生成的推理链数据，包含推理步骤和逻辑关系

## 🔧 环境配置

### 数据图片下载
下载对应数据集并配置到playground/data/cot的对应路径下
- **COCO**: [images](http://images.cocodataset.org/zips/train2014.zip) (82,783 images)
- **DocVQA**: [homepage](https://www.docvqa.org/datasets/docvqa) (10,196 images)
- **TextVQA**: [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) (25,119 images)
- **Visual7W**: [repo](https://github.com/yukezhu/visual7w-toolkit) (47,300 images)
- **GQA**: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) (148,854 images)
- **InfographicVQA**: [homepage](https://www.docvqa.org/datasets/infographicvqa) (5,485 images)

### 🌟 快速环境配置（推荐）

#### 方法1: 使用conda环境文件（最简单）
```bash
# 1. 克隆或下载项目
git clone <repository-url>
cd Generate_Data_Qwen2-VL

# 2. 创建conda环境（自动安装所有依赖）
conda env create -f environment.yml

# 3. 激活环境
conda activate qwen2vl

```

#### 方法2: 手动创建环境（更灵活）
```bash
# 1. 创建基础环境
conda create -n qwen2vl python=3.9 -y
conda activate qwen2vl

# 2. 安装PyTorch（根据你的CUDA版本选择）
# CUDA 12.6 (当前环境)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 

# 3. 安装其他依赖
pip install -r requirements.txt

```


### 📥 模型下载

#### 🤖 Qwen2-VL-7B-Instruct 模型

**自动下载脚本（推荐）：**
```bash
# 从ModelScope下载（国内用户推荐，速度快）
python download_model.py --source modelscope

# 从HuggingFace下载（国外用户）
python download_model.py --source huggingface

# 指定下载目录
python download_model.py --model-dir ./models/Qwen2-VL-7B-Instruct --source modelscope
```

**手动下载方法：**
```bash
# 方法1: ModelScope (国内推荐，约15GB)
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2-VL-7B-Instruct',
                 local_dir='./Qwen2-VL-7B-Instruct',
                 cache_dir='./cache')
"

# 方法2: HuggingFace (需要良好的网络)
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# 方法3: 使用HuggingFace Hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-7B-Instruct',
                 local_dir='./Qwen2-VL-7B-Instruct')
"
```

**模型文件结构验证：**
```bash
# 检查模型文件完整性
ls -la Qwen2-VL-7B-Instruct/
# 应该包含以下关键文件：
# - config.json                    # 模型配置
# - model-00001-of-00005.safetensors  # 模型权重文件
# - model-00002-of-00005.safetensors
# - model-00003-of-00005.safetensors
# - model-00004-of-00005.safetensors
# - model-00005-of-00005.safetensors
# - model.safetensors.index.json   # 权重索引
# - tokenizer.json                 # 分词器
# - preprocessor_config.json       # 预处理配置
# - generation_config.json         # 生成配置
```

## 🚀 使用方法


🎯 开始处理 DocVQA 数据集...
```

### 🔧 手动执行方式（高级用户）

#### 第一步：生成Bbox数据

**脚本功能：** `generate_bbox_one_agent_qwen.py`
- 使用Qwen2-VL模型分析图像和问题
- 生成精确的边界框坐标和描述
- 支持4层生成策略确保数据质量

```bash
conda activate qwen2vl
python generate_bbox_one_agent_qwen.py
```

**执行流程详解：**

1. **🎯 GPU选择界面**
   ```
   🚀 Qwen2-VL Bbox生成器
   ==================================================

   🎯 检测到 4 个GPU，选择使用方式:
      GPU 0: NVIDIA GeForce RTX 4090
         总内存: 24.0GB
         已使用: 2.1GB
         可用: 21.9GB
         📊 使用率: 8.8%

   选择使用方式:
      0. 使用所有GPU
      1. 使用两个GPU (推荐，更快)
      2. 使用单个GPU

   请选择 (0/1/2): 2
   ```

2. **📊 数据集选择**
   - DocVQA: 文档问答 (~12K复杂问题)
   - InfoVQA: 信息图表问答 (~22K复杂问题)
   - TextVQA: 文本问答 (~13K复杂问题)
   - Visual7W: 视觉问答 (~18K复杂问题)
   - GQA: 场景图问答 (~153K复杂问题)
   - VQAv2: 视觉问答v2 (~35K复杂问题)

3. **🔄 4层Bbox生成策略**
   - **Layer 1 (最高质量)**: Qwen2-VL识别 + OCR精确定位
     - 成功率: 60-80%
     - 精度最高，结合视觉理解和文本定位
   - **Layer 2 (中等质量)**: 仅Qwen2-VL识别
     - 成功率: 15-25%
     - OCR失败时的视觉方案
   - **Layer 3 (保底方案)**: OCR + 关键词匹配
     - 成功率: 5-15%
     - Qwen2-VL失效时的文本匹配
   - **Layer 4 (最后手段)**: 纯算法保底
     - 成功率: <5%
     - 确保每个问题都有输出

#### 第二步：生成推理链数据

**脚本功能：** `generate_relation_cycle.py`
- 基于bbox数据构建推理链
- 支持单步和多步推理
- 自动判断推理类型（顺序/并列）

```bash
conda activate qwen2vl
python generate_relation_cycle.py
```

**推理链生成流程详解：**

1. **🎯 推理模式选择**
   ```
   🎯 选择bbox生成模式:
     1. 仅生成单bbox推理链 (bbox_count == 1)
        - 适用于简单的直接回答问题
     2. 仅生成多bbox推理链 (bbox_count > 1)
        - 适用于复杂的多步推理问题
     3. 自动模式 (处理所有bbox数量)
        - 处理所有类型的问题

   请选择模式 (1/2/3): 3
   ```

2. **🔗 推理链类型**
   - **单步推理**: 直接回答类问题
     ```
     问题: "What is the contact person name?"
     推理: "P.CARTER" directly answers the question about contact person
     ```

   - **顺序推理**: 步骤依赖的问题 (A→B→C)
     ```
     问题: "What is the process to submit application?"
     推理链: Step1→Step2→Step3→Final Answer
     ```

   - **并列推理**: 并行证据的问题 (A→B; A→C)
     ```
     问题: "Which country has the highest GDP?"
     推理链: Country1_GDP; Country2_GDP; Country3_GDP → Comparison
     ```

3. **🤖 多轮Qwen分析**
   - 每轮分析选择最相关的bbox
   - 构建推理关系链条
   - 自动判断是否需要继续推理

### 📊 结果分析和质量评估

**数据质量分析脚本：**
```bash
python analyze_results.py
```

**分析内容包括：**
- 📈 **Bbox生成质量分布**
  - Layer 1 (最佳): 60-80%
  - Layer 2 (良好): 15-25%
  - Layer 3 (可用): 5-15%
  - Layer 4 (保底): <5%

- 🔗 **推理链类型统计**
  - 单步推理比例
  - 多步推理比例
  - 平均推理步数
  - 推理链完整性

- 📊 **数据集完整性检查**
  - 处理成功率
  - 错误类型分布
  - 数据格式验证

**示例输出：**
```
📊 DocVQA数据集分析报告
================================
总样本数: 11,995
处理成功: 11,892 (99.1%)

Bbox生成质量分布:
├── Layer 1 (混合方案): 8,934 (75.1%) ✅
├── Layer 2 (纯视觉): 2,156 (18.1%) ✅
├── Layer 3 (OCR保底): 658 (5.5%) ⚠️
└── Layer 4 (算法保底): 144 (1.2%) ⚠️

推理链类型分布:
├── 单步推理: 7,234 (60.8%)
├── 多步推理: 4,658 (39.2%)
└── 平均步数: 1.6步
```

### 💡 数据使用示例

**学习如何使用生成的数据：**
```bash
python example_usage.py
```

### 📁 数据集目录结构

```
dataset_with_GT/                    # 原始数据集
├── Docvqa/
│   └── DocVQA_complex_4plus.json  # 4步以上复杂问题
├── GQA/
│   └── GQA_merged_complex_6plus.json  # 6步以上复杂问题
├── InfoVQA/
│   └── InfoVQA_complex_4plus_parallel.json
├── TextVQA/
│   └── TextVQA_complex_3plus_parallel.json
├── VQAv2/
│   └── VQAv2_complex_5plus_parallel.json
└── Visual7W/
    └── Visual7W_complex_3plus_parallel.json

playground/data/cot/                # 图像文件
├── docvqa/          # DocVQA图像 (.png)
├── gqa/             # GQA图像 (.jpg)
├── textvqa/         # TextVQA图像 (.jpg)
├── coco/            # COCO图像 (.jpg) - VQAv2使用
├── v7w/             # Visual7W图像 (.jpg)
└── infographicsvqa/ # InfoVQA图像 (.jpeg)
```

### 📋 数据格式详解

#### 🔍 Bbox数据格式 (images_bbox/)

```json
{
  "question_id": "DocVQA_338",                    // 唯一问题ID
  "question": "what is the contact person name mentioned in letter?",
  "image_name": "xnbl0037_1",                    // 图像文件名（不含扩展名）
  "answers": ["P. Carter", "p. carter"],         // 标准答案列表
  "bbox_analysis": {
    "relevant_elements": [                       // 相关区域列表
      {
        "description": "Contact person name",   // 区域描述
        "bbox": [0.33, 0.31, 0.41, 0.34],     // 归一化坐标 [x1,y1,x2,y2]
        "selection_reason": "Contains the contact person information",
        "content_relation": "This region shows the name P.CARTER which directly answers the question"
      }
    ],
    "generation_method": "hybrid_qwen2vl_ocr",   // 生成方法
    "generation_layer": 1,                       // 生成层级 (1-4)
    "generation_description": "Generated by hybrid method: Qwen2-VL + OCR precise localization"
  }
}
```


## 🎯 生成结果详解

### 📦 Bbox生成结果分层

#### Layer 1: 混合方案 (最高质量 60-80%)
- **方法**: Qwen2-VL视觉理解 + OCR精确定位
- **优势**: 结合视觉语义理解和文本精确定位
- **适用**: 包含文本的复杂视觉问题
- **示例**:
  ```json
  {
    "generation_method": "hybrid_qwen2vl_ocr",
    "generation_layer": 1,
    "bbox": [0.245, 0.156, 0.387, 0.189],  // 精确的文本边界
    "match_info": {
      "ocr_confidence": 0.95,
      "text_match_score": 0.87
    }
  }
  ```

#### Layer 2: 纯视觉方案 (中等质量 15-25%)
- **方法**: 仅使用Qwen2-VL进行区域识别
- **优势**: 处理OCR无法识别的视觉元素
- **适用**: 图像、图标、复杂布局
- **示例**:
  ```json
  {
    "generation_method": "qwen2vl_only",
    "generation_layer": 2,
    "bbox": [0.1, 0.2, 0.4, 0.6],  // 视觉区域边界
    "description": "Chart showing sales data"
  }
  ```

#### Layer 3: OCR保底方案 (可用质量 5-15%)
- **方法**: OCR文本检测 + 关键词匹配
- **优势**: Qwen2-VL失效时的文本方案
- **适用**: 简单文本问题
- **示例**:
  ```json
  {
    "generation_method": "emergency_ocr",
    "generation_layer": 3,
    "bbox": [0.3, 0.4, 0.5, 0.45],
    "relevance": "Contains keyword 'total' relevant to the question"
  }
  ```

#### Layer 4: 算法保底 (保底质量 <5%)
- **方法**: 基于问题关键词的算法生成
- **优势**: 确保每个问题都有输出
- **适用**: 所有其他方法都失败的情况
- **示例**:
  ```json
  {
    "generation_method": "basic_fallback",
    "generation_layer": 4,
    "bbox": [0.05, 0.1, 0.3, 0.15],  // 假设位置
    "content": "Text containing 'contact'"
  }
  ```


