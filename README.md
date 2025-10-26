# CoCoT Data Generation System
This is a data generation system for Collaborative Cross-modal Chain-of-Thought (CoCoT) based on Qwen2-VL, designed to produce high-quality training data with dynamic multi-region grounding and relation-aware reasoning chains.

🔍 Related Research: This system implements the methodology described in the paper "Watch Wider and Think Deeper: Collaborative Cross-modal Chain-of-Thought for Complex Visual Reasoning", addressing limitations of single-region CoT methods through multi-region collaboration.

## 📁 Project Structure

```
Generate_Data_Qwen2-VL/
├── 🔧 Core Scripts
│   ├── generate_bbox_one_agent_qwen.py    # Main script: generates bbox data
│   └── generate_relation_cycle.py         # Main script: generates reasoning chain data

├── 📖 Documentation & Configuration
│   ├── README.md                          
│   ├── requirements.txt                   # Pip dependency list
│   ├── current_requirements.txt           # Current environment complete package list
│   ├── environment.yml                    # Conda environment configuration
│   ├── Dockerfile                         # Docker image configuration
│   └── docker-compose.yml                # Docker orchestration configuration

├── 🤖 Model Files
│   └── Qwen2-VL-7B-Instruct/             # Qwen2-VL model files

├── 📊 Datasets
│   ├── dataset_with_GT/                   # Original datasets (with Ground Truth)
│   │   ├── Docvqa/DocVQA_complex_4plus.json                       # DocVQA dataset
│   │   ├── GQA/GQA_merged_complex_6plus.json                      # GQA dataset
│   │   ├── InfoVQA/InfoVQA_complex_4plus_parallel.json            # InfoVQA dataset
│   │   ├── TextVQA/TextVQA_complex_3plus_parallel.json           # TextVQA dataset
│   │   ├── VQAv2/VQA_v2_train_merged.json                        # VQAv2 dataset
│   │   └── Visual7W/Visual7W_complex_3plus_parallel.json        # Visual7W dataset
│   └── playground/                        # Data storage directory
│       └── data/                         # Various intermediate and final data
│           └── cot/                      # Image data categorized by dataset
│              ├── docvqa/ffbf0023_4.png...              # DocVQA images
│              ├── gqa/1.jpg...                            # GQA images
│              ├── textvqa/0a0bc91825468c45.jpg...             # TextVQA images
│              ├── coco/COCO_train2014_000000000009.jpg...       # COCO images (VQAv2)
│              ├── v7w/v7w_1.jpg...                 # Visual7W images
│              └── infographicsvqa/10002.jpeg...    # InfoVQA images
│          
├── 📦 Generated Results
│   ├── images_bbox/                       # Generated bbox data
│   │   ├── DocVQA_complex_one_agent.json
│   │   ├── GQA_complex_one_agent.json
│   │   ├── InfoVQA_complex_one_agent.json
│   │   ├── TextVQA_complex_one_agent.json
│   │   ├── VQAv2_complex_one_agent.json
│   │   └── Visual7W_complex_one_agent.json
│   └── reasoning_chains/                  # Generated reasoning chain data
│       ├── DocVQA_complex_reasoning_chains_one_agent.json
│       ├── GQA_complex_reasoning_chains_one_agent.json
│       ├── InfoVQA_complex_reasoning_chains_one_agent.json
│       ├── TextVQA_complex_reasoning_chains_one_agent.json
│       ├── VQAv2_complex_reasoning_chains_one_agent.json
│       └── Visual7W_complex_reasoning_chains_one_agent.json
└── 🗂️ Other Files
    ├── __pycache__/                       # Python cache files
    └── *.log                              # Runtime log files
```

## 📋 File Function Description

### 🔧 Core Scripts
- **`generate_bbox_one_agent_qwen.py`**: Uses Qwen2-VL to generate bounding box data with a 4-layer generation strategy
- **`generate_relation_cycle.py`**: Builds reasoning chains based on bbox data, supporting single-step and multi-step reasoning

### 📖 Configuration Files
- **`requirements.txt`**: Carefully curated pip dependency list with version pinning
- **`environment.yml`**: Conda environment configuration file for one-click environment creation
- **`Dockerfile`**: Docker image configuration supporting containerized deployment
- **`docker-compose.yml`**: Docker orchestration configuration for simplified container usage

### 📊 Data Directories
- **`dataset_with_GT/`**: Original datasets containing questions, answers, and image paths
- **`playground/data/cot/`**: Image files, categorized and stored by dataset
- **`images_bbox/`**: Generated bbox data containing bounding box coordinates and descriptions
- **`reasoning_chains/`**: Generated reasoning chain data containing reasoning steps and logical relationships

## 🔧 Environment Setup

### Image Data Download
Download corresponding datasets and configure them in the appropriate paths under `playground/data/cot`:
- **COCO**: [images](http://images.cocodataset.org/zips/train2014.zip) (82,783 images)
- **DocVQA**: [homepage](https://www.docvqa.org/datasets/docvqa) (10,196 images)
- **TextVQA**: [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) (25,119 images)
- **Visual7W**: [repo](https://github.com/yukezhu/visual7w-toolkit) (47,300 images)
- **GQA**: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip) (148,854 images)
- **InfographicVQA**: [homepage](https://www.docvqa.org/datasets/infographicvqa) (5,485 images)

### 🌟 Quick Environment Setup (Recommended)

#### Method 1: Using Conda Environment File (Simplest)
```bash
# 1. Clone or download the project
git clone <repository-url>
cd Generate_Data_Qwen2-VL

# 2. Create conda environment (automatically installs all dependencies)
conda env create -f environment.yml

# 3. Activate environment
conda activate qwen2vl
```

#### Method 2: Manual Environment Creation (More Flexible)
```bash
# 1. Create base environment
conda create -n qwen2vl python=3.9 -y
conda activate qwen2vl

# 2. Install PyTorch (choose based on your CUDA version)
# CUDA 12.6 (current environment)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 

# 3. Install other dependencies
pip install -r requirements.txt
```

### 📥 Model Download

#### 🤖 Qwen2-VL-7B-Instruct Model

**Automatic Download Script (Recommended):**
```bash
# Download from ModelScope (recommended for Chinese users, faster)
python download_model.py --source modelscope

# Download from HuggingFace (international users)
python download_model.py --source huggingface

# Specify download directory
python download_model.py --model-dir ./models/Qwen2-VL-7B-Instruct --source modelscope
```

**Manual Download Methods:**
```bash
# Method 1: ModelScope (recommended in China, ~15GB)
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2-VL-7B-Instruct',
                 local_dir='./Qwen2-VL-7B-Instruct',
                 cache_dir='./cache')
"

# Method 2: HuggingFace (requires good network connection)
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

# Method 3: Using HuggingFace Hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-7B-Instruct',
                 local_dir='./Qwen2-VL-7B-Instruct')
"
```

**Model File Structure Verification:**
```bash
# Check model file integrity
ls -la Qwen2-VL-7B-Instruct/
# Should contain these key files:
# - config.json                    # Model configuration
# - model-00001-of-00005.safetensors  # Model weight files
# - model-00002-of-00005.safetensors
# - model-00003-of-00005.safetensors
# - model-00004-of-00005.safetensors
# - model-00005-of-00005.safetensors
# - model.safetensors.index.json   # Weight index
# - tokenizer.json                 # Tokenizer
# - preprocessor_config.json       # Preprocessor configuration
# - generation_config.json         # Generation configuration
```

## 🚀 Usage

### 🎯 Automated Execution (Recommended for Beginners)

**One-click execution script:**
```bash
chmod +x run_all.sh
./run_all.sh
```

**Execution process:**
1. ✅ Environment check and dependency verification
2. 🤖 Automatic model download and configuration
3. 📊 Sequential processing of six datasets
4. 📈 Real-time progress display and quality statistics

**Example output:**
```
🚀 Starting Qwen2-VL Data Generation System...
✅ Environment check passed
🤖 Model loaded successfully: Qwen2-VL-7B-Instruct
📊 Starting to process 6 datasets...

🎯 Starting DocVQA dataset processing...
```

### 🔧 Manual Execution (Advanced Users)

#### Step 1: Generate Bbox Data

**Script function:** `generate_bbox_one_agent_qwen.py`
- Uses Qwen2-VL model to analyze images and questions
- Generates precise bounding box coordinates and descriptions
- Supports 4-layer generation strategy to ensure data quality

```bash
conda activate qwen2vl
python generate_bbox_one_agent_qwen.py
```

**Detailed execution process:**

1. **🎯 GPU Selection Interface**
   ```
   🚀 Qwen2-VL Bbox Generator
   ==================================================

   🎯 Detected 4 GPUs, select usage mode:
      GPU 0: NVIDIA GeForce RTX 4090
         Total memory: 24.0GB
         Used: 2.1GB
         Available: 21.9GB
         📊 Usage: 8.8%

   Select usage mode:
      0. Use all GPUs
      1. Use two GPUs (recommended, faster)
      2. Use single GPU

   Please select (0/1/2): 2
   ```

2. **📊 Dataset Selection**
   - DocVQA: Document question answering (~12K complex questions)
   - InfoVQA: Infographic question answering (~22K complex questions)
   - TextVQA: Text-based question answering (~13K complex questions)
   - Visual7W: Visual question answering (~18K complex questions)
   - GQA: Scene graph question answering (~153K complex questions)
   - VQAv2: Visual question answering v2 (~35K complex questions)

3. **🔄 4-Layer Bbox Generation Strategy**
   - **Layer 1 (Highest Quality)**: Qwen2-VL recognition + OCR precise localization
     - Success rate: 60-80%
     - Highest precision, combines visual understanding and text localization
   - **Layer 2 (Medium Quality)**: Qwen2-VL recognition only
     - Success rate: 15-25%
     - Visual solution when OCR fails
   - **Layer 3 (Fallback Solution)**: OCR + keyword matching
     - Success rate: 5-15%
     - Text matching when Qwen2-VL fails
   - **Layer 4 (Last Resort)**: Pure algorithmic fallback
     - Success rate: <5%
     - Ensures output for every question

#### Step 2: Generate Reasoning Chain Data

**Script function:** `generate_relation_cycle.py`
- Builds reasoning chains based on bbox data
- Supports single-step and multi-step reasoning
- Automatically determines reasoning type (sequential/parallel)

```bash
conda activate qwen2vl
python generate_relation_cycle.py
```

**Reasoning Chain Generation Process:**

1. **🎯 Reasoning Mode Selection**
   ```
   🎯 Select bbox generation mode:
     1. Generate single bbox reasoning chains only (bbox_count == 1)
        - Suitable for simple direct-answer questions
     2. Generate multi-bbox reasoning chains only (bbox_count > 1)
        - Suitable for complex multi-step reasoning questions
     3. Automatic mode (process all bbox counts)
        - Processes all question types

   Please select mode (1/2/3): 3
   ```

2. **🔗 Reasoning Chain Types**
   - **Single-step reasoning**: Direct answer questions
     ```
     Question: "What is the contact person name?"
     Reasoning: "P.CARTER" directly answers the question about contact person
     ```

   - **Sequential reasoning**: Step-dependent questions (A→B→C)
     ```
     Question: "What is the process to submit application?"
     Reasoning chain: Step1→Step2→Step3→Final Answer
     ```

   - **Parallel reasoning**: Questions requiring parallel evidence (A→B; A→C)
     ```
     Question: "Which country has the highest GDP?"
     Reasoning chain: Country1_GDP; Country2_GDP; Country3_GDP → Comparison
     ```

3. **🤖 Multi-round Qwen Analysis**
   - Each round selects the most relevant bbox
   - Builds reasoning relationship chains
   - Automatically determines if further reasoning is needed

### 📊 Results Analysis and Quality Assessment

**Data quality analysis script:**
```bash
python analyze_results.py
```

**Analysis includes:**
- 📈 **Bbox generation quality distribution**
  - Layer 1 (Best): 60-80%
  - Layer 2 (Good): 15-25%
  - Layer 3 (Usable): 5-15%
  - Layer 4 (Fallback): <5%

- 🔗 **Reasoning chain type statistics**
  - Single-step reasoning ratio
  - Multi-step reasoning ratio
  - Average reasoning steps
  - Reasoning chain completeness

- 📊 **Dataset completeness check**
  - Processing success rate
  - Error type distribution
  - Data format validation

**Example output:**
```
📊 DocVQA Dataset Analysis Report
================================
Total samples: 11,995
Successfully processed: 11,892 (99.1%)

Bbox Generation Quality Distribution:
├── Layer 1 (Hybrid): 8,934 (75.1%) ✅
├── Layer 2 (Vision-only): 2,156 (18.1%) ✅
├── Layer 3 (OCR fallback): 658 (5.5%) ⚠️
└── Layer 4 (Algorithm fallback): 144 (1.2%) ⚠️

Reasoning Chain Type Distribution:
├── Single-step reasoning: 7,234 (60.8%)
├── Multi-step reasoning: 4,658 (39.2%)
└── Average steps: 1.6 steps
```

### 💡 Data Usage Example

**Learn how to use the generated data:**
```bash
python example_usage.py
```

### 📁 Dataset Directory Structure

```
dataset_with_GT/                    # Original datasets
├── Docvqa/
│   └── DocVQA_complex_4plus.json  # 4+ step complex questions
├── GQA/
│   └── GQA_merged_complex_6plus.json  # 6+ step complex questions
├── InfoVQA/
│   └── InfoVQA_complex_4plus_parallel.json
├── TextVQA/
│   └── TextVQA_complex_3plus_parallel.json
├── VQAv2/
│   └── VQAv2_complex_5plus_parallel.json
└── Visual7W/
    └── Visual7W_complex_3plus_parallel.json

playground/data/cot/                # Image files
├── docvqa/          # DocVQA images (.png)
├── gqa/             # GQA images (.jpg)
├── textvqa/         # TextVQA images (.jpg)
├── coco/            # COCO images (.jpg) - used by VQAv2
├── v7w/             # Visual7W images (.jpg)
└── infographicsvqa/ # InfoVQA images (.jpeg)
```

### 📋 Data Format Details

#### 🔍 Bbox Data Format (images_bbox/)

```json
{
  "question_id": "DocVQA_338",                    // Unique question ID
  "question": "what is the contact person name mentioned in letter?",
  "image_name": "xnbl0037_1",                    // Image filename (without extension)
  "answers": ["P. Carter", "p. carter"],         // Standard answer list
  "bbox_analysis": {
    "relevant_elements": [                       // Relevant regions list
      {
        "description": "Contact person name",   // Region description
        "bbox": [0.33, 0.31, 0.41, 0.34],     // Normalized coordinates [x1,y1,x2,y2]
        "selection_reason": "Contains the contact person information",
        "content_relation": "This region shows the name P.CARTER which directly answers the question"
      }
    ],
    "generation_method": "hybrid_qwen2vl_ocr",   // Generation method
    "generation_layer": 1,                       // Generation layer (1-4)
    "generation_description": "Generated by hybrid method: Qwen2-VL + OCR precise localization"
  }
}
```

## 🎯 Generated Results Details

### 📦 Bbox Generation Results Layering

#### Layer 1: Hybrid Solution (Highest Quality 60-80%)
- **Method**: Qwen2-VL visual understanding + OCR precise localization
- **Advantage**: Combines visual semantic understanding with precise text localization
- **Applicable**: Complex visual questions containing text
- **Example**:
  ```json
  {
    "generation_method": "hybrid_qwen2vl_ocr",
    "generation_layer": 1,
    "bbox": [0.245, 0.156, 0.387, 0.189],  // Precise text boundaries
    "match_info": {
      "ocr_confidence": 0.95,
      "text_match_score": 0.87
    }
  }
  ```

#### Layer 2: Vision-only Solution (Medium Quality 15-25%)
- **Method**: Uses only Qwen2-VL for region recognition
- **Advantage**: Handles visual elements that OCR cannot recognize
- **Applicable**: Images, icons, complex layouts
- **Example**:
  ```json
  {
    "generation_method": "qwen2vl_only",
    "generation_layer": 2,
    "bbox": [0.1, 0.2, 0.4, 0.6],  // Visual region boundaries
    "description": "Chart showing sales data"
  }
  ```

#### Layer 3: OCR Fallback Solution (Usable Quality 5-15%)
- **Method**: OCR text detection + keyword matching
- **Advantage**: Text-based solution when Qwen2-VL fails
- **Applicable**: Simple text questions
- **Example**:
  ```json
  {
    "generation_method": "emergency_ocr",
    "generation_layer": 3,
    "bbox": [0.3, 0.4, 0.5, 0.45],
    "relevance": "Contains keyword 'total' relevant to the question"
  }
  ```

#### Layer 4: Algorithm Fallback (Fallback Quality <5%)
- **Method**: Algorithmic generation based on question keywords
- **Advantage**: Ensures output for every question
- **Applicable**: When all other methods fail
- **Example**:
  ```json
  {
    "generation_method": "basic_fallback",
    "generation_layer": 4,
    "bbox": [0.05, 0.1, 0.3, 0.15],  // Assumed position
    "content": "Text containing 'contact'"
  }
  ```
