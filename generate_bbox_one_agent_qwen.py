"""
直接使用Qwen2-VL生成Bbox
========================
让Qwen2-VL直接分析图片和问题，生成bbox和description
再让OCR生成bbox
"""

import os
import sys

def select_gpu_before_torch():
    """在导入torch之前选择GPU"""
    print("🚀 Qwen2-VL Bbox生成器")
    print("=" * 50)

    # 临时导入torch来检测GPU
    import subprocess

    try:
        # 使用nvidia-smi获取GPU信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            print("❌ 无法获取GPU信息，将使用默认设置")
            return '0'

        gpu_info = result.stdout.strip().split('\n')
        gpu_count = len(gpu_info)

        if gpu_count == 1:
            print(f"🎯 只有1个GPU可用，自动选择GPU 0")
            return '0'

        print(f"\n🎯 检测到 {gpu_count} 个GPU，选择使用方式:")

        for i, info in enumerate(gpu_info):
            parts = info.split(', ')
            if len(parts) >= 4:
                name = parts[0]
                total_mb = float(parts[1])
                used_mb = float(parts[2])
                free_mb = float(parts[3])

                total_gb = total_mb / 1024
                used_gb = used_mb / 1024
                free_gb = free_mb / 1024

                print(f"   GPU {i}: {name}")
                print(f"      总内存: {total_gb:.1f}GB")
                print(f"      已使用: {used_gb:.1f}GB")
                print(f"      可用: {free_gb:.1f}GB")

                # 如果GPU使用率很高，给出提示
                usage_percent = (used_gb / total_gb) * 100
                if usage_percent > 50:
                    print(f"      ⚠️ 使用率: {usage_percent:.1f}% (较高)")
                elif usage_percent > 10:
                    print(f"      📊 使用率: {usage_percent:.1f}%")
                print()

        print("选择使用方式:")
        print("   0. 使用所有GPU")
        print("   1. 使用两个GPU (推荐，更快)")
        print("   2. 使用单个GPU")

        while True:
            try:
                choice = input("请选择 (0/1/2): ").strip()
                if choice == "0":
                    print(f"✅ 已选择使用所有 {gpu_count} 个GPU")
                    return ','.join(str(i) for i in range(gpu_count))
                elif choice == "1":
                    # 使用两个GPU
                    if gpu_count >= 2:
                        print("请选择要使用的两个GPU:")
                        selected_gpus = []

                        # 选择第一个GPU
                        while True:
                            try:
                                gpu1 = input(f"请选择第一个GPU (0-{gpu_count-1}): ").strip()
                                gpu1_id = int(gpu1)
                                if 0 <= gpu1_id < gpu_count:
                                    selected_gpus.append(gpu1_id)
                                    break
                                else:
                                    print(f"❌ 请输入0到{gpu_count-1}之间的数字")
                            except ValueError:
                                print("❌ 请输入有效的数字")

                        # 选择第二个GPU
                        while True:
                            try:
                                gpu2 = input(f"请选择第二个GPU (0-{gpu_count-1}，不能与第一个相同): ").strip()
                                gpu2_id = int(gpu2)
                                if 0 <= gpu2_id < gpu_count:
                                    if gpu2_id != selected_gpus[0]:
                                        selected_gpus.append(gpu2_id)
                                        break
                                    else:
                                        print("❌ 第二个GPU不能与第一个相同")
                                else:
                                    print(f"❌ 请输入0到{gpu_count-1}之间的数字")
                            except ValueError:
                                print("❌ 请输入有效的数字")

                        gpu_str = ','.join(str(gpu) for gpu in selected_gpus)
                        print(f"✅ 已选择使用GPU {selected_gpus[0]}和{selected_gpus[1]}")
                        return gpu_str
                    else:
                        print("❌ 可用GPU数量不足2个，请选择其他选项")
                elif choice == "2":
                    while True:
                        try:
                            gpu_choice = input(f"请选择单个GPU (0-{gpu_count-1}): ").strip()
                            gpu_id = int(gpu_choice)
                            if 0 <= gpu_id < gpu_count:
                                print(f"✅ 已选择GPU {gpu_id}")
                                return str(gpu_id)
                            else:
                                print(f"❌ 请输入0到{gpu_count-1}之间的数字")
                        except ValueError:
                            print("❌ 请输入有效的数字")
                else:
                    print("❌ 请输入0、1或2")
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n❌ 用户取消")
                sys.exit(1)

    except Exception as e:
        print(f"❌ 获取GPU信息失败: {e}")
        print("将使用默认GPU 0")
        return '0'

# 在导入torch之前选择GPU并设置环境变量
selected_gpu = select_gpu_before_torch()

# �🚨 重要：必须在导入torch之前设置CUDA环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '0' #cuda 内核异步启动
os.environ['TORCH_USE_CUDA_DSA'] = '0' #禁用 CUDA 的 Device-Side Assertions (DSA) 降低运行开销
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' #允许显存动态扩展，提升利用率
os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu  # 设置选择的GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免tokenizers并行处理警告

print(f"🔧 设置CUDA_VISIBLE_DEVICES = {selected_gpu}")

import json
import torch
from time import sleep
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def calculate_iou(bbox1, bbox2):
    """计算两个bbox的IoU (Intersection over Union)"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def remove_duplicate_bboxes(elements, iou_threshold=0.5):
    """去除重复的bbox"""
    if not elements:
        return elements

    # 按bbox面积排序，保留较大的bbox
    elements_with_area = []
    for elem in elements:
        bbox = elem.get('bbox', [])
        if len(bbox) == 4:
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            elements_with_area.append((elem, area))

    # 按面积降序排序
    elements_with_area.sort(key=lambda x: x[1], reverse=True)

    filtered_elements = []
    for elem, area in elements_with_area:
        bbox = elem['bbox']
        is_duplicate = False

        # 检查是否与已保留的bbox重复
        for kept_elem in filtered_elements:
            kept_bbox = kept_elem['bbox']
            iou = calculate_iou(bbox, kept_bbox)
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_elements.append(elem)

    return filtered_elements

# 常量定义
DISTANCE_THRESHOLD = 200  # 像素距离阈值
CONFIDENCE_THRESHOLD = 0.5  # OCR置信度阈值
MIN_WORD_LENGTH = 2  # 最小词长度
TEXT_MATCH_THRESHOLD = 0.5  # 文本匹配阈值
MARGIN_RATIO = 1  # 区域扩展比例



# 验证环境变量设置
print("🔧 环境变量设置:")
for key in ['CUDA_LAUNCH_BLOCKING', 'TORCH_USE_CUDA_DSA', 'PYTORCH_CUDA_ALLOC_CONF', 'TOKENIZERS_PARALLELISM']:
    print(f"  {key}: {os.environ.get(key, '未设置')}")

# GPU选择已在导入torch之前完成

# 配置路径 - 修复相对路径问题
MODEL_PATH = "Qwen2-VL-7B-Instruct"

# 数据集配置
DATASETS = {
    # ===== Visual CoT 按数据集分别处理 =====
    "viscot_flickr30k": { #2号 全新 暂停
        "name": "Visual-CoT-Flickr30k",
        "image_folder": "playground/data/cot/flickr30k",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_flickr30k_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "flickr30k",
        "default_max_samples": None,  # 135,735条
        "total_samples": 135735
    },

    "viscot_gqa": { #3号 全新 暂停
        "name": "Visual-CoT-GQA",
        "image_folder": "playground/data/cot/gqa",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_gqa_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "gqa",
        "default_max_samples": None,  # 88,294条
        "total_samples": 88294
    },

    "viscot_openimages": { # 报错
        "name": "Visual-CoT-OpenImages",
        "image_folder": "playground/data/cot/openimages",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_openimages_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "openimages",
        "default_max_samples": None,  # 43,053条
        "total_samples": 43053
    },

    "viscot_docvqa": { # 1号 全新 开始
        "name": "Visual-CoT-DocVQA",
        "image_folder": "playground/data/cot/docvqa",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_docvqa_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "docvqa",
        "default_max_samples": None,  # 33,453条
        "total_samples": 33453
    },

    "viscot_textcap": {
        "name": "Visual-CoT-TextCap",
        "image_folder": "playground/data/cot/textcap",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_textcap_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "textcap",
        "default_max_samples": None,  # 32,152条
        "total_samples": 32152
    },

    "viscot_v7w": { #0号 全新 运行
        "name": "Visual-CoT-Visual7W",
        "image_folder": "playground/data/cot/v7w",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_v7w_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "v7w",
        "default_max_samples": None,  # 30,491条
        "total_samples": 30491
    },

    "viscot_textvqa": { #ok 旧版本
        "name": "Visual-CoT-TextVQA",
        "image_folder": "playground/data/cot/textvqa",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_textvqa_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "textvqa",
        "default_max_samples": None,  # 18,524条
        "total_samples": 18524
    },

    "viscot_infographicsvqa": {  #号跑
        "name": "Visual-CoT-InfographicsVQA",
        "image_folder": "playground/data/cot/infographicsvqa",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_infographicsvqa_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "infographicsvqa",
        "default_max_samples": None,  # 15,055条 到13090
        "total_samples": 15055
    },

    "viscot_cub": {
        "name": "Visual-CoT-CUB",
        "image_folder": "playground/data/cot/cub",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_cub_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "cub",
        "default_max_samples": None,  # 3,987条
        "total_samples": 3987
    },

    "viscot_vsr": {
        "name": "Visual-CoT-VSR",
        "image_folder": "playground/data/cot/vsr",
        "data_file": "playground/data/viscot_363k.json",
        "output_file": "images_bbox/VisCoT_vsr_one_agent.json",
        "image_id_field": "image",
        "question_id_field": "id",
        "dataset_filter": "vsr",
        "default_max_samples": None,  # 3,376条
        "total_samples": 3376
    },

    # ===== dataset_with_GT 复杂问题数据集 =====
    "gqa_complex": { # 37593
    
        "name": "GQA-Complex",
        "image_folder": "playground/data/cot/gqa",
        "data_file": "dataset_with_GT/GQA/GQA_merged_complex_6plus.json",
        "output_file": "images_bbox/GQA_complex_one_agent.json",
        "image_id_field": "imageId",
        "question_id_field": "question_id",
        "default_max_samples": None,  # 处理全部数据
        "total_samples": 153272,  # 153,272个复杂问题
        "data_format": "gqa_complex"
    },

    "docvqa_complex": { # 13 complete
        "name": "DocVQA-Complex",
        "image_folder": "playground/data/cot/docvqa",
        "data_file": "dataset_with_GT/Docvqa/DocVQA_complex_4plus.json",
        "output_file": "images_bbox/DocVQA_complex_one_agent.json",
        "image_id_field": "imageId",
        "question_id_field": "question_id",
        "default_max_samples": None,  # 处理全部数据
        "total_samples": 11995,  # 约12K个复杂问题
        "data_format": "docvqa_complex"
    },

    "infovqa_complex": { # 13 # 21668
        "name": "InfoVQA-Complex",
        "image_folder": "playground/data/cot/infographicsvqa",
        "data_file": "dataset_with_GT/InfoVQA/InfoVQA_complex_4plus_parallel.json",
        "output_file": "images_bbox/InfoVQA_complex_one_agent.json",
        "image_id_field": "imageId",
        "question_id_field": "question_id",
        "default_max_samples": None,  # 处理全部数据
        "total_samples": 22331,  # 22,331个复杂问题
        "data_format": "infovqa_complex"
    },

    "textvqa_complex": { # complete
        "name": "TextVQA-Complex",
        "image_folder": "playground/data/cot/textvqa",
        "data_file": "dataset_with_GT/TextVQA/TextVQA_complex_3plus_parallel.json",
        "output_file": "images_bbox/TextVQA_complex_one_agent.json",
        "image_id_field": "imageId",
        "question_id_field": "question_id",
        "default_max_samples": None,  # 处理全部数据
        "total_samples": 12508,  # 12,508个复杂问题
        "data_format": "textvqa_complex"
    },

    "visual7w_complex": {# complete
        "name": "Visual7W-Complex",
        "image_folder": "playground/data/cot/v7w",
        "data_file": "dataset_with_GT/Visual7W/Visual7W_complex_3plus_parallel.json",
        "output_file": "images_bbox/Visual7W_complex_one_agent.json",
        "image_id_field": "imageId",
        "question_id_field": "question_id",
        "default_max_samples": None,  # 处理全部数据
        "total_samples": 17954,  # 17,954个复杂问题
        "data_format": "visual7w_complex"
    },

    "vqav2_complex": { # complete 
        "name": "VQAv2-Complex",
        "image_folder": "playground/data/cot/coco",  # VQAv2使用COCO图像
        "data_file": "dataset_with_GT/VQAv2/VQAv2_complex_5plus_parallel.json",
        "output_file": "images_bbox/VQAv2_complex_one_agent.json",
        "image_id_field": "imageId",
        "question_id_field": "question_id",
        "default_max_samples": None,  # 处理全部数据
        "total_samples": 35383,  # 35,383个复杂问题
        "data_format": "vqav2_complex"
    }

}
# 全局变量
model = None
processor = None

# 初始化qwen
def initialize_qwen_model():
    """初始化Qwen2-VL模型"""
    global model, processor

    if model is None:
        print("🚀 正在加载Qwen2-VL模型...")

        # 加载处理器 - 使用与debug_qwen.py相同的简单配置
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        # 加载模型 - 使用多GPU加速
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # 半精度浮点，float16
            device_map="auto",  # 自动分配到多个GPU
            trust_remote_code=True, #允许加载自定义代码
            low_cpu_mem_usage=True #优化CPU的内存使用
        )

        print(f"✅ 模型已加载到GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')}")
        # 显示模型分布情况
        print("✅ 模型加载完成，GPU分布情况:")
        device_map = {}
        dtype_info = {}
        for _, param in model.named_parameters():
            device = str(param.device)
            dtype = str(param.dtype)

            #统计各设备上的参数数量
            if device not in device_map:
                device_map[device] = 0
            device_map[device] += param.numel()

            #统计各精度的参数数量
            if dtype not in dtype_info:
                dtype_info[dtype] = 0
            dtype_info[dtype] += param.numel()

        for device, param_count in device_map.items():
            print(f"   {device}: {param_count:,} 参数")

        print("📊 模型精度分布:")
        for dtype, param_count in dtype_info.items():
            print(f"   {dtype}: {param_count:,} 参数")

        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            for i in range(min(2, torch.cuda.device_count())):  # 显示2个GPU
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"   GPU {i}: 已分配 {allocated:.1f}GB, 缓存 {cached:.1f}GB")

# 移除详细的内存信息打印以提高速度

# 清理GPU
def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        for i in range(min(2, torch.cuda.device_count())):  # 清理2个GPU
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device=i)

# 重置模型状态
def reset_model_state():
    """重置模型状态"""
    global model, processor
    try:
        print("🔄 重置模型状态...")
        # 彻底清理GPU内存
        torch.cuda.empty_cache()
        for i in range(min(2, torch.cuda.device_count())):
            torch.cuda.synchronize(device=i)

        # 重置模型状态
        if hasattr(model, 'eval'):
            model.eval()

        # 检查模型是否需要重新加载
        try:
            # 尝试简单操作检查模型状态
            next(model.parameters()).device
            print("✅ 模型状态重置完成")
        except Exception:
            print("⚠️ 模型状态异常，尝试重新加载...")
            # 重新初始化模型
            model = None
            processor = None
            initialize_qwen_model()
    except Exception as e:
        print(f"⚠️ 模型状态重置失败: {e}")

# 获取GPU可用memeory
def get_available_gpu_memory():
    """获取可用GPU内存（GB）"""
    if not torch.cuda.is_available():
        return 0

    min_free_memory = float('inf')
    for i in range(min(2, torch.cuda.device_count())):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        free = total - cached
        min_free_memory = min(min_free_memory, free)

    return min_free_memory

# 根据GPU内存自适应调整图像尺寸限制
def adaptive_image_size(_, base_max_size=1500):
    """根据GPU内存自适应调整图像尺寸限制"""
    available_memory = get_available_gpu_memory()

    if available_memory > 10:  # 充足内存（还有10G）
        max_size = base_max_size
    elif available_memory > 8:  # 中等内存
        max_size = int(base_max_size * 0.8)  # 1200
    elif available_memory > 6:  # 较少内存
        max_size = int(base_max_size * 0.6)  # 900
    else:  # 内存紧张
        max_size = int(base_max_size * 0.4)  # 600

    print(f"🧠 可用GPU内存: {available_memory:.1f}GB, 图像尺寸限制: {max_size}px")
    return max_size

# sub-输入对话+图片，如果无法正常运行则缩减token
def generate_qwen_response(messages, max_tokens=512):
    """生成Qwen2-VL响应 - 快速版本，不清理内存"""
    global model, processor

    try:
        # 准备输入
        text = processor.apply_chat_template(
            messages, # 输入
            tokenize=False, # 是否在应用聊天模板后立即对文本进行 分词（tokenization）
            add_generation_prompt=True # 添加生成提示符（如 "<|assistant|>"）
        )
        image_inputs,video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], #文本
            images=image_inputs, #图像输入
            video=video_inputs,
            padding=True, #自动填充到相同长度
            return_tensors="pt", #返回pytorch张量
        )

        # 把输入移动到模型设备
        model_device = next(model.parameters()).device
        inputs = inputs.to(model_device)

        # 生成响应 - 使用自适应token策略
        with torch.no_grad():
            # 清理GPU缓存避免数值问题
            torch.cuda.empty_cache()

            # 自适应调整token数量
            available_memory = get_available_gpu_memory()
            if available_memory < 5:  # 内存不足时减少token
                actual_tokens = min(max_tokens, 512)
                print(f"⚠️ GPU内存不足({available_memory:.1f}GB)，减少token到{actual_tokens}")
            else:
                actual_tokens = max_tokens

            print(f"🔄 开始生成，max_tokens={actual_tokens}")

        # 生成响应 
        with torch.no_grad():
            # 清理GPU缓存避免数值问题（copy版本的关键步骤）
            torch.cuda.empty_cache()

            # 使用最保守的生成参数避免数值稳定性问题
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=actual_tokens,  # 使用上面调整后的token
                do_sample=False,  # 使用贪心解码（每次选概率最高的token），保证稳定生成
                temperature=1.0,  # 不缩放logits，AI完全按学习到的概率选择下一个词（最接近训练数据风格）
                pad_token_id=processor.tokenizer.eos_token_id, # 空格
                eos_token_id=processor.tokenizer.eos_token_id, # 结束符
                repetition_penalty=1.0,  # 使用默认值，不会主动避免重复 
                use_cache=True,  # 启用缓存提高稳定性
                output_scores=False,  # 禁用分数输出
                output_attentions=False,  # 禁用注意力输出
                output_hidden_states=False  # 禁用隐藏状态输出
            )
            print(f"✅ 生成完成")

        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip() if output_text else None

    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU内存不足: {e}")
        torch.cuda.empty_cache()
        return None
    except RuntimeError as e:
        # 如果初始设置无法生成，尝试使用更保守的参数重新生成
        if "probability tensor contains either `inf`, `nan` or element < 0" in str(e):
            print(f"❌ 数值稳定性错误: {e}")
            print("🔄 尝试使用更保守的生成参数...")

            # 尝试使用更保守的参数重新生成
            try:
                torch.cuda.empty_cache()  # 清理缓存

                # 重新准备输入（可能有助于解决数值问题）
                inputs = processor(
                    text=[messages],
                    images=[messages[0]["content"][0]["image"]],
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=64,  # 大幅减少token数量
                        do_sample=False,  # 确定性生成
                        temperature=1.0,
                        top_k=50,  # 限制采样范围
                        pad_token_id=processor.tokenizer.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        repetition_penalty=1.0,
                        use_cache=False,  # 禁用缓存
                        output_attentions=False,
                        output_hidden_states=False
                    )

                # 解码输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                print("✅ 保守参数生成成功")
                return output_text[0].strip() if output_text else None

            except Exception as e2:
                print(f"❌ 保守参数生成也失败: {e2}")
                return None
        else:
            print(f"❌ 运行时错误: {e}")
            return None
    except Exception as e:
        print(f"❌ 生成响应时出错: {e}")
        print(f"🔍 错误类型: {type(e).__name__}")
        import traceback
        print(f"🔍 详细错误: {traceback.format_exc()}")
        return None

# layer1或2-简化版本：统一提示词，根据OCR增强结果分层
# ========================================
# 🎯 简化后的新架构说明
# ========================================
# Layer 1/2: 统一的Qwen2-VL方法（简化为2种策略）
#   1 - 统一提示词 + OCR增强成功
#   2 - 统一提示词 + OCR增强失败
# Layer 3: OCR + 关键词匹配
# Layer 4: 纯算法保底
# ========================================
def generate_bboxes_for_question(image_path, question):
    """统一的bbox生成方法：智能选择OCR增强，包含多种Qwen2-VL策略"""
    try:
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        print(f"🔍 原始图像尺寸: {image.size}")

        # 检查图像尺寸，如果任一边超过3000则缩放
        max_dimension = max(image.size)
        if max_dimension > 2000:
            scale_ratio = round(2000 / max_dimension, 3)
            new_width = int(image.size[0] * scale_ratio)
            new_height = int(image.size[1] * scale_ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📐 图像缩放: {original_size} -> {image.size} (scale: {scale_ratio:.3f})")
        else:
            print(f"📐 使用原始图像尺寸: {image.size}")
            scale_ratio = 1.0

        # 保存图像尺寸信息用于归一化坐标
        image._original_size = original_size
        image._scale_ratio = scale_ratio

        # 🎯 简化策略：统一的提示词，根据OCR增强结果分层
        prompt = f"""Question: {question}


        
Analyze the image carefully and identify ALL visual elements that could help answer this question. Look for multiple relevant items including text, objects, numbers, signs, labels, etc.

For each relevant element you find, provide:
1. The reason for selecting this bbox
2. How its content relates to the question
3. Bounding box coordinates
4. A brief description (1-3 words) of what the element contains

Return ONLY this JSON format:

{{
  "relevant_elements": [
    {{
      "description": "brief description (1-3 words)",
      "selection_reason": "why this bbox was selected",
      "content_relation": "how the content in this bbox relates to the question",
      "bbox": [x1, y1, x2, y2]
    }},
    ...,
    {{
      "description": "brief description (1-3 words)",
      "selection_reason": "why this bbox was selected",
      "content_relation": "how the content in this bbox relates to the question",
      "bbox": [x1, y1, x2, y2]
    }}
  ]
}}

IMPORTANT:
- Each region contains only ONE element (text, number, icon, object, etc.)
- Look for multiple pieces of evidence that support the answer
- Include both primary and supporting visual elements
- Coordinates are normalized 0-1 format
Return JSON only."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        print("🔄 使用统一提示词生成bbox...")
        response = generate_qwen_response(messages, max_tokens=512)
        sleep(2)

        if response:
            print(f"🔍 模型原始响应长度: {len(response)}")
            print(f"🔍 模型原始响应: {response}")

            # 清理响应提取JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            elif response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            print(f"🔍 清理后响应长度: {len(response)}")
            print(f"🔍 清理后响应: {response}")

            # 尝试解析JSON
            result = None
            try:
                result = json.loads(response)
                print(f"✅ JSON解析成功")
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
                # 尝试修复JSON格式
                print("🔧 尝试修复JSON格式...")
                try:
                    fixed_response = fix_json_format(response)
                    result = json.loads(fixed_response)
                    print(f"✅ JSON修复成功")
                except json.JSONDecodeError:
                    print(f"❌ JSON修复也失败，降级到OCR方案...")
                    return generate_emergency_bboxes(question, image_path, scale_ratio)

            if result:
                # 处理结果格式，传入图像尺寸用于坐标归一化和问题用于关键词检查
                result = process_qwen_result(result, "bbox", image.size[0], image.size[1], question)

                # 检查是否有有效的元素
                if not result.get('relevant_elements'):
                    print("❌ Qwen2-VL没有生成有效的bbox，降级到Layer 3...")
                    return generate_emergency_bboxes(question, image_path, scale_ratio)

                # 🎯 关键：对所有成功的结果都尝试OCR增强
                print("🔧 启用混合方案：使用OCR精确化bbox位置...")

                # 为每个元素添加rough_bbox信息（使用原始的bbox作为粗略区域）
                elements_with_rough_bbox = []
                for element in result['relevant_elements']:
                    if isinstance(element, dict) and 'bbox' in element:
                        element_copy = element.copy()
                        element_copy['rough_bbox'] = element['bbox']  # 使用原始bbox作为粗略区域
                        elements_with_rough_bbox.append(element_copy)
                    else:
                        elements_with_rough_bbox.append(element)

                enhanced_elements = match_content_with_ocr(elements_with_rough_bbox, image_path, question)

                # 检查OCR增强是否成功：看第一个元素是否有match_info
                ocr_enhanced_success = False
                if enhanced_elements:
                    for element in enhanced_elements:
                        if isinstance(element, dict) and 'match_info' in element:
                            ocr_enhanced_success = True
                            break

                if ocr_enhanced_success:
                    # OCR增强成功 - Layer 1 (最高质量)
                    # 去除重复的bbox，但保留rough_bbox字段
                    enhanced_elements = remove_duplicate_bboxes(enhanced_elements, iou_threshold=0.5)
                    result['relevant_elements'] = enhanced_elements
                    result["generation_method"] = "hybrid_qwen2vl_ocr"
                    result["generation_layer"] = 1
                    result["generation_description"] = "Generated by hybrid method: Qwen2-VL + OCR precise localization"
                    print(f"✅ OCR增强成功：{len(enhanced_elements)} 个精确bbox (Layer 1, 去重后)")
                    return result
                else:
                    # OCR增强失败，但Qwen2-VL结果可用 - Layer 2 (中等质量)
                    # 保留rough_bbox字段，使用原始结果
                    clean_elements = []
                    for element in result['relevant_elements']:
                        if isinstance(element, dict):
                            # 保留rough_bbox字段，不删除
                            clean_elements.append(element)
                        else:
                            clean_elements.append(element)

                    # 去除重复的bbox
                    clean_elements = remove_duplicate_bboxes(clean_elements, iou_threshold=0.5)
                    result['relevant_elements'] = clean_elements
                    result["generation_method"] = "qwen2vl_only"
                    result["generation_layer"] = 2
                    result["generation_description"] = "Generated by Qwen2-VL only (OCR enhancement failed)"
                    print(f"✅ Qwen2-VL成功（OCR增强失败）：{len(result['relevant_elements'])} 个bbox (Layer 2, 去重后)")
                    return result
        else:
            print("❌ Qwen2-VL无响应，降级到OCR方案...")
            return generate_emergency_bboxes(question, image_path, scale_ratio)

        # 如果到这里说明出现了意外情况，使用保底方案
        print("🔄 意外情况，使用保底方案...")
        return generate_emergency_bboxes(question, image_path, scale_ratio)

    except Exception as e:
        print(f"❌ 生成bbox时出错: {e}")
        print("🔄 使用最后保底方案...")
        return generate_emergency_bboxes(question, image_path, 1.0)

def extract_question_keywords(question):
    """从问题中提取关键词"""
    import re

    # 转换为小写
    question = question.lower()

    # 移除常见的疑问词和停用词
    stop_words = {
        'what', 'where', 'when', 'why', 'how', 'who', 'which', 'is', 'are', 'was', 'were',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'can',
        'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must', 'do', 'does',
        'did', 'have', 'has', 'had', 'be', 'been', 'being', 'you', 'i', 'we', 'they', 'it'
    }

    # 提取单词
    words = re.findall(r'\b[a-zA-Z]+\b', question)

    # 过滤停用词和短词
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    return keywords

def check_content_relevance(content_relation, question_keywords, min_keyword_match=1):
    """检查content_relation是否包含问题的关键词"""
    if not content_relation or not question_keywords:
        return False

    content_lower = content_relation.lower()

    # 计算匹配的关键词数量
    matched_keywords = 0
    for keyword in question_keywords:
        if keyword in content_lower:
            matched_keywords += 1

    return matched_keywords >= min_keyword_match

def process_qwen_result(result, bbox_field, image_width=None, image_height=None, question=None):
    """处理Qwen2-VL的结果格式"""
    # 检查结果格式
    if isinstance(result, list):
        # 如果返回的是数组，转换为期望的字典格式
        print("🔧 检测到数组格式，转换为字典格式")
        result = {"relevant_elements": result}
    elif not isinstance(result, dict):
        # 如果既不是字典也不是数组，创建空结果
        print("⚠️ 检测到非预期格式，创建空结果")
        result = {"relevant_elements": []}

    # 验证和清理结果
    if result.get('relevant_elements'):
        print(f"🔍 原始元素数量: {len(result['relevant_elements'])}")
        valid_elements = []
        for i, element in enumerate(result['relevant_elements']):
            print(f"🔍 检查元素 {i}: {element}")

            if isinstance(element, dict) and ('description' in element or 'selection_reason' in element or 'content_relation' in element) and bbox_field in element:
                bbox = element[bbox_field]
                print(f"   📦 原始bbox: {bbox}")

                if isinstance(bbox, list) and len(bbox) == 4:
                    # 检查每个坐标
                    coord_valid = all(isinstance(x, (int, float)) for x in bbox)

                    if coord_valid:
                        # 检查是否需要归一化（如果坐标>1，说明是像素坐标）
                        if any(x > 1 for x in bbox) and image_width and image_height:
                            # 转换像素坐标为归一化坐标
                            normalized_bbox = [
                                bbox[0] / image_width,   # x1
                                bbox[1] / image_height,  # y1
                                bbox[2] / image_width,   # x2
                                bbox[3] / image_height   # y2
                            ]
                            # 确保坐标在0-1范围内
                            normalized_bbox = [max(0, min(1, coord)) for coord in normalized_bbox]
                            element[bbox_field] = normalized_bbox
                            print(f"   🔄 归一化后bbox: {normalized_bbox}")
                            bbox = normalized_bbox

                        range_valid = all(0 <= x <= 1 for x in bbox)

                        # 检查宽度和高度是否为0
                        width = bbox[2] - bbox[0]  # x2 - x1
                        height = bbox[3] - bbox[1]  # y2 - y1
                        size_valid = width > 0 and height > 0

                        print(f"   ✓ 坐标类型有效: {coord_valid}")
                        print(f"   ✓ 坐标范围有效 (0-1): {range_valid}")
                        print(f"   ✓ 尺寸有效 (宽度={width:.4f}, 高度={height:.4f}): {size_valid}")

                        if range_valid and size_valid:
                            # 统一bbox字段名为'bbox'
                            if bbox_field != 'bbox':
                                element['bbox'] = element.pop(bbox_field)

                            # 直接添加所有有效的bbox，不进行关键词筛选
                            valid_elements.append(element)
                        else:
                            if not range_valid:
                                print(f"   ❌ 元素 {i} 坐标范围无效 (不在0-1范围内): {bbox}")
                            if not size_valid:
                                print(f"   ❌ 元素 {i} 尺寸无效 (宽度={width:.4f}, 高度={height:.4f}): {bbox}")
                    else:
                        print(f"   ❌ 元素 {i} 坐标类型无效: {bbox}")
                else:
                    print(f"   ❌ 元素 {i} bbox格式无效: {bbox}")
            else:
                missing_fields = []
                if not isinstance(element, dict):
                    missing_fields.append("不是字典")
                elif 'content' not in element:
                    missing_fields.append("缺少content")
                elif bbox_field not in element:
                    missing_fields.append(f"缺少{bbox_field}")
                print(f"   ❌ 元素 {i} 验证失败: {', '.join(missing_fields)}")

        # 去除重复的bbox
        valid_elements = remove_duplicate_bboxes(valid_elements, iou_threshold=0.5)
        result['relevant_elements'] = valid_elements
        print(f"✅ 验证后保留 {len(valid_elements)} 个有效元素 (去重后)")
    else:
        # 确保有relevant_elements字段
        result['relevant_elements'] = []
        print("⚠️ 没有找到有效的relevant_elements")

    return result

def fix_json_format(json_str):
    """尝试修复常见的JSON格式问题"""
    # 移除可能的多余字符
    json_str = json_str.strip()

    # 修复双逗号问题 (如 "0.5,," -> "0.5,")
    import re
    json_str = re.sub(r',+', ',', json_str)  # 将多个连续逗号替换为单个逗号

    # 如果JSON被截断，尝试找到最后一个完整的对象
    if not json_str.endswith('}') and not json_str.endswith(']'):
        # 找到最后一个完整的元素
        last_complete_brace = json_str.rfind('}')
        if last_complete_brace > 0:
            # 检查是否需要添加结束符
            temp_str = json_str[:last_complete_brace + 1]
            # 计算大括号平衡
            open_braces = temp_str.count('{')
            close_braces = temp_str.count('}')
            open_brackets = temp_str.count('[')
            close_brackets = temp_str.count(']')

            # 添加缺失的结束符
            if open_brackets > close_brackets:
                temp_str += ']' * (open_brackets - close_brackets)
            if open_braces > close_braces:
                temp_str += '}' * (open_braces - close_braces)

            json_str = temp_str

    return json_str


# layer3-紧急生成bbox，基于问题里的关键词，用OCR标记bbox，OCR + 关键词匹配 (3.0)
def generate_emergency_bboxes(question, image_path, _=1.0):
    """最后保底方案：基于关键词生成bbox"""
    print("🚨 使用紧急保底方案：基于关键词生成bbox")

    try:
        # 初始化PaddleOCR作为保底 - 直接加载
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
            print("✅ PaddleOCR初始化成功 (保底模式)")
        except Exception as e:
            print(f"❌ PaddleOCR初始化失败: {e}")
            return generate_basic_fallback_bboxes(question)

        # 从问题中提取关键词
        keywords = extract_keywords_from_question(question)
        print(f"🔍 提取的关键词: {keywords}")

        # 执行OCR
        if hasattr(ocr, 'predict'):
            ocr_results = ocr.predict(image_path, cls=True)
        else:
            ocr_results = ocr.ocr(image_path, cls=True)

        if not ocr_results or not ocr_results[0]:
            print("❌ OCR未检测到文本，使用基本保底方案")
            return generate_basic_fallback_bboxes(question)

        # 获取图像尺寸用于坐标归一化
        from PIL import Image
        image = Image.open(image_path)
        image_width, image_height = image.size

        # 查找关键词对应的bbox
        emergency_elements = []
        found_keywords = set()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            for line in ocr_results[0]:
                text = line[1][0].strip().lower()
                if keyword_lower in text or any(word in text for word in keyword_lower.split()):
                    # 转换PaddleOCR的bbox格式为统一的[x1, y1, x2, y2]格式
                    bbox_points = line[0]
                    bbox_coords = normalize_bbox_format(bbox_points)
                    if bbox_coords:
                        # 归一化坐标到0-1范围
                        x1, y1, x2, y2 = bbox_coords
                        normalized_bbox = [
                            round(x1 / image_width, 3),
                            round(y1 / image_height, 3),
                            round(x2 / image_width, 3),
                            round(y2 / image_height, 3)
                        ]
                        emergency_elements.append({
                            "type": "text",
                            "content": line[1][0],
                            "bbox": normalized_bbox,
                            "relevance": f"Contains keyword '{keyword}' relevant to the question"
                        })
                        found_keywords.add(keyword)
                        break

        # 如果没找到关键词，至少返回一些OCR文本
        if not emergency_elements:
            print("⚠️ 未找到关键词匹配，返回前3个OCR文本")
            for line in ocr_results[0][:3]:
                bbox_points = line[0]
                bbox_coords = normalize_bbox_format(bbox_points)
                if bbox_coords:
                    # 归一化坐标到0-1范围
                    x1, y1, x2, y2 = bbox_coords
                    normalized_bbox = [
                        round(x1 / image_width, 3),
                        round(y1 / image_height, 3),
                        round(x2 / image_width, 3),
                        round(y2 / image_height, 3)
                    ]
                    emergency_elements.append({
                        "type": "text",
                        "content": line[1][0],
                        "bbox": normalized_bbox,
                        "relevance": f"Prominent text that might be relevant to: {question}"
                    })

        # 坐标已经归一化，无需进一步处理

        print(f"✅ 保底方案生成了 {len(emergency_elements)} 个bbox")
        return {
            "question_analysis": f"Emergency keyword-based analysis for: {question}",
            "relevant_elements": emergency_elements,
            "answer_reasoning": f"Found text elements related to keywords: {list(found_keywords)}",
            "generation_method": "emergency_ocr",
            "generation_layer": 3,  # Layer 3: OCR + 关键词匹配
            "generation_description": "Generated by emergency OCR-based keyword matching when all other methods failed"
        }

    except Exception as e:
        print(f"❌ 保底方案出错: {e}")
        return generate_basic_fallback_bboxes(question)

# layer4-最后保底-给左边的三个框，纯算法保底 (4.0)
def generate_basic_fallback_bboxes(question):
    """最基本的保底方案"""
    print("🚨 使用最基本的保底方案")

    # 提取关键词
    keywords = extract_keywords_from_question(question)

    # 为每个关键词生成一个假设的bbox
    emergency_elements = []
    for i, keyword in enumerate(keywords[:3]):
        y_offset = i * 120 + 50
        emergency_elements.append({
            "type": "text",
            "content": f"Text containing '{keyword}'",
            "bbox": [50, y_offset, 300, y_offset + 50],
            "relevance": f"Assumed location for keyword '{keyword}' from question"
        })

    return {
        "question_analysis": f"Basic fallback analysis for: {question}",
        "relevant_elements": emergency_elements,
        "answer_reasoning": f"Generated basic bboxes for keywords: {keywords[:3]}",
        "generation_method": "basic_fallback",
        "generation_layer": 4,  # Layer 4: 纯算法保底
        "generation_description": "Generated by basic fallback with assumed bbox positions when all other methods failed"
    }

# 统一bbox格式
def normalize_bbox_format(bbox):
    """
    将不同格式的bbox统一转换为 [x1, y1, x2, y2] 格式
    支持的输入格式：
    1. [x1, y1, x2, y2] - 直接返回
    2. [[x1, y1, x2, y2]] - 嵌套列表格式
    3. [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] - 4个角点格式
    """
    if not bbox or not isinstance(bbox, list):
        return None

    try:
        # 格式1: [x1, y1, x2, y2] - 直接返回
        if len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox):
            return bbox

        # 格式2: [[x1, y1, x2, y2]] - 嵌套列表格式
        if len(bbox) == 1 and isinstance(bbox[0], list) and len(bbox[0]) == 4:
            return bbox[0]

        # 格式3: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] - 4个角点格式
        if len(bbox) == 4 and all(isinstance(point, list) and len(point) == 2 for point in bbox):
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
            return [x1, y1, x2, y2]

        # 其他格式暂不支持
        return None

    except Exception:
        return None

# 从问题中提取关键词
def extract_keywords_from_question(question):
    """从问题中提取关键词"""
    # 简单的关键词提取
    import re

    # 移除常见的停用词
    stop_words = {'what', 'which', 'how', 'where', 'when', 'why', 'who', 'is', 'are', 'was', 'were',
                  'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                  'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
                  'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'has', 'have'}

    # 提取单词
    words = re.findall(r'\b[a-zA-Z]+\b', question.lower())

    # 过滤停用词和短词
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # 返回前5个关键词
    return keywords[:5]


# 选择数据集
def select_dataset():
    """选择要处理的数据集"""
    print("\n📊 请选择要处理的数据集:")
    for i, (key, config) in enumerate(DATASETS.items(), 1):
        # 显示基本信息
        max_samples_str = "全部" if config['default_max_samples'] is None else str(config['default_max_samples'])
        total_samples_str = f" (总共 {config['total_samples']:,} 条)" if 'total_samples' in config else ""
        print(f"   {i}. {config['name']} ({key}) - 默认处理 {max_samples_str} 条{total_samples_str}")

        # 处理单文件或多文件配置
        if 'data_files' in config:
            data_files_str = ', '.join(config['data_files'])
            print(f"      数据文件: {data_files_str}")
        elif 'data_file' in config:
            print(f"      数据文件: {config['data_file']}")

        print(f"      图像目录: {config['image_folder']}")
        print(f"      输出文件: {config['output_file']}")

        # 显示viscot数据集的详细信息
        if 'total_samples' in config:
            print(f"      总数据量: {config['total_samples']:,} 条")

        if 'datasets_breakdown' in config:
            print(f"      数据集构成:")
            for dataset, count in config['datasets_breakdown'].items():
                print(f"        - {dataset}: {count:,} 条")

        if 'dataset_filter' in config:
            print(f"      过滤条件: 仅处理 {config['dataset_filter']} 数据")

        print()

    while True:
        try:
            choice = input(f"请输入选择 (1-{len(DATASETS)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(DATASETS):
                dataset_key = list(DATASETS.keys())[choice_num - 1]
                selected_config = DATASETS[dataset_key]
                print(f"✅ 已选择: {selected_config['name']}")
                # 数据集键名和完整配置字典
                return dataset_key, selected_config
            else:
                print(f"❌ 请输入 1 到 {len(DATASETS)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")

# 检查目标文件中的结果，返回需要处理的样本和已存在的结果
def check_existing_results(output_file, samples):
    """检查已存在的结果，只做统计，不过滤样本"""
    if not os.path.exists(output_file):
        return samples, []

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)

        # 统计各种类型的结果
        good_results_count = 0  # layer 1/2
        backup_results_count = 0  # layer 3/4

        for result in existing_results:
            bbox_analysis = result.get('bbox_analysis')
            if bbox_analysis and bbox_analysis.get('generation_layer', 1) >= 3:
                backup_results_count += 1
            else:
                good_results_count += 1

        new_samples_count = len(samples) - len(existing_results)

        print(f"📊 检查已存在结果:")
        print(f"   已存在: {len(existing_results)} 条结果")
        print(f"   其中好结果: {good_results_count} 条 (layer 1/2)")
        if backup_results_count > 0:
            print(f"   其中备用方案: {backup_results_count} 条 (generation_layer >= 3)")
        if new_samples_count > 0:
            print(f"   新样本: {new_samples_count} 条")

        return samples, existing_results

    except Exception as e:
        print(f"⚠️ 读取已存在文件失败: {e}")
        return samples, []

# 查找图片文件
def find_image_file(image_name, base_path, data_format=None):
    """查找图片文件"""
    # 如果 image_name 已经包含扩展名，先去掉
    if '.' in image_name:
        image_name = image_name.split('.')[0]

    # 根据数据集类型确定可能的扩展名
    if data_format == 'gqa_complex':
        extensions = ['.jpg', '.jpeg']
    elif data_format == 'docvqa_complex':
        extensions = ['.png', '.jpg', '.jpeg']
    elif data_format == 'infovqa_complex':
        extensions = ['.jpeg', '.jpg', '.png']
    elif data_format == 'textvqa_complex':
        extensions = ['.jpg', '.jpeg', '.png']
    elif data_format == 'visual7w_complex':
        extensions = ['.jpg', '.jpeg', '.png']
    elif data_format == 'vqav2_complex':
        extensions = ['.jpg', '.jpeg', '.png']
    else:
        extensions = ['.jpg', '.jpeg', '.png']

    # 直接在base_path中查找图片
    for ext in extensions:
        image_path = os.path.join(base_path, image_name + ext)
        if os.path.isfile(image_path):
            return image_path

    # 特殊处理：如果是数字ID，尝试查找所有可能的格式
    if image_name.isdigit():
        # 根据数据集类型生成可能的文件名格式
        possible_names = [image_name]  # 原始名称

        if data_format == 'gqa_complex':
            # GQA: 通常是数字ID
            possible_names.extend([
                f"{int(image_name):012d}",  # 12位补零
                f"n{image_name:08d}",  # n开头8位补零
            ])
        elif data_format == 'vqav2_complex':
            # VQAv2: COCO格式
            possible_names.extend([
                f"COCO_train2014_{int(image_name):012d}",
                f"COCO_val2014_{int(image_name):012d}",
                f"{int(image_name):012d}",
            ])
        else:
            # 通用格式
            possible_names.extend([
                f"{int(image_name):012d}",  # 12位补零
                f"COCO_train2014_{int(image_name):012d}",  # COCO格式
                f"flickr30k_{image_name}",  # Flickr30k格式
            ])

        for name in possible_names:
            for ext in extensions:
                image_path = os.path.join(base_path, name + ext)
                if os.path.isfile(image_path):
                    return image_path

    # 如果还是找不到，尝试模糊匹配（包含image_name的文件）
    try:
        if os.path.exists(base_path):
            for filename in os.listdir(base_path):
                if filename.lower().endswith(tuple(extensions)):
                    # 去掉扩展名进行比较
                    file_base = filename.rsplit('.', 1)[0]
                    if image_name in file_base or file_base in image_name:
                        return os.path.join(base_path, filename)
    except Exception as e:
        print(f"⚠️ 搜索图片时出错: {e}")

    return None

# 读取 viscot363等类似的GT数据集
def load_dataset_data(dataset_config):
    """加载数据集数据"""
    data_file = dataset_config['data_file']
    data_format = dataset_config.get('data_format', '')

    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None

    try:
        print(f"📖 加载数据文件: {data_file}")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset_name = dataset_config['name']

        if dataset_name.startswith('Visual-CoT'):
            # Visual CoT数据处理
            if isinstance(data, list):
                samples = data
            else:
                samples = data.get('data', data)

            print(f"✅ 成功加载 {len(samples)} 条数据")

            # 如果指定了数据集过滤器，只保留特定数据集的数据
            if 'dataset_filter' in dataset_config:
                filter_dataset = dataset_config['dataset_filter']
                original_count = len(samples)
                samples = [s for s in samples if s.get('dataset') == filter_dataset]
                print(f"✅ 过滤数据集 '{filter_dataset}': {len(samples)}/{original_count} 条数据")

            return samples

        elif data_format.endswith('_complex'):
            # dataset_with_GT 复杂问题数据集处理
            # 这些数据集的格式是字典，key是问题ID，value是问题信息
            if isinstance(data, dict):
                # 转换为 (key, value) 元组列表，方便处理
                samples = [(key, value) for key, value in data.items()]
                print(f"✅ 成功加载 {len(samples)} 条复杂问题数据")
                return samples
            elif isinstance(data, list):
                # 如果已经是列表格式，直接返回
                samples = data
                print(f"✅ 成功加载 {len(samples)} 条数据")
                return samples
            else:
                print(f"❌ 不支持的数据格式: {type(data)}")
                return None
        else:
            # 通用处理
            if isinstance(data, list):
                samples = data
            elif 'data' in data:
                samples = data['data']
            elif 'annotations' in data:
                samples = data['annotations']
            else:
                samples = list(data.values()) if isinstance(data, dict) else data

            print(f"✅ 成功加载数据集，共 {len(samples)} 个样本")
            return samples

    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return None

# 从 viscot363k里面提取问题、GT bbox、GT answer、image info
def get_sample_info(sample, dataset_config):
    """从样本中提取信息"""
    dataset_name = dataset_config['name']
    data_format = dataset_config.get('data_format', '')

    if dataset_name.startswith('Visual-CoT'):
        # Visual CoT数据格式处理
        conversations = sample.get('conversations', [])
        if len(conversations) < 2:
            return None

        # 提取问题（第一个conversation，去掉bbox指令）
        question_raw = conversations[0].get('value', '').replace('<image>', '').strip()
        question = question_raw.replace('Please provide the bounding box coordinate of the region that can help you answer the question better.', '').strip()
        question = question.rstrip('.?!').strip()

        # 提取bbox坐标（第二个conversation）
        bbox_str = conversations[1].get('value', '').strip()
        bbox_coords = None
        if bbox_str.startswith('[') and bbox_str.endswith(']'):
            try:
                bbox_coords = eval(bbox_str)  # 解析bbox坐标
            except:
                bbox_coords = None

        # 提取真正的答案（最后一个conversation）
        answer = conversations[-1].get('value', '').strip() if len(conversations) > 2 else ''

        # 提取图片信息和GT bbox
        images = sample.get('image', [])
        if not images:
            return None

        # 处理图片路径和GT bbox
        if isinstance(images, list) and len(images) > 1:
            # 第二个元素可能包含GT bbox信息：cot/v7w/v7w_276.jpg###[446, 246, 502, 345]
            image_with_bbox = images[1]
            if '###' in image_with_bbox:
                image_path, bbox_str = image_with_bbox.split('###', 1)
                try:
                    gt_bbox = eval(bbox_str.strip())  # 解析GT bbox
                except:
                    gt_bbox = None
            else:
                image_path = image_with_bbox
                gt_bbox = None
        else:
            image_path = images[0] if isinstance(images, list) else images
            gt_bbox = None

        # 从路径中提取图片名：cot/docvqa/abc.jpg -> abc
        image_name = os.path.basename(image_path).split('.')[0] if image_path else ''

        # 提取数据集信息
        dataset_type = sample.get('dataset', 'unknown')

        # 生成唯一的question_id
        question_id = f"{dataset_type}_{image_name}_{hash(question) % 100000}"

        return {
            'question_id': question_id,
            'question': question,
            'image_name': image_name,
            'image_path': image_path,
            'answers': [answer] if answer else [],
            'GT_bbox': gt_bbox,  # 真正的GT bbox（像素坐标）
            'viscot_bbox': bbox_coords,  # Visual-CoT生成的bbox（归一化坐标）
            # 不保存Visual-CoT的原始数据：dataset_type, conversations, dataset_name
        }
    elif data_format.endswith('_complex'):
        # dataset_with_GT 复杂问题数据集格式处理
        # 这些数据集的格式是字典，key是问题ID，value是问题信息

        # 如果sample是字典的一个条目，需要提取key和value
        if isinstance(sample, tuple) and len(sample) == 2:
            # (key, value) 格式
            sample_key, sample_data = sample
            question_id = sample_key
        else:
            # 直接是sample数据
            sample_data = sample
            question_id = sample.get('question_id', sample.get('id', ''))

        # 提取基本信息
        question = sample_data.get('question', '')
        answer = sample_data.get('answer', '')
        image_id = sample_data.get('imageId', sample_data.get('image_id', ''))

        # 处理答案格式
        all_answers = sample_data.get('all_answers', [])
        if not all_answers and answer:
            all_answers = [answer]

        # 根据不同数据集处理图像名称
        if data_format == 'gqa_complex':
            # GQA: imageId 通常是数字，对应 playground/data/cot/gqa/xxx.jpg
            image_name = str(image_id)
        elif data_format == 'docvqa_complex':
            # DocVQA: imageId 通常是文档ID，对应 playground/data/cot/docvqa/xxx.png
            image_name = str(image_id)
        elif data_format == 'infovqa_complex':
            # InfoVQA: 需要从原始数据中获取image_local_name
            # 由于复杂数据集中imageId为空，需要从sample_key中提取原始ID
            if hasattr(sample_data, 'get') and sample_data.get('image_local_name'):
                image_name = sample_data['image_local_name'].replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
            else:
                # 从sample_key中提取：InfoVQA_train_65718 -> 65718
                parts = question_id.split('_')
                if len(parts) >= 3:
                    original_id = parts[-1]  # 获取最后一部分作为ID
                    # 需要查找对应的image_local_name，这里先用ID作为fallback
                    image_name = original_id
                else:
                    image_name = str(image_id)
        elif data_format == 'textvqa_complex':
            # TextVQA: imageId 对应 playground/data/cot/textvqa/xxx.jpg
            image_name = str(image_id)
        elif data_format == 'visual7w_complex':
            # Visual7W: imageId 对应 playground/data/cot/v7w/v7w_xxx.jpg
            image_name = f"v7w_{image_id}"
        elif data_format == 'vqav2_complex':
            # VQAv2: imageId 对应 playground/data/cot/vqav2/xxx.jpg
            image_name = str(image_id)
        else:
            image_name = str(image_id)

        return {
            'question_id': question_id,
            'question': question,
            'image_name': image_name,
            'image_id': image_id,
            'answers': all_answers,
            'data_format': data_format
        }
    else:
        # 通用处理
        question_id_field = dataset_config['question_id_field']
        image_id_field = dataset_config['image_id_field']
        return {
            'question_id': sample.get(question_id_field),
            'question': sample.get('question'),
            'image_name': sample.get(image_id_field, ''),
            'image_id': sample.get(image_id_field, ''),
            'answers': sample.get('answers', [sample.get('answer', '')])
        }

# 主要处理命令和逻辑的程序
# 生成之前失败的数据 + 新数据
def process_samples_with_config(dataset_config, max_samples=None):
    """处理数据生成bbox（指定配置）"""
    # 如果没有指定max_samples，使用数据集的默认值
    if max_samples is None:
        max_samples = dataset_config['default_max_samples']

    # 显示处理信息
    if max_samples is None:
        print(f"📊 将处理 {dataset_config['name']} 数据集的所有数据")
    else:
        print(f"📊 将处理 {dataset_config['name']} 数据集的前 {max_samples} 条数据")

    # 输出文件路径
    output_file = dataset_config['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 检查文件是否存在并询问用户选择
    existing_results = []
    start_index = 0

    # 如果已经有输出文件
    if os.path.exists(output_file):
        print(f"📁 发现已存在的文件: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            print(f"📊 已有 {len(existing_results)} 条数据")

            # 检查需要重新生成的样本（generation_layer >= 3）
            # 优先级是用AI模型生成的bbox（Layer 1和2），Layer 3和4需要重新生成
            retry_indices = []
            for idx, result in enumerate(existing_results):
                bbox_analysis = result.get('bbox_analysis')
                if bbox_analysis and bbox_analysis.get('generation_layer', 1) >= 3:
                    retry_indices.append(idx)

            print(f"🔍 发现 {len(retry_indices)} 个使用备用方案的样本 (generation_layer >= 3)")

            # 询问是否先处理备用方案样本
            process_retry_first = False
            if retry_indices:
                while True:
                    retry_choice = input(f"\n🔄 是否先重新生成 {len(retry_indices)} 个备用方案样本？\n1. 是，先处理备用方案样本\n2. 否，跳过\n请输入 1 或 2: ").strip()
                    if retry_choice == "1":
                        process_retry_first = True
                        print(f"✅ 将先重新生成 {len(retry_indices)} 个备用方案样本")
                        break
                    elif retry_choice == "2":
                        print("✅ 跳过备用方案样本的重新生成")
                        break
                    else:
                        print("❌ 无效选择，请输入 1 或 2")

            # 询问后续处理方式
            while True:
                choice_text = "\n请选择后续操作:\n1. 接着生成新样本 (从第{}条开始)\n2. 重新开始 (覆盖现有文件)\n3. 仅处理备用方案样本后结束\n请输入选择: ".format(len(existing_results) + 1)

                choice = input(choice_text).strip()

                if choice == "1":
                    print(f"✅ 选择接着生成，从第 {len(existing_results) + 1} 条开始")
                    start_index = len(existing_results)
                    continue_generation = True
                    break
                elif choice == "2":
                    print("✅ 选择重新开始，将覆盖现有文件")
                    existing_results = []
                    start_index = 0
                    continue_generation = True
                    process_retry_first = False  # 重新开始时不需要处理备用方案
                    break
                elif choice == "3":
                    print("✅ 选择仅处理备用方案样本")
                    start_index = len(existing_results)
                    continue_generation = False
                    if not process_retry_first:
                        print("⚠️ 但您之前选择跳过备用方案样本，将直接结束")
                        return
                    break
                else:
                    print("❌ 无效选择，请输入 1、2 或 3")
        except Exception as e:
            print(f"⚠️ 读取现有文件失败: {e}")
            print("将重新开始生成")
            existing_results = []
            start_index = 0
            # 初始化缺失的变量
            process_retry_first = False
            retry_indices = []
            continue_generation = True
    else:
        print(f"📁 文件不存在，将创建新文件: {output_file}")
        # 初始化变量
        process_retry_first = False
        retry_indices = []
        continue_generation = True

    # 初始化模型
    print("🚀 初始化Qwen2-VL模型...")
    initialize_qwen_model()

    # 加载数据集
    all_samples = load_dataset_data(dataset_config)
    if not all_samples:
        print("❌ 无法加载数据集")
        return

    # 显示数据集总数量
    total_samples = len(all_samples)
    print(f"📊 数据集总样本数: {total_samples:,}")

    # 限制处理数量
    if max_samples is not None and len(all_samples) > max_samples:
        all_samples = all_samples[:max_samples]
        print(f"📊 限制处理数量为: {max_samples:,}")
    else:
        print(f"📊 将处理全部 {len(all_samples):,} 条数据")

    # 检查已存在的结果，过滤出需要处理的样本
    samples, existing_results = check_existing_results(output_file, all_samples)

    if len(samples) == 0:
        print("✅ 所有数据都已处理完成，无需重新生成")
        return

    results = existing_results.copy()  # 复制已有结果
    remaining_samples = len(samples)  # 需要处理的样本数量
    already_processed = len(existing_results)  # 已处理的样本数量

    print(f"📊 处理状态: 已完成 {already_processed:,} 条，剩余 {remaining_samples:,} 条需要处理")
    consecutive_failures = 0  # 连续失败计数器

    # 第一阶段：处理备用方案样本（如果选择了的话）
    if process_retry_first and retry_indices:
        print(f"\n🔄 第一阶段：重新生成 {len(retry_indices)} 个备用方案样本")
        for idx, i in enumerate(retry_indices):
            entry = samples[i]
            sample_info = get_sample_info(entry, dataset_config)

            question_id = sample_info['question_id']
            question = sample_info['question']
            image_name = sample_info['image_name']

            old_layer = results[i].get('bbox_analysis', {}).get('generation_layer', 'unknown')
            print(f"\n🔄 重新生成样本 {i+1} (进度: {idx+1}/{len(retry_indices)}, ID: {question_id})")
            print(f"   原generation_layer: {old_layer}")
            print(f"   问题: {question[:100]}...")

            # 查找图片文件
            data_format = sample_info.get('data_format', dataset_config.get('data_format', ''))
            image_path = find_image_file(image_name, dataset_config['image_folder'], data_format)

            if not image_path:
                print(f"❌ 找不到图片: {image_name}")
                continue

            try:
                # 重新生成bbox
                print("🔍 分析图片和问题，生成相关bbox...")
                bbox_analysis = generate_bboxes_for_question(image_path, question)

                # 构建结果条目并更新到原位置
                result_entry = dict(sample_info)
                result_entry["bbox_analysis"] = bbox_analysis
                results[i] = result_entry

                # 统计信息
                if bbox_analysis and bbox_analysis.get('relevant_elements'):
                    bbox_count = len(bbox_analysis['relevant_elements'])
                    new_layer = bbox_analysis.get('generation_layer', 'unknown')
                    print(f"✅ 识别了 {bbox_count} 个相关元素")
                    if new_layer == 1:
                        print(f"🎉 重新生成成功！从layer {old_layer} 提升到 layer 1")
                    else:
                        print(f"⚠️ 仍为备用方案 layer {new_layer}")
                    consecutive_failures = 0
                else:
                    print("⚠️ 重新生成仍未识别到相关元素")
                    consecutive_failures += 1

                    if consecutive_failures >= 3:
                        print(f"🔄 连续失败{consecutive_failures}次，重置模型状态...")
                        reset_model_state()
                        consecutive_failures = 0

            except Exception as e:
                print(f"❌ 重新生成时出错: {e}")

            # 每处理1个样本就保存一次
            print(f"💾 保存结果... (重新生成进度: {idx+1}/{len(retry_indices)})")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 每处理100个样本清理一次GPU内存
            if (idx + 1) % 100 == 0:
                print("🧹 清理GPU内存...")
                clear_gpu_memory()

        print(f"✅ 第一阶段完成：重新生成了 {len(retry_indices)} 个备用方案样本")

    # 第二阶段：继续生成新样本（如果选择了的话）
    if continue_generation:
        if start_index >= remaining_samples:
            print(f"✅ 所有数据已处理完成！总共 {len(results)} 条")
            return

        print(f"\n📊 第二阶段：处理第 {start_index + 1} 到第 {remaining_samples} 条新数据...")
        processing_indices = list(range(start_index, remaining_samples))
        total_to_process = remaining_samples - start_index

        for idx, i in enumerate(processing_indices):
            entry = samples[i]
            sample_info = get_sample_info(entry, dataset_config)

            question_id = sample_info['question_id']
            question = sample_info['question']
            image_name = sample_info['image_name']

            current_progress = idx + 1
            # 正常处理模式
            print(f"\n📋 处理样本 {i+1}/{remaining_samples} (进度: {current_progress}/{total_to_process}, ID: {question_id})")
            print(f"   问题: {question[:100]}...")

            # 查找图片文件
            data_format = sample_info.get('data_format', dataset_config.get('data_format', ''))
            image_path = find_image_file(image_name, dataset_config['image_folder'], data_format)

            if not image_path:
                print(f"❌ 找不到图片: {image_name}")
                # 仍然保存条目，但没有bbox
                result_entry = dict(sample_info)
                result_entry["bbox_analysis"] = None
                results.append(result_entry)
                continue

            try:
                # 简化版本：直接分析图片和问题生成bbox
                print("🔍 分析图片和问题，生成相关bbox...")
                bbox_analysis = generate_bboxes_for_question(image_path, question)

                # 构建结果条目
                result_entry = dict(sample_info)
                result_entry["bbox_analysis"] = bbox_analysis

                # 正常模式：追加到结果列表
                results.append(result_entry)

                # 统计信息和失败处理
                if bbox_analysis and bbox_analysis.get('relevant_elements'):
                    bbox_count = len(bbox_analysis['relevant_elements'])
                    print(f"✅ 识别了 {bbox_count} 个相关元素")
                    consecutive_failures = 0  # 重置失败计数器
                else:
                    print("⚠️ 未识别到相关元素")
                    consecutive_failures += 1

                    # 连续失败太多次时重置模型状态
                    if consecutive_failures >= 3:
                        print(f"🔄 连续失败{consecutive_failures}次，重置模型状态...")
                        reset_model_state()
                        consecutive_failures = 0

            except Exception as e:
                print(f"❌ 处理样本时出错: {e}")
                # 保存基本信息
                result_entry = dict(sample_info)
                result_entry["bbox_analysis"] = None
                # 正常模式：追加到结果列表
                results.append(result_entry)

            # 每处理1个样本就保存一次（实时保存）
            print(f"💾 保存结果... (总进度: {len(results)}/{total_samples:,})")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # 每处理100个样本清理一次GPU内存
            if (i + 1) % 100 == 0:
                print("🧹 清理GPU内存...")
                clear_gpu_memory()
    else:
        print("✅ 跳过第二阶段：不继续生成新样本")

    # 最终保存
    print(f"\n💾 保存最终结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 统计信息
    total_with_bbox = sum(1 for r in results if r.get('bbox_analysis') and r['bbox_analysis'] and r['bbox_analysis'].get('relevant_elements'))
    total_bbox_count = sum(
        len(r.get('bbox_analysis', {}).get('relevant_elements', []))
        for r in results
        if r.get('bbox_analysis') is not None
    )

    # 统计匹配质量信息
    similarity_scores = []
    ocr_confidences = []
    quality_scores = []

    for r in results:
        bbox_analysis = r.get('bbox_analysis', {})
        elements = bbox_analysis.get('relevant_elements', [])
        for element in elements:
            match_info = element.get('match_info', {})
            if match_info:
                similarity = match_info.get('semantic_similarity')
                if similarity is not None:
                    similarity_scores.append(similarity)

                confidence = match_info.get('ocr_confidence')
                if confidence is not None:
                    ocr_confidences.append(confidence)

                quality_score = match_info.get('match_quality_score')
                if quality_score is not None:
                    quality_scores.append(quality_score)

    print(f"\n📊 处理完成统计:")
    print(f"   总样本数: {len(results)}")
    print(f"   有bbox的样本: {total_with_bbox}")
    print(f"   总bbox数量: {total_bbox_count}")
    print(f"   成功率: {total_with_bbox/len(results)*100:.1f}%")

    # 匹配质量分数统计
    if quality_scores:
        excellent_count = sum(1 for score in quality_scores if score >= 0.9)
        good_count = sum(1 for score in quality_scores if 0.8 <= score < 0.9)
        acceptable_count = sum(1 for score in quality_scores if 0.6 <= score < 0.8)
        poor_count = sum(1 for score in quality_scores if 0.5 <= score < 0.6)
        very_poor_count = sum(1 for score in quality_scores if score < 0.5)
        total_matches = len(quality_scores)

        print(f"\n🎯 匹配质量分布:")
        print(f"   优秀 (≥0.9): {excellent_count} ({excellent_count/total_matches*100:.1f}%)")
        print(f"   良好 (0.8-0.9): {good_count} ({good_count/total_matches*100:.1f}%)")
        print(f"   可接受 (0.6-0.8): {acceptable_count} ({acceptable_count/total_matches*100:.1f}%)")
        print(f"   较差 (0.5-0.6): {poor_count} ({poor_count/total_matches*100:.1f}%)")
        if very_poor_count > 0:
            print(f"   很差 (<0.5): {very_poor_count} ({very_poor_count/total_matches*100:.1f}%)")

        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"\n📊 匹配质量分数统计:")
        print(f"   平均质量分数: {avg_quality:.3f}")
        print(f"   最高质量分数: {max(quality_scores):.3f}")
        print(f"   最低质量分数: {min(quality_scores):.3f}")

    # 相似度和置信度统计
    if similarity_scores:
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        print(f"\n📈 语义相似度统计:")
        print(f"   平均相似度: {avg_similarity:.3f}")
        print(f"   最高相似度: {max(similarity_scores):.3f}")
        print(f"   最低相似度: {min(similarity_scores):.3f}")

    if ocr_confidences:
        avg_confidence = sum(ocr_confidences) / len(ocr_confidences)
        print(f"\n🔍 OCR置信度统计:")
        print(f"   平均置信度: {avg_confidence:.3f}")
        print(f"   最高置信度: {max(ocr_confidences):.3f}")
        print(f"   最低置信度: {min(ocr_confidences):.3f}")


def calculate_semantic_similarity(text1, text2):
    """
    计算两个文本的语义相似性
    使用多种策略：精确匹配、包含关系、词汇重叠、编辑距离等
    返回0-1之间的相似度分数
    """
    if not text1 or not text2:
        return 0.0

    # 预处理：转换为小写，去除多余空格
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()

    # 1. 精确匹配 - 最高分
    if text1 == text2:
        return 1.0

    # 2. 包含关系 - 高分
    if text1 in text2 or text2 in text1:
        # 计算包含比例
        shorter = min(len(text1), len(text2))
        longer = max(len(text1), len(text2))
        return 0.9 * (shorter / longer)

    # 3. 词汇级别的匹配
    import re
    words1 = set(re.findall(r'\b\w+\b', text1))
    words2 = set(re.findall(r'\b\w+\b', text2))

    if not words1 or not words2:
        return 0.0

    # 计算词汇重叠率
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    jaccard_similarity = len(intersection) / len(union) if union else 0.0

    # 4. 字符级别的相似性（编辑距离）
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # 计算编辑距离相似性
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 0.0

    edit_distance = levenshtein_distance(text1, text2)
    edit_similarity = 1 - (edit_distance / max_len)

    # 5. 综合评分
    # 词汇重叠权重更高，编辑距离作为补充
    final_score = 0.7 * jaccard_similarity + 0.3 * edit_similarity

    # 确保分数在0-1范围内
    return max(0.0, min(1.0, final_score))


# 匹配和qwen相符的ocr文本
def match_content_with_ocr(qwen_elements, image_path, _=""):
    """
    简化的语义匹配策略：
    1. 优先在粗略区域内找分数>0.5的最高分匹配
    2. 如果粗略区域内没找到，再在整张图找分数最高的匹配
    3. 相似度相同时，选择OCR置信度最高的
    """
    try:
        # 初始化OCR - 直接加载，不使用函数
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

        # 1.获取OCR结果
        try:
            if hasattr(ocr, 'predict'):
                ocr_results = ocr.predict(image_path, cls=True)
            else:
                ocr_results = ocr.ocr(image_path, cls=True)
        except Exception as e:
            print(f"❌ OCR处理出错: {e}")
            ocr_results = None

        if not ocr_results or not ocr_results[0]:
            print("⚠️ OCR未检测到文本，使用原始结果")
            return qwen_elements

        # 调试：显示OCR识别到的所有文本
        print(f"🔍 OCR识别到 {len(ocr_results[0])} 个文本块:")
        for i, ocr_line in enumerate(ocr_results[0][:5]):  # 只显示前5个
            ocr_text = ocr_line[1][0].strip()
            confidence = ocr_line[1][1]
            print(f"   {i+1}. '{ocr_text}' (conf: {confidence:.2f})")

        # 获取图像尺寸
        from PIL import Image
        image = Image.open(image_path)
        image_width, image_height = image.size

        enhanced_elements = []

        # 2. 找到和qwen匹配的bbox
        for qwen_element in qwen_elements:
            # 初始化变量避免作用域问题
            rough_x1, rough_y1, rough_x2, rough_y2 = 0, 0, image_width, image_height
            # 处理两种可能的格式：字符串或字典
            if isinstance(qwen_element, str):
                qwen_content = qwen_element.strip().lower()
                rough_bbox = None  # 没有粗略区域信息
            else:
                # 优先使用description进行OCR匹配，如果没有则使用content_relation，最后使用selection_reason
                qwen_content = qwen_element.get('description',
                              qwen_element.get('content_relation',
                              qwen_element.get('selection_reason', ''))).strip().lower()
                rough_bbox = qwen_element.get('rough_bbox')  # 从qwen处获取粗略区域

            best_match = None

            print(f"🔍 寻找文本: '{qwen_content}'")

            # 2.1 将粗略区域转换为像素坐标，并适当扩大搜索范围
            if rough_bbox:
                print(f"   在粗略区域: {rough_bbox}")
                # 扩大边距，避免区域太小
                margin_x = int(MARGIN_RATIO * image_width)  # 水平边距
                margin_y = int(MARGIN_RATIO * image_height)  # 垂直边距

                rough_x1 = max(0, int(rough_bbox[0] * image_width) - margin_x)
                rough_y1 = max(0, int(rough_bbox[1] * image_height) - margin_y)
                rough_x2 = min(image_width, int(rough_bbox[2] * image_width) + margin_x)
                rough_y2 = min(image_height, int(rough_bbox[3] * image_height) + margin_y)

                print(f"   扩大后像素区域: [{rough_x1}, {rough_y1}, {rough_x2}, {rough_y2}]")
            else:
                print(f"   在全图区域: [{rough_x1}, {rough_y1}, {rough_x2}, {rough_y2}]")

            # 收集所有候选匹配项
            all_candidates = []

            # 遍历所有OCR结果，计算相似度
            for ocr_line in ocr_results[0]:
                bbox_points = ocr_line[0]
                ocr_text = ocr_line[1][0].strip().lower()
                confidence = ocr_line[1][1]

                # 计算OCR文本的中心点
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)

                # 检查是否在粗略区域内
                in_rough_area = (rough_x1 <= center_x <= rough_x2 and
                               rough_y1 <= center_y <= rough_y2)

                # 计算文本相似度 - 使用语义相似性匹配
                text_match = False
                score = 0

                # 计算语义相似性
                score = calculate_semantic_similarity(qwen_content, ocr_text)

                # 如果相似度超过0.5，认为匹配成功（因为我们会选择最高分的）
                if score >= 0.5:
                    text_match = True
                    # 显示高质量匹配的详细信息
                    if score >= 0.95:
                        print(f"   🎯 高质量匹配 (相似度: {score:.3f}): '{qwen_content}' ≈ '{ocr_text}'")
                    elif score >= 0.9:
                        print(f"   ✅ 良好匹配 (相似度: {score:.3f}): '{qwen_content}' ≈ '{ocr_text}'")
                else:
                    text_match = False

                if text_match:
                    all_candidates.append({
                        'bbox_points': bbox_points,
                        'ocr_text': ocr_text,
                        'confidence': confidence,
                        'score': score,
                        'center': (int(center_x), int(center_y)),
                        'in_rough_area': in_rough_area
                    })

            # 简化的语义匹配策略：清晰的优先级
            best_match = None
            match_strategy = ""

            if all_candidates:
                # 过滤出置信度足够的候选项
                valid_candidates = [c for c in all_candidates if c['confidence'] > CONFIDENCE_THRESHOLD]

                if valid_candidates:
                    if rough_bbox:
                        # 优先级1: 粗略区域内分数>0.5的最高分匹配
                        rough_area_candidates = [c for c in valid_candidates if c['in_rough_area'] and c['score'] > 0.5]
                        if rough_area_candidates:
                            best_match = max(rough_area_candidates, key=lambda x: (x['score'], x['confidence']))
                            match_strategy = f"粗略区域内最佳匹配 (相似度: {best_match['score']:.3f})"
                        else:
                            # 优先级2: 整张图找分数最高的匹配
                            best_match = max(valid_candidates, key=lambda x: (x['score'], x['confidence']))
                            match_strategy = f"全图最佳匹配 (相似度: {best_match['score']:.3f})"
                    else:
                        # 没有粗略区域时，直接选择全图最高分
                        best_match = max(valid_candidates, key=lambda x: (x['score'], x['confidence']))
                        match_strategy = f"全图最佳匹配 (相似度: {best_match['score']:.3f})"



            if best_match:
                # 转换OCR坐标为归一化格式
                bbox_points = best_match['bbox_points']
                x_coords = [point[0] for point in bbox_points]
                y_coords = [point[1] for point in bbox_points]
                x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

                # 归一化坐标
                norm_bbox = [
                    round(x1 / image_width, 3),
                    round(y1 / image_height, 3),
                    round(x2 / image_width, 3),
                    round(y2 / image_height, 3)
                ]

                # 处理两种可能的格式：字符串或字典
                if isinstance(qwen_element, str):
                    content = qwen_element
                else:
                    # 优先使用description，如果没有则使用content_relation，最后使用selection_reason
                    content = qwen_element.get('description',
                             qwen_element.get('content_relation',
                             qwen_element.get('selection_reason', qwen_element)))

                # 计算与粗略区域的距离（既然进入OCR增强，肯定有粗略区域）
                # 使用原始的rough_bbox来计算距离，而不是扩大后的区域
                if rough_bbox and len(rough_bbox) == 4:
                    # 使用原始粗略区域的中心点
                    original_rough_center_x = (rough_bbox[0] + rough_bbox[2]) * image_width / 2
                    original_rough_center_y = (rough_bbox[1] + rough_bbox[3]) * image_height / 2

                    distance_to_rough = ((best_match['center'][0] - original_rough_center_x)**2 +
                                       (best_match['center'][1] - original_rough_center_y)**2)**0.5
                    # 归一化距离（相对于图像对角线长度）
                    diagonal_length = (image_width**2 + image_height**2)**0.5
                    distance_to_rough = round(distance_to_rough / diagonal_length, 4)

                    print(f"   📏 距离粗略区域中心: {distance_to_rough:.4f} (归一化)")
                else:
                    # 理论上不应该到这里，因为进入OCR增强说明有粗略区域
                    print(f"⚠️ 警告：进入OCR增强但没有有效粗略区域: {rough_bbox}")
                    distance_to_rough = None

                # 构建match_info，不包含in_rough_area参数
                match_info = {
                    'semantic_similarity': round(best_match['score'], 4),  # 语义相似度
                    'ocr_confidence': round(best_match['confidence'], 4),  # OCR置信度
                    'ocr_text': best_match['ocr_text'],  # 实际匹配的OCR文本
                    'match_strategy': match_strategy,  # 匹配策略
                    'distance_to_rough': distance_to_rough,  # 总是包含距离信息
                    'match_quality_score': round(best_match['score'], 4)  # 匹配质量分数 (0-1)
                }

                # 保存所有原始字段
                element_with_match_info = {
                    'bbox': norm_bbox,
                    'match_info': match_info
                }

                # 保存原始的所有字段
                if isinstance(qwen_element, dict):
                    # 保存description, selection_reason, content_relation等字段
                    for key in ['description', 'selection_reason', 'content_relation']:
                        if key in qwen_element:
                            element_with_match_info[key] = qwen_element[key]

                    # 保存rough_bbox字段（Qwen2-VL生成的原始bbox）
                    if 'rough_bbox' in qwen_element:
                        element_with_match_info['rough_bbox'] = qwen_element['rough_bbox']

                    # 为了向后兼容，也保存content字段（如果存在）
                    if 'content' in qwen_element:
                        element_with_match_info['content'] = qwen_element['content']
                else:
                    # 如果是字符串格式，保存为content
                    element_with_match_info['content'] = content

                enhanced_elements.append(element_with_match_info)

                print(f"✅ 匹配成功({match_strategy}): '{qwen_content}' -> OCR: '{best_match['ocr_text']}' (相似度: {best_match['score']:.3f}) -> {norm_bbox}")
            else:
                # 分析未匹配的原因
                total_ocr_texts = len(ocr_results[0]) if ocr_results and ocr_results[0] else 0
                candidates_found = len(all_candidates)

                # 找到最高相似度（即使低于阈值）
                best_similarity = 0
                best_ocr_text = ""
                if ocr_results and ocr_results[0]:
                    for ocr_line in ocr_results[0]:
                        ocr_text = ocr_line[1][0].strip().lower()
                        similarity = calculate_semantic_similarity(qwen_content, ocr_text)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_ocr_text = ocr_text

                print(f"⚠️ 未找到匹配: '{qwen_content}' (OCR文本总数: {total_ocr_texts}, 候选数: {candidates_found}, 最高相似度: {best_similarity:.3f} vs '{best_ocr_text}')")
                
                # OCR匹配失败时，保留原始元素（包含rough_bbox）
                if isinstance(qwen_element, dict):
                    enhanced_elements.append(qwen_element)
                else:
                    enhanced_elements.append({'content': qwen_element})

        return enhanced_elements

    except Exception as e:
        print(f"❌ OCR匹配失败: {e}")
        return qwen_elements



if __name__ == "__main__":

    # 验证CUDA设置
    if torch.cuda.is_available():
        print(f"🔧 当前可见GPU数量: {torch.cuda.device_count()}")
        print(f"🔧 当前GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA不可用")
        exit(1)

    # 选择数据集
    dataset_key, dataset_config = select_dataset()

    # 处理数据
    max_samples = dataset_config['default_max_samples']

    process_samples_with_config(dataset_config, max_samples)

