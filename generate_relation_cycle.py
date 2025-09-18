"""
Qwen2-VL推理链构建系统
=====================
基于bbox信息构建从问题到答案的推理链：

核心思路：
问题关键词 → bbox1 → bbox2 → ... → 答案

推理关系：
1. 先后关系 - 有逻辑顺序，A必须在B之前
2. 并列关系 - 同等重要，可以并行处理
3. 没有关系 - 不相关，不参与推理链

输出：结构化的推理链，显示从问题到答案的完整路径
"""

import os
import sys
import json

def select_gpu_before_torch():
    """在导入torch之前选择GPU"""
    print("🚀 Qwen2-VL推理链构建器")
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
# 检查是否已经设置了CUDA_VISIBLE_DEVICES（被其他脚本调用时）
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # 只有在没有预设GPU时才进行交互式选择
    selected_gpu = select_gpu_before_torch()
    os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
    print(f"🔧 设置CUDA_VISIBLE_DEVICES = {selected_gpu}")
else:
    # 使用已经设置的GPU
    selected_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    print(f"🔧 使用预设的CUDA_VISIBLE_DEVICES = {selected_gpu}")

# 🚨 重要：必须在导入torch之前设置CUDA环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:False'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 现在可以安全导入torch和其他依赖
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==================== 生成控制配置 ====================
# 控制重新生成的范围
REGENERATION_CONFIG = {
    # 重新生成所有多bbox问题 (bbox_count > 1)
    "regenerate_multi_bbox": False,

    # 重新生成所有单bbox问题 (bbox_count == 1)
    "regenerate_single_bbox": False,

    # 重新生成所有问题 (忽略现有结果)
    "regenerate_all": False,

    # 自动重新生成失败的数据 (推理步数为0或错误状态)
    "regenerate_failed": True,

    # 只处理特定bbox数量的问题 (None表示处理所有)
    "target_bbox_count": None,  # 例如: 2, 3, 4 等

    # 最小bbox数量阈值 (小于此数量的问题会被跳过)
    "min_bbox_count": 1,

    # 最大bbox数量阈值 (大于此数量的问题会被跳过)
    "max_bbox_count": None,  # None表示无限制

    # 是否跳过已有推理链的问题
    "skip_existing": True,

    # 详细日志输出
    "verbose_logging": True,

    # 新增：bbox生成模式选择
    "bbox_generation_mode": "auto"  # "single", "multi", "auto"
}

# 数据集配置
DATASETS = {
    "docvqa": { # 13.0 ready 11995
        "name": "DocVQA",
        "image_folder": "playground/data/cot/docvqa",
        "bbox_file": "images_bbox/DocVQA_complex_one_agent.json",
        "output_file": "reasoning_chains/DocVQA_complex_reasoning_chains_one_agent.json",
        "image_id_field": "image_name",
        "question_id_field": "question_id",
        "default_max_samples": None  # 使用所有数据
    },
    "infovqa": {  # 13.1  1.4w/21668
        "name": "InfoVQA",
        "image_folder": "playground/data/cot/infographicsvqa",
        "bbox_file": "images_bbox/InfoVQA_complex_one_agent.json",
        "output_file": "reasoning_chains/InfoVQA_complex_reasoning_chains_one_agent.json",
        "image_id_field": "image_name",
        "question_id_field": "question_id",
        "default_max_samples": 21668  # 使用所有数据
    },
    "textvqa": { # 13.2 0.79 W/12508
        "name": "TextVQA", # 13.2 ready
        "image_folder": "playground/data/cot/textvqa",
        "bbox_file": "images_bbox/TextVQA_complex_one_agent.json",
        "output_file": "reasoning_chains/TextVQA_complex_reasoning_chains_one_agent.json",
        "image_id_field": "image_name",
        "question_id_field": "question_id",
        "default_max_samples": None  # 使用所有数据
    },
    "visual7w": { # ready 148.1/17954
        "name": "Visual7W",
        "image_folder": "playground/data/cot/v7w",
        "bbox_file": "images_bbox/Visual7W_complex_one_agent.json",
        "output_file": "reasoning_chains/Visual7W_complex_reasoning_chains_one_agent.json",
        "image_id_field": "image_name",
        "question_id_field": "question_id",
        "default_max_samples": None  # 使用所有数据
    },
    "gqa": { # 148.2/37592
        "name": "GQA",
        "image_folder": "playground/data/cot/gqa",
        "bbox_file": "images_bbox/GQA_complex_one_agent.json",
        "output_file": "reasoning_chains/GQA_complex_reasoning_chains_one_agent.json",
        "image_id_field": "image_name",
        "question_id_field": "question_id",
        "default_max_samples": None  # 使用所有数据
    },
    "vqav2": { # ready 148.3/35383
        "name": "VQAv2",
        "image_folder": "playground/data/cot/coco",
        "bbox_file": "images_bbox/VQAv2_complex_one_agent.json",
        "output_file": "reasoning_chains/VQAv2_complex_reasoning_chains_one_agent.json",
        "image_id_field": "image_name",
        "question_id_field": "question_id",
        "default_max_samples": None  # 使用所有数据
    }
}

# 配置路径
MODEL_PATH = "Qwen2-VL-7B-Instruct"

# 全局变量
model = None
processor = None

def crop_bbox_from_image(image, bbox_info):
    """从图像中裁剪bbox区域"""
    try:
        # 获取bbox坐标
        bbox_coords = bbox_info.get('bbox_coordinates', bbox_info.get('bbox', []))

        if not bbox_coords or len(bbox_coords) != 4:
            return None

        # 获取图像尺寸
        img_width, img_height = image.size

        # 处理归一化坐标 (0-1) 或像素坐标
        if all(coord <= 1.0 for coord in bbox_coords):
            # 归一化坐标，转换为像素坐标
            x1 = int(bbox_coords[0] * img_width)
            y1 = int(bbox_coords[1] * img_height)
            x2 = int(bbox_coords[2] * img_width)
            y2 = int(bbox_coords[3] * img_height)
        else:
            # 已经是像素坐标
            x1, y1, x2, y2 = map(int, bbox_coords)

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        # 确保x2 > x1, y2 > y1
        if x2 <= x1 or y2 <= y1:
            return None

        # 裁剪图像
        cropped_image = image.crop((x1, y1, x2, y2))

        # 如果裁剪区域太小，适当放大
        if cropped_image.size[0] < 50 or cropped_image.size[1] < 50:
            # 放大到至少100x100
            new_size = (max(100, cropped_image.size[0]), max(100, cropped_image.size[1]))
            cropped_image = cropped_image.resize(new_size, Image.Resampling.LANCZOS)

        return cropped_image

    except Exception as e:
        print(f"❌ 裁剪bbox失败: {e}")
        return None

def extract_question_keywords(question):
    """提取问题中的关键词作为推理链的起点"""
    import re

    question_lower = question.lower()

    # 移除常见的疑问词和停用词
    stop_words = {'what', 'which', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    # 提取关键词
    words = re.findall(r'\b\w+\b', question_lower)
    keywords = [word for word in words if word not in stop_words and len(word) > 2]

    # 提取数字和年份
    numbers = re.findall(r'\b\d{4}\b|\b\d+\b', question)

    # 提取引号中的内容
    quoted = re.findall(r'"([^"]*)"', question)

    all_keywords = keywords + numbers + quoted

    return {
        "keywords": keywords[:5],  # 限制关键词数量
        "numbers": numbers,
        "quoted_terms": quoted,
        "all_terms": all_keywords
    }


def analyze_spatial_relationships(bbox_list, used_regions, question_type="parallel"):
    """分析bbox的空间关系，为parallel推理提供提示"""
    if not bbox_list or len(bbox_list) < 2:
        return ""

    # 获取已使用的bbox坐标
    used_coords = []
    for idx in used_regions:
        if idx < len(bbox_list):
            coords = bbox_list[idx].get('bbox_coordinates', bbox_list[idx].get('bbox', []))
            if len(coords) == 4:
                used_coords.append((idx, coords))

    if not used_coords:
        return ""

    # 分析可用的bbox
    available_regions = []
    for i, bbox in enumerate(bbox_list):
        if i not in used_regions:
            coords = bbox.get('bbox_coordinates', bbox.get('bbox', []))
            if len(coords) == 4:
                available_regions.append((i, coords))

    if not available_regions:
        return ""

    # 分析空间关系
    spatial_hints = []
    comparison_hints = []

    for used_idx, used_coord in used_coords:
        used_x1, used_y1, used_x2, used_y2 = used_coord
        used_center_x = (used_x1 + used_x2) / 2
        used_center_y = (used_y1 + used_y2) / 2

        # 找到在相似位置的region
        horizontal_aligned = []
        vertical_aligned = []
        nearby_regions = []

        for region_idx, coords in available_regions:
            x1, y1, x2, y2 = coords
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 检查水平对齐 (相似的y坐标) - 适合表格行
            if abs(center_y - used_center_y) < 0.1:  # 10%的容差
                horizontal_aligned.append(f"Region {region_idx}")

            # 检查垂直对齐 (相似的x坐标) - 适合表格列
            if abs(center_x - used_center_x) < 0.1:  # 10%的容差
                vertical_aligned.append(f"Region {region_idx}")

            # 检查邻近区域 - 适合比较/排序
            distance = ((center_x - used_center_x)**2 + (center_y - used_center_y)**2)**0.5
            if distance < 0.3:  # 30%的距离内
                nearby_regions.append(f"Region {region_idx}")

        if horizontal_aligned:
            spatial_hints.append(f"Same row regions: {', '.join(horizontal_aligned)}")

        if vertical_aligned:
            spatial_hints.append(f"Same column regions: {', '.join(vertical_aligned)}")

        if nearby_regions:
            comparison_hints.append(f"Nearby comparison regions: {', '.join(nearby_regions)}")

    # 根据问题类型生成不同的提示
    hint_text = ""
    if question_type == "parallel":
        if spatial_hints:
            hint_text += f"\n🔍 Spatial Layout: {'; '.join(spatial_hints)}."
        if comparison_hints:
            hint_text += f"\n📊 For comparison questions: Consider {'; '.join(comparison_hints)}."

        # 添加排序/比较的特殊提示
        hint_text += f"\n💡 Comparison Strategy: For ranking questions (highest/lowest/most/least), look for regions with similar content types (numbers, percentages, names) that can be compared directly."

    return hint_text

def generate_content_based_reasoning(question, bbox_content, description, question_type="parallel"):
    """基于实际bbox内容生成推理，避免幻觉"""

    # 清理和标准化内容
    content = bbox_content or description or ""
    content = content.strip()

    if not content:
        return "Selected region contains relevant information"

    # 提取问题中的关键词
    question_lower = question.lower()
    content_lower = content.lower()

    # 检查内容是否直接包含答案
    question_words = question_lower.split()
    content_words = content_lower.split()

    # 寻找共同词汇
    common_words = set(question_words) & set(content_words)

    # 针对排序/比较问题的特殊处理
    if question_type == "parallel" and any(word in question_lower for word in ['second', 'third', 'highest', 'lowest', 'most', 'least']):
        # 检查是否包含数值信息
        import re
        numbers = re.findall(r'\d+\.?\d*%?', content)
        if numbers:
            return f"'{content}' contains numerical value {numbers[0]} which can be compared with other regions to determine ranking"
        elif any(word in content_lower for word in ['first', 'second', 'third', 'top', 'bottom']):
            return f"'{content}' contains ranking information that helps determine the position in comparison"
        else:
            return f"'{content}' represents one option that needs to be compared with others to answer the ranking question"

    if common_words:
        # 如果有共同词汇，说明内容相关
        if any(word in content_lower for word in ['dpi', 'dots', 'inch']):
            return f"'{content}' provides the specific measurement requested in the question"
        elif any(word in content_lower for word in ['font', 'serif', 'sans']):
            return f"'{content}' mentions the font type relevant to the question"
        elif any(word in content_lower for word in ['color', 'blue', 'red', 'green']):
            return f"'{content}' specifies the color information asked about"
        elif any(word in content_lower for word in ['number', 'count', 'total']):
            return f"'{content}' provides numerical information relevant to the question"
        else:
            return f"'{content}' contains keywords relevant to the question"
    else:
        # 如果没有明显共同词汇，生成通用但具体的推理
        return f"Region contains '{content}' which may provide context for answering the question"


def determine_single_bbox_role(question, bbox_content, suggested_role=None):
    """为单个bbox确定角色"""
    if suggested_role and suggested_role in ["direct_answer", "evidence", "keyword_match"]:
        return suggested_role

    question_lower = question.lower()
    content_lower = bbox_content.lower()

    # 检查是否包含直接答案的关键词
    if any(word in content_lower for word in ['yes', 'no', 'true', 'false']):
        return "direct_answer"

    # 检查是否包含问题关键词
    question_words = set(question_lower.split())
    content_words = set(content_lower.split())
    if len(question_words.intersection(content_words)) > 0:
        return "keyword_match"

    # 默认为证据
    return "evidence"

def analyze_question_type(question):
    """分析问题类型，判断需要顺序链条还是并列链条"""
    question_lower = question.lower()

    # 真正的顺序关系指示词（时间/步骤顺序）
    sequential_indicators = [
        'then', 'next', 'after', 'before', 'step', 'stage', 'process', 'sequence',
        'chronological', 'timeline', 'follow', 'subsequent', 'prior', 'earlier',
        'later', 'initially', 'finally', 'lastly', 'procedure', 'workflow',
        'step by step', 'one by one', 'in order', 'in sequence'
    ]

    # 排序/比较关系指示词（需要并列比较）
    ranking_comparison_indicators = [
        'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
        'highest', 'lowest', 'most', 'least', 'best', 'worst', 'top', 'bottom',
        'popular', 'common', 'frequent', 'rare', 'maximum', 'minimum', 'largest', 'smallest',
        'biggest', 'tiniest', 'greater', 'lesser', 'more', 'fewer', 'rank', 'ranking',
        'compare', 'comparison', 'versus', 'vs', 'better', 'worse', 'superior', 'inferior'
    ]

    # 并列关系指示词
    parallel_indicators = [
        'and', 'also', 'both', 'either', 'or', 'all', 'each', 'every', 'multiple',
        'various', 'different', 'several', 'many', 'list', 'enumerate', 'include',
        'contain', 'comprise', 'as well as', 'in addition', 'together'
    ]

    # 计算指示词出现次数
    sequential_count = sum(1 for word in sequential_indicators if word in question_lower)
    ranking_count = sum(1 for word in ranking_comparison_indicators if word in question_lower)
    parallel_count = sum(1 for word in parallel_indicators if word in question_lower)

    # 特殊模式检测
    has_time_sequence = any(pattern in question_lower for pattern in [
        'what happens', 'how to', 'steps to', 'process of', 'procedure for',
        'how do you', 'what do you do', 'what should you do'
    ])

    has_listing = any(pattern in question_lower for pattern in [
        'what are', 'list all', 'name all', 'which ones', 'what types',
        'what kinds', 'what categories', 'enumerate'
    ])

    has_ranking_comparison = any(pattern in question_lower for pattern in [
        'which is the', 'which has the', 'which country', 'which state',
        'which city', 'what is the most', 'what is the least', 'what percentage',
        'where is the', 'where can', 'how much', 'how many', 'how long'
    ])

    # 特殊情况：包含step但询问具体步骤内容的问题
    has_step_inquiry = any(pattern in question_lower for pattern in [
        'what is the first step', 'what is the next step', 'what is the last step',
        'what follows after', 'what comes after', 'what happens in the'
    ])

    # 判断问题类型
    if has_time_sequence or (sequential_count > 0 and ranking_count == 0 and not has_ranking_comparison):
        # 真正的时间/步骤顺序
        return "sequential"
    elif (sequential_count > 0 and 'step' in question_lower and 'process' in question_lower):
        # 明确的步骤过程问题
        return "sequential"
    elif has_step_inquiry:
        # 询问具体步骤的问题
        return "sequential"
    elif has_ranking_comparison or ranking_count > 0 or has_listing or parallel_count > sequential_count:
        # 排序比较、列举、并列关系
        return "parallel"
    else:
        # 默认根据问题结构判断
        if 'what' in question_lower and ('are' in question_lower or 'all' in question_lower):
            return "parallel"
        elif 'which' in question_lower or 'what' in question_lower or 'where' in question_lower or 'how' in question_lower:
            # 大部分which/what/where/how问题都是查询选择，属于并列
            return "parallel"
        else:
            return "sequential"

def build_reasoning_chain_with_multi_qwen(image_path, question, bbox_list):
    """使用多轮Qwen模型逐步构建推理链"""
    print("🔗 开始使用多轮Qwen构建推理链...")

    # 第一步：提取问题关键词
    keywords = extract_question_keywords(question)
    print(f"🔑 关键词: {keywords['keywords']}")

    # 第二步：构建bbox信息摘要
    bbox_summary = []
    for i, bbox in enumerate(bbox_list):
        content = bbox.get('bbox_description', bbox.get('description', ''))
        bbox_summary.append(f"Region {i}: {content}")

    reasoning_steps = []
    used_regions = set()
    current_context = f"Question: {question}\nKeywords: {', '.join(keywords['keywords'])}"

    # 特殊处理：如果只有一个bbox，直接进行单步推理
    if len(bbox_list) == 1:
        print(f"📝 单bbox推理模式")

        # 分析问题类型（单bbox模式也需要）
        question_type = analyze_question_type(question)

        try:
            single_bbox = bbox_list[0]
            bbox_content = single_bbox.get('bbox_description', single_bbox.get('description', ''))
            description = single_bbox.get('description', bbox_content)

            # 构建单步推理prompt - 让Qwen生成简洁的推理内容
            single_step_prompt = f"""{current_context}

Available region:
Region: {bbox_content}

Task: Analyze how this region answers the question. Generate a concise explanation (max 30 words).

IMPORTANT: Base your analysis ONLY on what you can actually see in the bbox image above.

Instructions:
1. Look at the actual content visible in the bbox image
2. Start with the key information from the region (e.g., "100 DPI")
3. Explain how it directly answers the question
4. Use format: "[key info] directly answers/provides [question aspect]"
5. Do NOT make up content that is not visible in the bbox

Example format: "100 DPI directly answers the question 'how many dots' for printed medium"

Output format:
SELECTED_REGION: Region 0
ROLE: direct_answer/evidence
REASONING: [key info] directly answers/provides [question aspect]
RELATIONSHIP: none
"""

            # 调用Qwen进行单步分析
            step_result = call_qwen_single_step(image_path, single_step_prompt, bbox_list)

            if step_result:
                print(f"🤖 单步Qwen输出:\n{step_result}")
                selected_region, role, reasoning, relationship = parse_single_step_result(step_result)

                # 🔧 单框推理：优先使用Qwen的reasoning，如果表现不好则使用content和description
                if reasoning and reasoning.strip() and reasoning.strip() not in ["Single region analysis", "Analysis of selected region"]:
                    # 使用Qwen生成的推理
                    generated_reasoning = reasoning.strip()
                    if len(generated_reasoning) > 200:  # 限制最大长度
                        generated_reasoning = generated_reasoning[:200] + "..."
                else:
                    # 回退到使用content和description
                    generated_reasoning = generate_content_based_reasoning(question, bbox_content, description, question_type)

                # 智能判断单bbox的role
                determined_role = determine_single_bbox_role(question, bbox_content, role)

                reasoning_steps.append({
                    "step": 1,
                    "bbox_index": 0,
                    "bbox_content": bbox_content,  # 保留原始description作为参考
                    "description": description,
                    "generated_reasoning": generated_reasoning,  # 使用智能选择的推理
                    "role": determined_role,
                    "relationship_to_previous": "none",
                    "qwen_analysis": step_result,
                    "bbox_coordinates": single_bbox.get('bbox_coordinates', single_bbox.get('bbox', []))
                })

                print(f"✅ 单步推理完成: {bbox_content[:50]}... (角色: {role or 'direct_answer'})")
            else:
                print(f"❌ 单步推理失败，使用基于内容的推理")
                # 智能判断role
                determined_role = determine_single_bbox_role(question, bbox_content)

                # 使用基于内容的推理作为回退
                generated_reasoning = generate_content_based_reasoning(question, bbox_content, description, question_type)

                reasoning_steps.append({
                    "step": 1,
                    "bbox_index": 0,
                    "bbox_content": bbox_content,
                    "description": description,
                    "generated_reasoning": generated_reasoning,
                    "role": determined_role,
                    "relationship_to_previous": "none",
                    "qwen_analysis": "Single bbox fallback with content-based reasoning",
                    "bbox_coordinates": single_bbox.get('bbox_coordinates', single_bbox.get('bbox', []))
                })

        except Exception as e:
            print(f"❌ 单步推理出错: {e}")
            # 创建默认的单步推理
            single_bbox = bbox_list[0]
            bbox_content = single_bbox.get('bbox_description', single_bbox.get('description', ''))
            description = single_bbox.get('description', bbox_content)

            # 智能判断role
            determined_role = determine_single_bbox_role(question, bbox_content)

            # 异常情况下使用基于内容的推理
            generated_reasoning = generate_content_based_reasoning(question, bbox_content, description, question_type)

            reasoning_steps.append({
                "step": 1,
                "bbox_index": 0,
                "bbox_content": bbox_content,
                "description": description,
                "generated_reasoning": generated_reasoning,
                "role": determined_role,
                "relationship_to_previous": "none",
                "qwen_analysis": "Error fallback with content-based reasoning",
                "bbox_coordinates": single_bbox.get('bbox_coordinates', single_bbox.get('bbox', []))
            })

    else:
        # 多轮推理：每轮找下一个最相关的bbox
        print(f"🔄 多步推理模式，共{len(bbox_list)}个bbox")
        for step_num in range(1, len(bbox_list) + 1):  # 处理所有可用的bbox
            print(f"\n🔍 第{step_num}轮Qwen分析...")

            # 构建当前可用的bbox列表（排除已使用的）
            available_regions = []
            available_bbox_list = []
            for i, bbox in enumerate(bbox_list):
                if i not in used_regions:
                    content = bbox.get('bbox_description', bbox.get('description', ''))
                    available_regions.append(f"Region {i}: {content}")
                    available_bbox_list.append(bbox)

            if not available_regions:
                print("📝 所有区域已使用完毕")
                break

            # 分析问题类型
            question_type = analyze_question_type(question)

            # 构建当前步骤的prompt
            if step_num == 1:
                role_instruction = f"Find the region that best matches the question keywords and serves as the starting point. IMPORTANT: This is just the first step - you should explore multiple regions to gather comprehensive information before concluding. Try to use most of the available regions. Question type appears to be: {question_type}"
                expected_role = "keyword_match"
            else:
                if question_type == "parallel":
                    # 为parallel类型添加空间位置提示
                    spatial_hint = analyze_spatial_relationships(bbox_list, used_regions, question_type)

                    # 根据问题类型提供更具体的指导
                    question_lower = question.lower()
                    if any(word in question_lower for word in ['second', 'third', 'highest', 'lowest', 'most', 'least']):
                        role_instruction = f"Continue exploring regions containing comparable values/options for ranking comparison. You should examine multiple regions to gather all relevant options before concluding. Look for: 1) Numbers/percentages that can be compared, 2) Items in the same category (countries, products, etc.), 3) Regions with similar formatting/layout.{spatial_hint}"
                    elif any(word in question_lower for word in ['what are', 'list', 'which ones', 'all']):
                        role_instruction = f"Continue finding regions that contain additional items/options to complete the list. Explore multiple regions to ensure you capture all relevant items. Look for regions with similar content structure or formatting.{spatial_hint}"
                    else:
                        role_instruction = f"Continue exploring regions that provide additional information to answer the question comprehensively. Don't conclude too early - examine multiple regions to gather complete evidence. For ranking/comparison questions, look for regions with comparable content (numbers, percentages, names) that can be directly compared.{spatial_hint}"
                else:
                    role_instruction = f"Based on the previous steps, find the next region that logically follows in sequence to answer the question."
                expected_role = "next_step"

            # 计算使用进度
            used_count = len(used_regions)
            total_count = len(bbox_list)
            progress_info = f"Progress: Used {used_count}/{total_count} regions. Try to explore most regions before concluding."

            step_prompt = f"""{current_context}

{progress_info}

Question Type Analysis: {question_type}
- Sequential questions need step-by-step reasoning (A->B->C)
- Parallel questions need multiple independent evidence (A->B; A->C)
- For parallel questions: Look for regions in similar spatial positions (same row/column) as they often contain related information

Previous reasoning steps:
{chr(10).join([f"Step {s['step']}: Region {s['bbox_index']} - {s.get('generated_reasoning', s['bbox_content'][:50])}..." for s in reasoning_steps])}

Available regions for this step:
{chr(10).join(available_regions)}

Task: {role_instruction}

IMPORTANT: Only use ROLE: conclusion when you have explored most regions and gathered comprehensive information.

Output format:
SELECTED_REGION: [Region X] (X must be 0-based integer: 0, 1, 2, etc.)
ROLE: [{expected_role}/evidence/conclusion]
REASONING: [Why this region is selected and how it contributes to answering the question]
RELATIONSHIP: [How this region relates to previous steps: sequential/parallel/none]
"""

            try:
                # 调用Qwen进行单步分析 - 传递可用的bbox列表
                step_result = call_qwen_single_step(image_path, step_prompt, available_bbox_list)

                if step_result:
                    print(f"🤖 Qwen第{step_num}轮输出:\n{step_result}")
                    # 解析单步结果 - 使用原来的格式
                    selected_region, role, reasoning, relationship = parse_single_step_result(step_result)

                    # 🔧 修改停止条件：尽量使用所有bbox
                    # 计算已使用的区域数量
                    used_regions_count = len(used_regions)
                    total_regions = len(bbox_list)

                    # 只有在满足以下条件时才允许停止：
                    # 1. 角色是conclusion，并且
                    # 2. 已经使用了至少80%的区域，或者至少3个区域
                    min_required_regions = max(3, int(total_regions * 0.8))

                    if role == "conclusion":
                        if used_regions_count >= min_required_regions:
                            print(f"🎯 检测到conclusion角色且已使用{used_regions_count}/{total_regions}个区域，结束推理链")
                            break
                        else:
                            print(f"⚠️ 检测到conclusion但只使用了{used_regions_count}/{total_regions}个区域，继续推理")
                            print(f"   需要至少使用{min_required_regions}个区域")
                            # 强制改为evidence角色继续推理
                            role = "evidence"

                    # 检查区域选择的有效性
                    if selected_region is None or selected_region >= len(bbox_list):
                        print(f"❌ 无有效区域选择: {selected_region}")
                        break

                    # 🔧 防止重复选择同一区域
                    if selected_region in used_regions:
                        print(f"⚠️ 区域{selected_region}已被使用，寻找其他区域")
                        # 寻找未使用的区域
                        available_region_indices = [i for i in range(len(bbox_list)) if i not in used_regions]
                        if available_region_indices:
                            selected_region = available_region_indices[0]
                            print(f"🔧 自动选择未使用的区域: {selected_region}")
                        else:
                            print("❌ 所有区域都已使用，结束推理")
                            break

                    bbox_element = bbox_list[selected_region]
                    bbox_content = bbox_element.get('bbox_description', bbox_element.get('description', ''))
                    description = bbox_element.get('description', bbox_content)

                    # 🔧 多框推理：优先使用Qwen的reasoning，如果没有则使用content和description
                    if reasoning and reasoning.strip() and reasoning.strip() not in ["Analysis of selected region", "Single region analysis"]:
                        # 使用Qwen生成的推理
                        generated_reasoning = reasoning.strip()
                        if len(generated_reasoning) > 200:  # 限制最大长度
                            generated_reasoning = generated_reasoning[:200] + "..."
                    else:
                        # 回退到使用content和description
                        generated_reasoning = generate_content_based_reasoning(question, bbox_content, description, question_type)

                    reasoning_steps.append({
                        "step": step_num,
                        "bbox_index": selected_region,
                        "bbox_content": bbox_content,
                        "description": description,
                        "generated_reasoning": generated_reasoning,  # 使用检查后的推理
                        "role": role,
                        "relationship_to_previous": relationship,
                        "qwen_analysis": step_result,
                        "qwen_continue_decision": True,  # 默认继续
                        "qwen_explanation": "继续推理",
                        "bbox_coordinates": bbox_element.get('bbox_coordinates', bbox_element.get('bbox', []))
                    })

                    used_regions.add(selected_region)
                    current_context += f"\nStep {step_num}: Region {selected_region} - {bbox_content}"

                    print(f"✅ 选择Region {selected_region}: {bbox_content} (角色: {role})")
                    print(f"📝 推理内容: {generated_reasoning[:80]}...")

                    # 保留原有的conclusion角色判断作为备用
                    if role == "conclusion":
                        print("🎯 角色为conclusion，结束推理链")
                        break
                else:
                    print(f"❌ 第{step_num}轮Qwen分析失败")
                    break

            except Exception as e:
                print(f"❌ 第{step_num}轮分析出错: {e}")
                break



    # 使用原来的最终答案，不重新生成
    if reasoning_steps:
        print(f"\n✅ 推理链生成完成，保留原有最终答案")
        # 这里final_answer会在调用函数中从原始数据获取
        final_answer = None  # 标记使用原有答案
    else:
        final_answer = "无法生成答案"

    # 分析问题类型并生成推理链文本
    question_type = analyze_question_type(question)
    chain_text_info = generate_reasoning_chain_text(reasoning_steps, question, question_type)

    # 确定链类型
    if len(reasoning_steps) == 1:
        chain_type = "single_step"
    elif question_type == "parallel":
        chain_type = "parallel"
    elif any(step.get('relationship_to_previous') == 'sequential' for step in reasoning_steps):
        chain_type = "sequential"
    else:
        chain_type = "linear"

    return {
        "chain_type": chain_type,
        "reasoning_steps": reasoning_steps,
        "total_steps": len(reasoning_steps),
        "final_answer": final_answer,
        "keywords_used": keywords,
        "multi_round_analysis": True,
        "question_type": question_type,
        "chain_text": chain_text_info.get("chain_text", ""),
        "chain_format": chain_text_info.get("chain_format", ""),
        "reasoning_chain_description": f"Question type: {question_type}, Chain: {chain_text_info.get('chain_text', '')}"
    }

def call_qwen_single_step(image_path, prompt, available_bbox_list=None):
    """调用Qwen进行单步分析 - 包含原图和bbox裁剪图"""
    try:
        original_image = Image.open(image_path).convert('RGB')

        # 构建消息内容
        content = [
            {"type": "image", "image": original_image},
            {"type": "text", "text": "Original image above. "}
        ]

        # 如果有可用的bbox列表，添加裁剪图
        if available_bbox_list:
            content.append({"type": "text", "text": "Available regions shown below:"})

            for i, bbox in enumerate(available_bbox_list):
                # 裁剪bbox区域
                crop_image = crop_bbox_from_image(original_image, bbox)
                if crop_image:
                    content.append({"type": "image", "image": crop_image})
                    bbox_desc = bbox.get('bbox_description', bbox.get('description', ''))
                    content.append({"type": "text", "text": f"Region {i+1}: {bbox_desc}"})

        content.append({"type": "text", "text": f"\n{prompt}"})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.eos_token_id
            )

        generated_ids_trimmed = []
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
            if len(out_ids) > len(in_ids):
                generated_ids_trimmed.append(out_ids[len(in_ids):])
            else:
                generated_ids_trimmed.append(out_ids)

        if generated_ids_trimmed:
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            if output_text and len(output_text) > 0:
                return output_text[0].strip()

    except Exception as e:
        print(f"❌ Qwen单步调用失败: {e}")

    return None

def parse_single_step_result(qwen_output):
    """解析单步Qwen输出 - 支持新的CONTINUE格式"""
    import re

    selected_region = None
    role = "reasoning_step"
    reasoning = ""
    relationship = "none"
    should_continue = True  # 默认继续
    explanation = ""

    lines = qwen_output.split('\n')

    for line in lines:
        line = line.strip()

        # 解析是否继续 - 新增
        continue_patterns = [
            r'CONTINUE:\s*(YES|NO)',                  # CONTINUE: YES/NO
            r'继续:\s*(是|否)',                       # 继续: 是/否
        ]

        for pattern in continue_patterns:
            continue_match = re.search(pattern, line, re.IGNORECASE)
            if continue_match:
                continue_value = continue_match.group(1).upper()
                should_continue = continue_value in ['YES', '是']
                break

        # 解析解释 - 新增
        explanation_patterns = [
            r'EXPLANATION:\s*(.+)',                   # EXPLANATION: ...
            r'解释:\s*(.+)',                          # 解释: ...
        ]

        for pattern in explanation_patterns:
            explanation_match = re.search(pattern, line, re.IGNORECASE)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                break

        # 解析选择的区域 - 支持NONE和各种格式
        region_patterns = [
            r'SELECTED_REGION:\s*NONE',               # SELECTED_REGION: NONE
            r'SELECTED_REGION:\s*.*?Region\s*(\d+)',  # SELECTED_REGION: Region 1
            r'SELECTED_REGION:\s*(\d+)',              # SELECTED_REGION: 1
            r'Region\s*(\d+)',                        # Region 1
            r'选择.*?区域\s*(\d+)',                    # 选择区域1
            r'I\s+choose\s+Region\s*(\d+)',          # I choose Region 1
            r'The\s+selected\s+region\s+is\s+(\d+)', # The selected region is 1
        ]

        for pattern in region_patterns:
            if 'NONE' in pattern:
                none_match = re.search(pattern, line, re.IGNORECASE)
                if none_match:
                    selected_region = None
                    break
            else:
                region_match = re.search(pattern, line, re.IGNORECASE)
                if region_match:
                    selected_region = int(region_match.group(1))
                    break

        # 解析角色 - 支持多种格式
        role_patterns = [
            r'ROLE:\s*(.+)',                          # ROLE: keyword_match
            r'角色:\s*(.+)',                          # 角色: 关键词匹配
            r'This\s+region\s+serves\s+as\s+(.+)',   # This region serves as evidence
        ]

        for pattern in role_patterns:
            role_match = re.search(pattern, line, re.IGNORECASE)
            if role_match:
                role = role_match.group(1).strip().lower()
                break

        # 解析推理 - 支持多种格式，包括拼写错误
        reasoning_patterns = [
            r'REASONING:\s*(.+)',                     # REASONING: This region...
            r'REASONon:\s*(.+)',                      # REASONon: (处理拼写错误)
            r'推理:\s*(.+)',                          # 推理: 这个区域...
            r'Because\s+(.+)',                       # Because this region contains...
            r'This\s+region\s+(.+)',                 # This region contains the keyword
        ]

        for pattern in reasoning_patterns:
            reasoning_match = re.search(pattern, line, re.IGNORECASE)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                break

        # 解析关系
        relationship_match = re.search(r'RELATIONSHIP:\s*.*?(sequential|parallel|none)', line, re.IGNORECASE)
        if relationship_match:
            relationship = relationship_match.group(1).lower()

    # 如果没有解析到区域且应该继续，尝试从整个输出中提取数字
    if selected_region is None and should_continue:
        # 查找任何数字，可能是区域编号
        numbers = re.findall(r'\b(\d+)\b', qwen_output)
        if numbers:
            # 取第一个数字作为区域编号
            try:
                selected_region = int(numbers[0])
                print(f"🔍 从输出中提取到区域编号: {numbers[0]}")
            except:
                pass

    print(f"🔍 解析结果: 区域={selected_region}, 角色={role}")
    print(f"📝 推理: {reasoning[:80]}...")

    return selected_region, role, reasoning, relationship



def clean_text_content(text):
    """清理文本内容，去除重复和错误的部分"""
    if not text:
        return ""

    # 去除常见的重复模式
    import re

    # 去除 "This 2. This 3. This" 这样的重复模式
    text = re.sub(r'\s+This\s+\d+\.\s+This\s*', ' ', text)
    text = re.sub(r'\s+This\s+\d+\.\s*$', '', text)

    # 去除多余的空格和换行
    text = ' '.join(text.split())

    # 去除重复的句子片段
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)

    cleaned_text = '. '.join(unique_sentences)
    if cleaned_text and not cleaned_text.endswith('.'):
        cleaned_text += '.'

    return cleaned_text

def generate_reasoning_chain_text(reasoning_steps, question, chain_type="auto"):
    """生成推理链的文本表示，支持顺序链条和并列链条"""

    if not reasoning_steps:
        return {
            "chain_text": "",
            "chain_format": "empty",
            "step_count": 0,
            "question_type": chain_type
        }

    # 如果没有指定链类型，自动判断
    if chain_type == "auto":
        chain_type = analyze_question_type(question)

    # 为每个步骤生成描述文本
    step_descriptions = []
    for step in reasoning_steps:
        bbox_index = step.get('bbox_index', 0)

        # 优先使用Qwen生成的推理内容，而不是原始description
        generated_reasoning = step.get('generated_reasoning', '')
        # 为了向后兼容，也检查旧的reasoning字段
        if not generated_reasoning:
            generated_reasoning = step.get('reasoning', '')

        original_description = step.get('description', '')
        bbox_content = step.get('bbox_content', '')

        # 清理文本内容
        if generated_reasoning and generated_reasoning not in ["Single region analysis", "Analysis of selected region"]:
            # 使用Qwen生成的推理内容（排除默认的占位符）
            step_text = clean_text_content(generated_reasoning)
        elif original_description:
            # 回退到原始描述
            step_text = clean_text_content(original_description)
        elif bbox_content:
            # 最后回退到bbox内容
            step_text = clean_text_content(bbox_content)
        else:
            step_text = f"Region_{bbox_index}"

        # 确保文本不会太长
        if len(step_text) > 150:
            step_text = step_text[:150] + "..."

        step_descriptions.append({
            'index': bbox_index,
            'text': step_text,
            'role': step.get('role', 'reasoning_step'),
            'relationship': step.get('relationship_to_previous', 'none')
        })

    # 根据链类型生成不同格式的推理链
    if chain_type == "sequential":
        # 顺序链条：bbox0->bbox1->bbox2->...
        chain_text = " -> ".join([step['text'] for step in step_descriptions])
        chain_format = "sequential"

    elif chain_type == "parallel":
        # 并列链条：bbox0->bbox1; bbox0->bbox2; ...
        if len(step_descriptions) > 1:
            root_step = step_descriptions[0]
            parallel_steps = step_descriptions[1:]

            # 生成并列关系 - 平行问题使用分号分割，不使用箭头
            parallel_chains = [root_step['text']]
            for step in parallel_steps:
                parallel_chains.append(step['text'])

            chain_text = "; ".join(parallel_chains)
            chain_format = "parallel"
        else:
            chain_text = step_descriptions[0]['text']
            chain_format = "single"

    else:
        # 混合或其他类型，根据实际关系生成
        chain_parts = []
        current_chain = [step_descriptions[0]['text']]

        for i in range(1, len(step_descriptions)):
            step = step_descriptions[i]
            relationship = step['relationship']

            if relationship == 'sequential':
                current_chain.append(step['text'])
            elif relationship == 'parallel':
                # 开始新的并列分支
                if len(current_chain) > 1:
                    chain_parts.append(" -> ".join(current_chain))
                    current_chain = [step_descriptions[0]['text'], step['text']]
                else:
                    current_chain.append(step['text'])
            else:
                current_chain.append(step['text'])

        # 添加最后的链条
        if current_chain:
            chain_parts.append(" -> ".join(current_chain))

        chain_text = "; ".join(chain_parts) if len(chain_parts) > 1 else chain_parts[0] if chain_parts else ""
        chain_format = "mixed"

    return {
        "chain_text": chain_text,
        "chain_format": chain_format,
        "step_count": len(step_descriptions),
        "question_type": chain_type
    }


def select_bbox_generation_mode():
    """选择bbox生成模式"""
    print("\n🎯 选择bbox生成模式:")
    print("  1. 仅生成单bbox推理链 (bbox_count == 1)")
    print("     - 适用于简单的直接回答问题")
    print("     - 只处理包含1个相关区域的问题")
    print("  2. 仅生成多bbox推理链 (bbox_count > 1)")
    print("     - 适用于复杂的多步推理问题")
    print("     - 只处理包含2个或更多相关区域的问题")
    print("  3. 自动模式 (处理所有bbox数量)")
    print("     - 处理所有类型的问题，不限制bbox数量")

    while True:
        try:
            choice = input("请选择模式 (1/2/3): ").strip()
            if choice == "1":
                print("✅ 已选择：仅生成单bbox推理链")
                print("   将只处理bbox_count == 1的问题")
                return "single"
            elif choice == "2":
                print("✅ 已选择：仅生成多bbox推理链")
                print("   将只处理bbox_count > 1的问题")
                return "multi"
            elif choice == "3":
                print("✅ 已选择：自动模式 (处理所有bbox数量)")
                print("   将处理所有bbox数量的问题")
                return "auto"
            else:
                print("❌ 请输入 1、2 或 3")
        except KeyboardInterrupt:
            print("\n👋 用户取消操作")
            exit(0)
        except Exception as e:
            print(f"❌ 输入错误: {e}")



def get_sample_info(item, dataset_config):
    """根据数据集配置获取样本信息"""
    image_id_field = dataset_config['image_id_field']
    question_id_field = dataset_config['question_id_field']
    image_folder = dataset_config['image_folder']

    image_name = item[image_id_field]
    image_path = find_image_file(image_folder, image_name)

    if not image_path:
        return None

    return {
        'question_id': item[question_id_field],
        'image_name': image_name,
        'image_path': image_path,
        'question': item['question']
    }

def find_image_file(image_folder, image_name):
    """查找图像文件，自动检测扩展名和特殊格式"""
    # 常见的图像扩展名
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    # 如果已经有扩展名，直接检查
    if '.' in image_name:
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            return image_path

    # 尝试不同的扩展名
    for ext in extensions:
        image_path = os.path.join(image_folder, image_name + ext)
        if os.path.exists(image_path):
            return image_path

    # 特殊处理：VQAv2/COCO格式 (纯数字ID -> COCO_train2014_000000xxxxxx.jpg)
    if image_name.isdigit() and 'coco' in image_folder.lower():
        # 补齐到12位数字
        padded_id = image_name.zfill(12)
        coco_formats = [
            f"COCO_train2014_{padded_id}",
            f"COCO_val2014_{padded_id}",
            f"COCO_test2015_{padded_id}"
        ]

        for coco_name in coco_formats:
            for ext in extensions:
                image_path = os.path.join(image_folder, coco_name + ext)
                if os.path.exists(image_path):
                    return image_path

    return None

def initialize_qwen_model():
    """初始化Qwen2-VL模型"""
    global model, processor

    if model is None:
        print("🚀 正在加载Qwen2-VL模型...")

        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            min_pixels=224 * 28 * 28,
            max_pixels=1024 * 28 * 28
        )

        # 加载模型
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto"  # 自动分配到多个GPU
        )

        print("✅ Qwen2-VL模型加载完成")
        print(f"📊 模型设备分布: {model.hf_device_map}")





def check_bbox_duplicates(bbox_list, similarity_threshold=0.9):
    """检查并去除重复的bbox"""
    if not bbox_list:
        return bbox_list, []

    print(f"🔍 检查bbox重复性...")
    print(f"   原始bbox数量: {len(bbox_list)}")

    def calculate_bbox_similarity(bbox1, bbox2):
        """计算两个bbox的相似度"""
        # 1. 坐标相似度
        coords1 = bbox1.get('bbox_coordinates', bbox1.get('bbox', []))
        coords2 = bbox2.get('bbox_coordinates', bbox2.get('bbox', []))

        coord_similarity = 0.0
        if coords1 and coords2 and len(coords1) == 4 and len(coords2) == 4:
            # 计算坐标差异
            coord_diff = sum(abs(c1 - c2) for c1, c2 in zip(coords1, coords2))
            coord_similarity = max(0, 1 - coord_diff / 4.0)  # 归一化到0-1

        # 2. 内容相似度
        content1 = bbox1.get('bbox_description', bbox1.get('description', '')).lower()
        content2 = bbox2.get('bbox_description', bbox2.get('description', '')).lower()

        content_similarity = 0.0
        if content1 and content2:
            import difflib
            content_similarity = difflib.SequenceMatcher(None, content1, content2).ratio()

        # 综合相似度 (坐标权重0.6，内容权重0.4)
        overall_similarity = coord_similarity * 0.6 + content_similarity * 0.4
        return overall_similarity, coord_similarity, content_similarity

    unique_bbox_list = []
    duplicate_info = []

    for i, bbox in enumerate(bbox_list):
        is_duplicate = False

        for j, unique_bbox in enumerate(unique_bbox_list):
            similarity, coord_sim, content_sim = calculate_bbox_similarity(bbox, unique_bbox)

            if similarity >= similarity_threshold:
                duplicate_info.append({
                    'original_index': i,
                    'duplicate_of': j,
                    'similarity': similarity,
                    'coord_similarity': coord_sim,
                    'content_similarity': content_sim,
                    'original_content': bbox.get('bbox_description', bbox.get('description', '')),
                    'duplicate_content': unique_bbox.get('bbox_description', unique_bbox.get('description', ''))
                })
                is_duplicate = True
                break

        if not is_duplicate:
            unique_bbox_list.append(bbox)

    # 打印去重结果
    removed_count = len(bbox_list) - len(unique_bbox_list)
    print(f"   去重后bbox数量: {len(unique_bbox_list)}")
    print(f"   移除重复bbox: {removed_count} 个")

    if duplicate_info:
        print(f"   重复bbox详情:")
        for i, dup in enumerate(duplicate_info[:5]):  # 只显示前5个
            print(f"     {i+1}. 索引{dup['original_index']} 与 索引{dup['duplicate_of']} 重复")
            print(f"        相似度: {dup['similarity']:.2f} (坐标:{dup['coord_similarity']:.2f}, 内容:{dup['content_similarity']:.2f})")
            print(f"        内容: '{dup['original_content'][:50]}...'")

        if len(duplicate_info) > 5:
            print(f"     ... 还有 {len(duplicate_info) - 5} 个重复项")

    return unique_bbox_list, duplicate_info

def extract_bbox_content(bbox):
    """从bbox中提取可用的内容描述"""
    # 尝试多个可能的描述字段
    description_fields = [
        'bbox_description',
        'description',
        'content',
        'text',
        'ocr_text',
        'extracted_text'
    ]

    for field in description_fields:
        content = bbox.get(field, '')
        if content and len(content.strip()) >= 3:
            return content.strip()

    # 如果没有找到描述，尝试生成基于坐标的描述
    coords = bbox.get('bbox_coordinates', bbox.get('bbox', []))
    if coords and len(coords) == 4:
        x1, y1, x2, y2 = coords
        # 生成基于位置的描述
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area > 0.5:  # 大区域
            return f"Large region at coordinates ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"
        elif area > 0.1:  # 中等区域
            return f"Medium region at coordinates ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"
        else:  # 小区域
            return f"Small region at coordinates ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"

    return ""

def validate_bbox_data(bbox_list):
    """验证bbox数据的完整性和有效性，尝试修复缺失的描述"""
    print(f"🔍 验证bbox数据...")

    valid_bbox_list = []
    invalid_count = 0
    fixed_count = 0

    for i, bbox in enumerate(bbox_list):
        issues = []
        fixed_bbox = bbox.copy()  # 创建副本以便修改

        # 检查坐标
        coords = bbox.get('bbox_coordinates', bbox.get('bbox', []))
        if not coords:
            issues.append("缺少坐标")
        elif len(coords) != 4:
            issues.append(f"坐标格式错误(长度{len(coords)})")
        elif not all(isinstance(x, (int, float)) for x in coords):
            issues.append("坐标类型错误")

        # 检查并尝试修复描述
        original_description = bbox.get('bbox_description', bbox.get('description', ''))

        if not original_description or len(original_description.strip()) < 3:
            # 尝试提取其他内容
            extracted_content = extract_bbox_content(bbox)

            if extracted_content:
                # 修复描述
                fixed_bbox['bbox_description'] = extracted_content
                fixed_bbox['description'] = extracted_content
                fixed_count += 1
                print(f"   🔧 修复bbox {i}描述: '{extracted_content[:50]}...'")
            else:
                issues.append("描述过短或缺失且无法修复")

        # 检查坐标有效性
        if coords and len(coords) == 4:
            x1, y1, x2, y2 = coords
            if x1 >= x2 or y1 >= y2:
                issues.append("坐标顺序错误")
            if any(coord < 0 for coord in coords):
                issues.append("坐标为负值")

        # 如果只是描述问题且已修复，则认为是有效的
        critical_issues = [issue for issue in issues if "描述" not in issue]

        if critical_issues:
            invalid_count += 1
            if invalid_count <= 3:  # 只显示前3个无效项
                print(f"   ❌ 无效bbox {i}: {', '.join(critical_issues)}")
                print(f"      坐标: {coords}")
                print(f"      描述: '{original_description[:50]}...'")
        else:
            valid_bbox_list.append(fixed_bbox)

    print(f"   有效bbox: {len(valid_bbox_list)} 个")
    print(f"   无效bbox: {invalid_count} 个")
    print(f"   修复描述: {fixed_count} 个")

    return valid_bbox_list

def print_reasoning_chain_examples():
    """打印推理链格式示例"""
    print("\n📋 推理链格式示例:")
    print("=" * 60)

    print("\n🔄 顺序链条 (Sequential Chain):")
    print("   适用于: 步骤性问题、时间顺序问题、流程问题")
    print("   格式: [Description1]: Content1 -> [Description2]: Content2 -> [Description3]: Content3")
    print("   示例: [Title Region]: Company Annual Report -> [Date Section]: Year 2023 -> [Financial Data]: Revenue $100M")

    print("\n🔀 并列链条 (Parallel Chain):")
    print("   适用于: 列举问题、多选项问题、并列关系问题")
    print("   格式: [Root]: Content -> [Branch1]: Content1; [Root]: Content -> [Branch2]: Content2")
    print("   示例: [Question]: What products? -> [Product A]: Laptop; [Question]: What products? -> [Product B]: Phone")

    print("\n🔗 混合链条 (Mixed Chain):")
    print("   适用于: 复杂问题，既有顺序又有并列关系")
    print("   格式: 根据实际关系动态生成")
    print("   示例: [Start]: Process -> [Step1]: Input; [Start]: Process -> [Step2]: Output")

    print("\n💡 关键改进:")
    print("   ✅ 用 description 和 content 替代 bbox 索引")
    print("   ✅ 根据问题类型自动选择链条格式")
    print("   ✅ 支持顺序推理和并列推理")
    print("   ✅ 更直观的推理链表示")
    print("=" * 60)


def load_existing_results(dataset_config):
    """加载已有的结果，并创建ID到结果的映射，识别需要重新生成的失败数据"""
    output_file = dataset_config['output_file']
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results_list = json.load(f)

            # 创建ID到结果的映射，便于快速查找和更新
            results_map = {}
            failed_ids = []
            success_count = 0

            for result in results_list:
                result_id = result.get('id')
                if result_id:
                    results_map[result_id] = result

                    # 检查是否为失败的数据（推理步数为0或错误状态）
                    reasoning_chain = result.get('reasoning_chain', {})
                    reasoning_steps = reasoning_chain.get('reasoning_steps', [])
                    chain_type = reasoning_chain.get('chain_type', '')

                    if (len(reasoning_steps) == 0 or
                        chain_type == 'error' or
                        reasoning_chain.get('error')):
                        failed_ids.append(result_id)
                    else:
                        success_count += 1

            print(f"📂 加载了 {len(results_map)} 个已有结果")
            print(f"   ✅ 成功生成: {success_count} 个")
            print(f"   ❌ 需要重新生成: {len(failed_ids)} 个")

            if failed_ids:
                print(f"   🔄 失败的ID示例: {failed_ids[:5]}...")

            # 将失败的ID列表添加到dataset_config中，供后续使用
            dataset_config['failed_ids'] = set(failed_ids)

            return results_list, results_map
        except Exception as e:
            print(f"⚠️ 加载现有结果失败: {e}")
            return [], {}

    # 如果文件不存在，初始化空的失败ID集合
    dataset_config['failed_ids'] = set()
    return [], {}


def save_results(results, dataset_config):
    """保存关系分析结果"""
    output_file = dataset_config['output_file']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def create_ordered_results_list(bbox_data, results_map, dataset_config):
    """根据输入数据的顺序创建有序的结果列表"""
    ordered_results = []
    for item in bbox_data:
        question_id = item.get(dataset_config['question_id_field'])
        if question_id and question_id in results_map:
            ordered_results.append(results_map[question_id])
    return ordered_results

def generate_reasoning_chains_with_bbox(dataset_config):
    """使用bbox信息生成推理链"""
    print("🚀 开始基于bbox的推理链构建...")

    # 初始化Qwen模型
    print("🔄 正在初始化Qwen模型...")
    initialize_qwen_model()

    # 加载bbox数据
    print("📂 加载bbox数据...")
    bbox_file = dataset_config['bbox_file']

    with open(bbox_file, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)

    # 处理样本数量设置
    max_samples = dataset_config.get('default_max_samples', None)  # 默认处理所有样本
    total_samples = len(bbox_data)

    if max_samples is not None and total_samples > max_samples:
        bbox_data = bbox_data[:max_samples]
        print(f"📊 限制处理前 {max_samples} 个问题（总共 {total_samples} 个）")
    else:
        print(f"📊 处理所有 {total_samples} 个问题")

    # 加载已有结果
    _, existing_results_map = load_existing_results(dataset_config)
    processed_ids = set(existing_results_map.keys())

    # 根据配置决定是否跳过已处理的问题
    if REGENERATION_CONFIG['skip_existing'] and not REGENERATION_CONFIG['regenerate_all']:
        print(f"📋 已处理 {len(processed_ids)} 个问题，跳过重复处理")
    else:
        print(f"📋 已有 {len(processed_ids)} 个结果，根据配置可能会重新生成")
        if REGENERATION_CONFIG['regenerate_all']:
            print("🔄 配置为重新生成所有问题")
            processed_ids = set()  # 清空已处理ID，重新生成所有
        elif REGENERATION_CONFIG['regenerate_multi_bbox'] or REGENERATION_CONFIG['regenerate_single_bbox']:
            print("🔄 配置为重新生成特定类型的问题")

    # 处理每个问题
    for idx, item in enumerate(bbox_data):
        question_id = item.get(dataset_config['question_id_field'])

        print(f"\n{'='*60}")
        print(f"🔄 处理问题 {idx+1}/{len(bbox_data)}: {question_id}")

        # 获取样本信息
        sample_info = get_sample_info(item, dataset_config)
        if not sample_info:
            print(f"❌ 无法获取样本信息，跳过")
            continue

        print(f"❓ 问题: {sample_info['question']}")

        # 获取bbox分析结果
        bbox_analysis = item.get('bbox_analysis', {})
        relevant_elements = bbox_analysis.get('relevant_elements', [])

        if len(relevant_elements) < 1:
            print(f"⏭️ 没有相关bbox，跳过推理链构建")
            continue

        bbox_count = len(relevant_elements)
        print(f"📦 原始相关bbox数量: {bbox_count}")

        # ==================== 生成控制逻辑 ====================
        should_process = True
        skip_reason = ""

        # 1. 检查是否已处理且需要跳过
        is_failed_data = question_id in dataset_config.get('failed_ids', set())

        if question_id in processed_ids and REGENERATION_CONFIG['skip_existing'] and not REGENERATION_CONFIG['regenerate_all']:
            # 如果是失败的数据且配置为重新生成失败数据，强制重新生成
            if is_failed_data and REGENERATION_CONFIG['regenerate_failed']:
                should_process = True
                print(f"🔄 重新生成失败的数据: {question_id}")
            elif not REGENERATION_CONFIG['regenerate_multi_bbox'] and not REGENERATION_CONFIG['regenerate_single_bbox']:
                should_process = False
                skip_reason = "已处理且配置为跳过现有结果"
            elif REGENERATION_CONFIG['regenerate_multi_bbox'] and bbox_count > 1:
                should_process = True
                print(f"🔄 重新生成多bbox问题 (bbox数量: {bbox_count})")
            elif REGENERATION_CONFIG['regenerate_single_bbox'] and bbox_count == 1:
                should_process = True
                print(f"🔄 重新生成单bbox问题 (bbox数量: {bbox_count})")
            else:
                should_process = False
                skip_reason = f"已处理，不符合重新生成条件 (bbox数量: {bbox_count})"

        # 2. 检查bbox数量限制
        if should_process:
            if REGENERATION_CONFIG['min_bbox_count'] and bbox_count < REGENERATION_CONFIG['min_bbox_count']:
                should_process = False
                skip_reason = f"bbox数量 {bbox_count} 小于最小阈值 {REGENERATION_CONFIG['min_bbox_count']}"
            elif REGENERATION_CONFIG['max_bbox_count'] and bbox_count > REGENERATION_CONFIG['max_bbox_count']:
                should_process = False
                skip_reason = f"bbox数量 {bbox_count} 大于最大阈值 {REGENERATION_CONFIG['max_bbox_count']}"
            elif REGENERATION_CONFIG['target_bbox_count'] and bbox_count != REGENERATION_CONFIG['target_bbox_count']:
                should_process = False
                skip_reason = f"bbox数量 {bbox_count} 不等于目标数量 {REGENERATION_CONFIG['target_bbox_count']}"

        # 3. 检查bbox生成模式
        if should_process:
            bbox_mode = REGENERATION_CONFIG.get('bbox_generation_mode', 'auto')
            if bbox_mode == 'single' and bbox_count != 1:
                should_process = False
                skip_reason = f"bbox生成模式为'single'，但当前bbox数量为 {bbox_count}"
            elif bbox_mode == 'multi' and bbox_count <= 1:
                should_process = False
                skip_reason = f"bbox生成模式为'multi'，但当前bbox数量为 {bbox_count}"
            # 'auto'模式处理所有bbox数量

        # 4. 检查特定类型重新生成
        if should_process and not REGENERATION_CONFIG['regenerate_all']:
            if REGENERATION_CONFIG['regenerate_multi_bbox'] and bbox_count <= 1:
                should_process = False
                skip_reason = f"配置为只重新生成多bbox问题，但当前bbox数量为 {bbox_count}"
            elif REGENERATION_CONFIG['regenerate_single_bbox'] and bbox_count != 1:
                should_process = False
                skip_reason = f"配置为只重新生成单bbox问题，但当前bbox数量为 {bbox_count}"

        # 决定是否跳过
        if not should_process:
            if REGENERATION_CONFIG['verbose_logging']:
                print(f"⏭️ 跳过: {skip_reason}")
            else:
                print(f"⏭️ 跳过问题 {question_id}")
            continue

        # 如果是重新生成，从现有结果映射中移除
        if question_id in processed_ids:
            if question_id in existing_results_map:
                del existing_results_map[question_id]
                print(f"🗑️ 移除现有结果，准备重新生成")
        # ==================== 生成控制逻辑结束 ====================

        # 🔍 数据检查和清理
        print(f"🔧 开始数据检查和清理...")

        # 1. 验证bbox数据有效性
        valid_elements = validate_bbox_data(relevant_elements)

        if len(valid_elements) < 1:
            print(f"⏭️ 没有有效bbox，跳过推理链构建")
            continue

        # 2. 去除重复bbox
        unique_elements, _ = check_bbox_duplicates(valid_elements, similarity_threshold=0.85)

        if len(unique_elements) < 1:
            print(f"⏭️ 去重后没有有效bbox，跳过推理链构建")
            continue

        print(f"✅ 最终可用bbox数量: {len(unique_elements)}")

        # 特殊提示：单bbox情况
        if len(unique_elements) == 1:
            print(f"📝 单bbox推理模式 - 将进行直接分析")

        # 更新相关元素为清理后的数据
        relevant_elements = unique_elements

        # 显示数据清理统计
        original_count = len(bbox_analysis.get('relevant_elements', []))
        final_count = len(relevant_elements)
        removed_count = original_count - final_count

        if removed_count > 0:
            print(f"📊 数据清理统计: 原始{original_count}个 → 最终{final_count}个 (移除{removed_count}个)")

        # 🔥 核心：使用Qwen构建推理链
        try:
            reasoning_chain = build_reasoning_chain_with_multi_qwen(sample_info['image_path'], sample_info['question'], relevant_elements)
            print(f"✅ 推理链构建完成")
            print(f"🔗 链类型: {reasoning_chain['chain_type']}")
            print(f"📊 推理步数: {reasoning_chain['total_steps']}")
            print(f"❓ 问题类型: {reasoning_chain.get('question_type', 'unknown')}")
            print(f"🔗 推理链: {reasoning_chain.get('chain_text', '')}")

            # 显示推理链
            for step in reasoning_chain['reasoning_steps']:
                generated_reasoning = step.get('generated_reasoning', '')
                original_description = step.get('description', step.get('bbox_content', ''))

                if generated_reasoning:
                    display_text = generated_reasoning[:60] + "..." if len(generated_reasoning) > 60 else generated_reasoning
                    print(f"   步骤{step['step']}: {display_text} (角色: {step['role']})")
                else:
                    print(f"   步骤{step['step']}: [{original_description[:30]}...] (角色: {step['role']})")

            if reasoning_chain.get('parallel_bbox'):
                print(f"🔀 并列关系: {len(reasoning_chain['parallel_bbox'])} 个")

        except Exception as e:
            print(f"❌ 推理链构建失败: {e}")
            reasoning_chain = {
                "chain_type": "error",
                "error": str(e),
                "reasoning_steps": []
            }

        # 保存结果 - 精简版本，去除重复字段
        result = {
            "id": question_id,
            "image": [sample_info['image_name']],
            "question": sample_info['question'],
            "reasoning_chain": reasoning_chain,
            "bbox_elements": relevant_elements,
            "ground_truth_answers": item.get('answers', []),
            # 统计信息（可选，用于分析）
            "stats": {
                "bbox_count": len(relevant_elements),
                "original_bbox_count": len(bbox_analysis.get('relevant_elements', [])),
                "removed_bbox_count": len(bbox_analysis.get('relevant_elements', [])) - len(relevant_elements),
                "data_cleaning_applied": True
            }
        }

        # 将新结果添加到映射中
        existing_results_map[question_id] = result

        # 每5个问题保存一次
        if len(existing_results_map) % 5 == 0:
            # 创建按输入顺序排列的结果列表
            ordered_results = create_ordered_results_list(bbox_data, existing_results_map, dataset_config)
            save_results(ordered_results, dataset_config)
            print(f"💾 已保存 {len(ordered_results)} 个结果")

    # 最终保存 - 确保结果按输入数据顺序排列
    final_ordered_results = create_ordered_results_list(bbox_data, existing_results_map, dataset_config)
    save_results(final_ordered_results, dataset_config)

    # 统计数据清理效果
    total_original_bbox = sum(r.get('original_bbox_count', 0) for r in final_ordered_results)
    total_final_bbox = sum(r.get('bbox_count', 0) for r in final_ordered_results)
    total_removed_bbox = sum(r.get('removed_bbox_count', 0) for r in final_ordered_results)

    print(f"\n🎉 处理完成！总共生成了 {len(final_ordered_results)} 个推理链")
    print(f"📁 结果保存在: {dataset_config['output_file']}")
    print(f"\n📊 数据清理总体统计:")
    print(f"   原始bbox总数: {total_original_bbox}")
    print(f"   最终bbox总数: {total_final_bbox}")
    print(f"   移除bbox总数: {total_removed_bbox}")
    if total_original_bbox > 0:
        removal_rate = (total_removed_bbox / total_original_bbox) * 100
        print(f"   移除率: {removal_rate:.1f}%")

    return final_ordered_results

if __name__ == "__main__":
    # 验证CUDA设置
    print("\n🔧 验证GPU环境:")
    if torch.cuda.is_available():
        print(f"✅ CUDA可用")
        print(f"🔧 当前可见GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"🔧 GPU {i}: {torch.cuda.get_device_name(i)}")
            # 显示GPU内存信息
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3
            print(f"      总内存: {total_memory:.1f}GB, 已分配: {allocated_memory:.1f}GB, 缓存: {cached_memory:.1f}GB")
    else:
        print("❌ CUDA不可用")
        exit(1)

    # 验证环境变量设置
    print("\n🔧 环境变量设置:")
    for key in ['CUDA_LAUNCH_BLOCKING', 'TORCH_USE_CUDA_DSA', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_VISIBLE_DEVICES', 'TOKENIZERS_PARALLELISM']:
        print(f"  {key}: {os.environ.get(key, '未设置')}")

    # 检查可用的数据集文件
    print("\n📂 检查数据集文件...")
    available_datasets = []
    for name, config in DATASETS.items():
        if os.path.exists(config['bbox_file']):
            available_datasets.append(name)

    if not available_datasets:
        print("❌ 没有找到可用的bbox数据文件")
        print("请确保以下文件存在:")
        for name, config in DATASETS.items():
            print(f"  - {config['bbox_file']}")
        exit(1)

    print("📋 可用的数据集:")
    for i, name in enumerate(available_datasets):
        config = DATASETS[name]
        print(f"  {i+1}. {config['name']} ({name}) - {config['bbox_file']}")

    # 让用户选择数据集
    try:
        choice = input(f"\n请选择数据集 (1-{len(available_datasets)}): ").strip()
        choice_idx = int(choice) - 1

        if 0 <= choice_idx < len(available_datasets):
            dataset_name = available_datasets[choice_idx]
        else:
            print("❌ 无效的选择，使用第一个数据集")
            dataset_name = available_datasets[0]
    except (ValueError, KeyboardInterrupt):
        print("❌ 无效输入，使用第一个数据集")
        dataset_name = available_datasets[0]

    # 选择bbox生成模式
    bbox_mode = select_bbox_generation_mode()
    REGENERATION_CONFIG['bbox_generation_mode'] = bbox_mode

    # 配置数据集
    dataset_config = DATASETS[dataset_name].copy()
    # 修改输出文件名为推理链
    dataset_config['output_file'] = dataset_config['output_file'].replace('relations', 'reasoning_chains')
    # dataset_config['default_max_samples'] = None  # 处理所有样本

    print(f"\n🚀 开始为 {dataset_config['name']} 数据集构建推理链...")
    print(f"📂 输入文件: {dataset_config['bbox_file']}")
    print(f"📁 输出文件: {dataset_config['output_file']}")

    # 显示当前生成配置
    print(f"\n⚙️ 当前生成配置:")
    print(f"   🎯 bbox生成模式: {REGENERATION_CONFIG['bbox_generation_mode']}")
    print(f"   🔄 重新生成所有问题: {REGENERATION_CONFIG['regenerate_all']}")
    print(f"   📦 重新生成多bbox问题: {REGENERATION_CONFIG['regenerate_multi_bbox']}")
    print(f"   📝 重新生成单bbox问题: {REGENERATION_CONFIG['regenerate_single_bbox']}")
    print(f"   🔧 重新生成失败数据: {REGENERATION_CONFIG['regenerate_failed']}")
    print(f"   🎯 目标bbox数量: {REGENERATION_CONFIG['target_bbox_count'] or '所有'}")
    print(f"   📊 bbox数量范围: {REGENERATION_CONFIG['min_bbox_count']} - {REGENERATION_CONFIG['max_bbox_count'] or '无限制'}")
    print(f"   ⏭️ 跳过已有结果: {REGENERATION_CONFIG['skip_existing']}")

    # 询问用户是否要修改配置
    try:
        modify_config = input(f"\n是否要修改生成配置? (y/N): ").strip().lower()
        if modify_config in ['y', 'yes']:
            print(f"\n📝 配置选项:")
            print(f"  1. 修改bbox生成模式")
            print(f"  2. 重新生成所有问题")
            print(f"  3. 重新生成多bbox问题 (bbox_count > 1)")
            print(f"  4. 重新生成单bbox问题 (bbox_count == 1)")
            print(f"  5. 只处理特定bbox数量的问题")
            print(f"  6. 设置bbox数量范围")
            print(f"  7. 使用当前配置")

            choice = input(f"请选择 (1-7): ").strip()

            if choice == '1':
                # 修改bbox生成模式
                new_mode = select_bbox_generation_mode()
                REGENERATION_CONFIG['bbox_generation_mode'] = new_mode
                print(f"✅ 已更新bbox生成模式为: {new_mode}")
            elif choice == '2':
                REGENERATION_CONFIG['regenerate_all'] = True
                REGENERATION_CONFIG['regenerate_multi_bbox'] = False
                REGENERATION_CONFIG['regenerate_single_bbox'] = False
                REGENERATION_CONFIG['skip_existing'] = False
                print(f"✅ 配置为重新生成所有问题")
            elif choice == '3':
                REGENERATION_CONFIG['regenerate_multi_bbox'] = True
                REGENERATION_CONFIG['regenerate_single_bbox'] = False
                REGENERATION_CONFIG['regenerate_all'] = False
                print(f"✅ 配置为重新生成多bbox问题")
            elif choice == '4':
                REGENERATION_CONFIG['regenerate_single_bbox'] = True
                REGENERATION_CONFIG['regenerate_multi_bbox'] = False
                REGENERATION_CONFIG['regenerate_all'] = False
                print(f"✅ 配置为重新生成单bbox问题")
            elif choice == '5':
                target_count = input(f"请输入目标bbox数量: ").strip()
                try:
                    REGENERATION_CONFIG['target_bbox_count'] = int(target_count)
                    print(f"✅ 配置为只处理bbox数量为 {target_count} 的问题")
                except ValueError:
                    print(f"❌ 无效输入，使用默认配置")
            elif choice == '6':
                min_count = input(f"请输入最小bbox数量 (当前: {REGENERATION_CONFIG['min_bbox_count']}): ").strip()
                max_count = input(f"请输入最大bbox数量 (当前: {REGENERATION_CONFIG['max_bbox_count'] or '无限制'}): ").strip()
                try:
                    if min_count:
                        REGENERATION_CONFIG['min_bbox_count'] = int(min_count)
                    if max_count:
                        REGENERATION_CONFIG['max_bbox_count'] = int(max_count)
                    print(f"✅ 配置bbox数量范围: {REGENERATION_CONFIG['min_bbox_count']} - {REGENERATION_CONFIG['max_bbox_count'] or '无限制'}")
                except ValueError:
                    print(f"❌ 无效输入，使用默认配置")
            else:
                print(f"✅ 使用当前配置")
    except KeyboardInterrupt:
        print(f"\n✅ 使用当前配置")

    # 显示最终配置
    print(f"\n🎯 最终生成配置:")
    print(f"   🎯 bbox生成模式: {REGENERATION_CONFIG['bbox_generation_mode']}")
    print(f"   🔄 重新生成所有问题: {REGENERATION_CONFIG['regenerate_all']}")
    print(f"   📦 重新生成多bbox问题: {REGENERATION_CONFIG['regenerate_multi_bbox']}")
    print(f"   📝 重新生成单bbox问题: {REGENERATION_CONFIG['regenerate_single_bbox']}")
    print(f"   🔧 重新生成失败数据: {REGENERATION_CONFIG['regenerate_failed']}")
    print(f"   🎯 目标bbox数量: {REGENERATION_CONFIG['target_bbox_count'] or '所有'}")
    print(f"   📊 bbox数量范围: {REGENERATION_CONFIG['min_bbox_count']} - {REGENERATION_CONFIG['max_bbox_count'] or '无限制'}")

    # 显示新功能说明
    print_reasoning_chain_examples()

    # 运行推理链构建
    results = generate_reasoning_chains_with_bbox(dataset_config)

    print(f"\n🎉 推理链构建完成！")
    print(f"📊 总共处理了 {len(results)} 个问题")


