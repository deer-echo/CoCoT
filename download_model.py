#!/usr/bin/env python3
"""
Qwen2-VL模型下载脚本
===================

自动下载Qwen2-VL-7B-Instruct模型文件
支持从HuggingFace和ModelScope下载

使用方法:
    python download_model.py [--source huggingface|modelscope] [--model-dir path]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def print_section(title):
    """打印章节标题"""
    print(f"\n{'='*60}")
    print(f"📥 {title}")
    print('='*60)

def check_git_lfs():
    """检查git-lfs是否安装"""
    try:
        result = subprocess.run(['git', 'lfs', 'version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Git LFS已安装: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git LFS未安装")
        print("请先安装Git LFS:")
        print("  Ubuntu/Debian: sudo apt install git-lfs")
        print("  CentOS/RHEL: sudo yum install git-lfs")
        print("  macOS: brew install git-lfs")
        print("  Windows: 从 https://git-lfs.github.io/ 下载安装")
        return False

def download_from_huggingface(model_dir):
    """从HuggingFace下载模型"""
    print_section("从HuggingFace下载Qwen2-VL-7B-Instruct")
    
    if not check_git_lfs():
        return False
    
    # 初始化git-lfs
    try:
        subprocess.run(['git', 'lfs', 'install'], check=True)
        print("✅ Git LFS初始化成功")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Git LFS初始化失败: {e}")
    
    # 克隆模型仓库
    model_url = "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct"
    
    try:
        print(f"📥 开始下载模型到: {model_dir}")
        print("⏳ 这可能需要较长时间，请耐心等待...")
        
        subprocess.run([
            'git', 'clone', model_url, str(model_dir)
        ], check=True)
        
        print("✅ 模型下载完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_from_modelscope(model_dir):
    """从ModelScope下载模型"""
    print_section("从ModelScope下载Qwen2-VL-7B-Instruct")
    
    try:
        # 检查是否安装了modelscope
        import modelscope
        print(f"✅ ModelScope已安装: {modelscope.__version__}")
    except ImportError:
        print("❌ ModelScope未安装")
        print("正在安装ModelScope...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'modelscope'], check=True)
            import modelscope
            print("✅ ModelScope安装成功")
        except Exception as e:
            print(f"❌ ModelScope安装失败: {e}")
            return False
    
    try:
        from modelscope import snapshot_download
        
        print(f"📥 开始下载模型到: {model_dir}")
        print("⏳ 这可能需要较长时间，请耐心等待...")
        
        # 下载模型
        snapshot_download(
            model_id='qwen/Qwen2-VL-7B-Instruct',
            cache_dir=str(model_dir.parent),
            local_dir=str(model_dir)
        )
        
        print("✅ 模型下载完成")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def verify_model_files(model_dir):
    """验证模型文件完整性"""
    print_section("验证模型文件")
    
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors.index.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name}")
            missing_files.append(file_name)
    
    # 检查模型权重文件
    model_files = list(model_dir.glob("model-*.safetensors"))
    if model_files:
        total_size = sum(f.stat().st_size for f in model_files) / 1024**3
        print(f"✅ 模型权重文件: {len(model_files)} 个 ({total_size:.1f}GB)")
    else:
        print("❌ 未找到模型权重文件")
        missing_files.append("model-*.safetensors")
    
    if missing_files:
        print(f"\n⚠️ 缺少文件: {', '.join(missing_files)}")
        print("模型可能下载不完整，请重新下载")
        return False
    else:
        print("\n✅ 所有必需文件都存在，模型验证通过")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载Qwen2-VL-7B-Instruct模型")
    parser.add_argument(
        "--source", 
        choices=["huggingface", "modelscope"], 
        default="huggingface",
        help="下载源 (默认: huggingface)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("Qwen2-VL-7B-Instruct"),
        help="模型保存目录 (默认: ./Qwen2-VL-7B-Instruct)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载，即使目录已存在"
    )
    
    args = parser.parse_args()
    
    print("🚀 Qwen2-VL模型下载工具")
    print("="*60)
    print(f"下载源: {args.source}")
    print(f"保存目录: {args.model_dir}")
    
    # 检查目录是否已存在
    if args.model_dir.exists() and not args.force:
        print(f"\n⚠️ 目录 {args.model_dir} 已存在")
        
        # 验证现有文件
        if verify_model_files(args.model_dir):
            print("✅ 模型文件已存在且完整，无需重新下载")
            return True
        
        # 询问是否重新下载
        while True:
            choice = input("是否要重新下载? (y/N): ").strip().lower()
            if choice in ['y', 'yes']:
                print("🗑️ 删除现有目录...")
                import shutil
                shutil.rmtree(args.model_dir)
                break
            elif choice in ['n', 'no', '']:
                print("👋 取消下载")
                return False
            else:
                print("请输入 y 或 n")
    
    # 创建目录
    args.model_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # 根据选择的源下载模型
    success = False
    if args.source == "huggingface":
        success = download_from_huggingface(args.model_dir)
    elif args.source == "modelscope":
        success = download_from_modelscope(args.model_dir)
    
    if success:
        # 验证下载的文件
        verify_model_files(args.model_dir)
        
        print_section("下载完成")
        print("🎉 模型下载成功！")
        print(f"📁 模型位置: {args.model_dir.absolute()}")
        print("\n下一步:")
        print("1. 运行环境检查: python check_environment.py")
        print("2. 开始生成数据: python quick_start.py")
    else:
        print_section("下载失败")
        print("❌ 模型下载失败")
        print("请检查网络连接或尝试其他下载源")
        
        if args.source == "huggingface":
            print("💡 建议尝试: python download_model.py --source modelscope")
        else:
            print("💡 建议尝试: python download_model.py --source huggingface")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
