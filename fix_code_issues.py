# -*- coding: utf-8 -*-
"""
自动修复脚本
修复项目中的硬编码路径和LangChain版本兼容性问题
"""

import os
import re
import shutil
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 备份目录
BACKUP_DIR = os.path.join(PROJECT_ROOT, 'backup_before_fix')

# 需要修复的文件列表
FILES_TO_FIX = [
    {
        'path': 'rag_qa/edu_text_spliter/edu_model_text_spliter.py',
        'fixes': [
            {
                'type': 'hardcoded_path',
                'line_number': 24,
                'old': r"model=r'D:\workspace\workspace_python\python_1022\dev07_rag\integrated_qa_system\rag_qa\models\nlp_bert_document-segmentation_chinese-base',",
                'new_lines': [
                    "        # 动态获取模型路径",
                    "        current_dir = os.path.dirname(os.path.abspath(__file__))",
                    "        rag_qa_path = os.path.dirname(current_dir)",
                    "        model_path = os.path.join(rag_qa_path, 'models', 'nlp_bert_document-segmentation_chinese-base')",
                    "        p = pipeline(",
                    "            task=\"document-segmentation\",",
                    "            model=model_path,",
                    "            device=\"cpu\")"
                ]
            },
            {
                'type': 'import',
                'old': 'from langchain.text_splitter import CharacterTextSplitter',
                'new': 'from langchain_text_splitters import CharacterTextSplitter'
            }
        ]
    },
    {
        'path': 'rag_qa/core/vector_store.py',
        'fixes': [
            {
                'type': 'import',
                'old': 'from langchain.docstore.document import Document',
                'new': 'from langchain_core.documents import Document'
            }
        ]
    },
    {
        'path': 'rag_qa/edu_text_spliter/edu_chinese_recursive_text_splitter.py',
        'fixes': [
            {
                'type': 'import',
                'old': 'from langchain.text_splitter import RecursiveCharacterTextSplitter',
                'new': 'from langchain_text_splitters import RecursiveCharacterTextSplitter'
            }
        ]
    },
    {
        'path': 'rag_qa/edu_document_loaders/edu_pdfloader.py',
        'fixes': [
            {
                'type': 'import',
                'old': 'from langchain.text_splitter import CharacterTextSplitter',
                'new': 'from langchain_text_splitters import CharacterTextSplitter'
            }
        ]
    },
    {
        'path': 'rag_qa/core/document_processor.py',
        'fixes': [
            {
                'type': 'import',
                'old': 'from langchain.text_splitter import MarkdownTextSplitter',
                'new': 'from langchain_text_splitters import MarkdownTextSplitter'
            }
        ]
    },
    {
        'path': 'rag_qa/core/strategy_selector.py',
        'fixes': [
            {
                'type': 'import',
                'old': 'from langchain.prompts import PromptTemplate',
                'new': 'from langchain_core.prompts import PromptTemplate'
            }
        ]
    },
    {
        'path': 'rag_qa/core/prompts.py',
        'fixes': [
            {
                'type': 'import',
                'old': 'from langchain.prompts import PromptTemplate',
                'new': 'from langchain_core.prompts import PromptTemplate'
            }
        ]
    }
]


def create_backup():
    """创建备份"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_subdir = os.path.join(BACKUP_DIR, f'backup_{timestamp}')
    os.makedirs(backup_subdir)
    
    print(f"📦 创建备份目录: {backup_subdir}")
    
    for file_info in FILES_TO_FIX:
        file_path = os.path.join(PROJECT_ROOT, file_info['path'])
        if os.path.exists(file_path):
            # 保持目录结构
            rel_dir = os.path.dirname(file_info['path'])
            backup_file_dir = os.path.join(backup_subdir, rel_dir)
            os.makedirs(backup_file_dir, exist_ok=True)
            
            backup_file_path = os.path.join(backup_subdir, file_info['path'])
            shutil.copy2(file_path, backup_file_path)
            print(f"  ✅ 已备份: {file_info['path']}")
    
    return backup_subdir


def fix_import_statement(file_path, old_import, new_import):
    """修复导入语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if old_import in content:
        content = content.replace(old_import, new_import)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def fix_hardcoded_path(file_path, old_line, new_lines):
    """修复硬编码路径"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找包含旧代码的行
    for i, line in enumerate(lines):
        if old_line.strip() in line:
            # 找到起始位置(pipeline调用开始)
            # 需要替换从 p = pipeline 到 device="cpu") 的整个块
            start_idx = i
            # 向上查找 p = pipeline 的位置
            while start_idx > 0 and 'p = pipeline' not in lines[start_idx]:
                start_idx -= 1
            
            # 向下查找结束位置
            end_idx = i
            while end_idx < len(lines) and 'device="cpu")' not in lines[end_idx]:
                end_idx += 1
            
            # 获取缩进
            indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            
            # 替换代码块
            new_block = [' ' * indent + line + '\n' for line in new_lines]
            lines = lines[:start_idx] + new_block + lines[end_idx+1:]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
    
    return False


def apply_fixes():
    """应用所有修复"""
    print("\n🔧 开始修复文件...\n")
    
    fixed_count = 0
    failed_count = 0
    
    for file_info in FILES_TO_FIX:
        file_path = os.path.join(PROJECT_ROOT, file_info['path'])
        
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_info['path']}")
            failed_count += 1
            continue
        
        print(f"📝 处理文件: {file_info['path']}")
        
        for fix in file_info['fixes']:
            try:
                if fix['type'] == 'import':
                    if fix_import_statement(file_path, fix['old'], fix['new']):
                        print(f"  ✅ 已修复导入: {fix['old'][:50]}...")
                        fixed_count += 1
                    else:
                        print(f"  ⚠️  未找到: {fix['old'][:50]}...")
                
                elif fix['type'] == 'hardcoded_path':
                    if fix_hardcoded_path(file_path, fix['old'], fix['new_lines']):
                        print(f"  ✅ 已修复硬编码路径")
                        fixed_count += 1
                    else:
                        print(f"  ⚠️  未找到硬编码路径")
            
            except Exception as e:
                print(f"  ❌ 修复失败: {e}")
                failed_count += 1
    
    return fixed_count, failed_count


def verify_fixes():
    """验证修复结果"""
    print("\n🔍 验证修复结果...\n")
    
    # 检查是否还有旧的导入
    old_patterns = [
        'from langchain.docstore.document import',
        'from langchain.text_splitter import',
        'from langchain.prompts import',
        r"r'D:\\workspace\\workspace_python"
    ]
    
    issues_found = []
    
    for file_info in FILES_TO_FIX:
        file_path = os.path.join(PROJECT_ROOT, file_info['path'])
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in old_patterns:
                if pattern in content:
                    issues_found.append(f"{file_info['path']}: 仍包含 '{pattern}'")
    
    if issues_found:
        print("⚠️  发现以下问题:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ 所有修复已成功应用!")
        return True


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  自动修复工具 - 硬编码路径和LangChain兼容性".center(60))
    print("="*60 + "\n")
    
    # 创建备份
    backup_dir = create_backup()
    print(f"\n✅ 备份完成: {backup_dir}\n")
    
    # 应用修复
    fixed, failed = apply_fixes()
    
    # 验证
    success = verify_fixes()
    
    # 总结
    print("\n" + "="*60)
    print("  修复总结".center(60))
    print("="*60)
    print(f"✅ 成功修复: {fixed} 项")
    if failed > 0:
        print(f"❌ 失败: {failed} 项")
    print(f"📦 备份位置: {backup_dir}")
    
    if success:
        print("\n🎉 所有问题已修复!")
        print("\n下一步:")
        print("1. 运行 'python check_config.py' 验证配置")
        print("2. 安装依赖: pip install langchain-core langchain-text-splitters")
        print("3. 启动应用: python app.py")
    else:
        print("\n⚠️  部分问题未完全修复,请手动检查")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
