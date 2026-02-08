# -*-coding:utf-8-*-
"""
Ragas 评估模块

该模块使用 Ragas 框架对 RAG 系统进行评估，衡量检索和生成质量。
评估指标包括：
    - faithfulness (忠实度): 回答是否忠于检索到的上下文
    - answer_relevancy (答案相关性): 回答与问题的相关程度
    - context_precision (上下文精确率): 检索到的上下文中有多少是真正有用的
    - context_recall (上下文召回率): 需要的信息被检索到了多少

使用方法：
    python rag_qa/rag_assesment/rag_as.py
"""

# 导入 pandas 库，用于数据处理和保存 CSV 文件
import pandas as pd
# 导入 json 模块，用于加载评估数据
import json
# 导入 os 模块，用于路径操作
import os
import sys

# 设置路径，确保能导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)

# 导入 ragas 库的 evaluate 函数，用于执行 RAG 评估
from ragas import evaluate
# 导入 ragas 的评估指标
from ragas.metrics import (
    faithfulness,       # 忠实度：回答是否基于给定上下文
    answer_relevancy,   # 答案相关性：回答与问题的相关程度
    context_precision,  # 上下文精确率：检索到的上下文是否精准
    context_recall      # 上下文召回率：是否检索到了足够的相关信息
)
# 导入 datasets 库的 Dataset 类，用于构建 RAGAS 所需的数据格式
from datasets import Dataset
# 导入 langchain_community 的 Ollama 聊天模型和嵌入模型，用于本地模型调用
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# 导入项目配置
from base import logger, Config


def load_evaluate_data(json_path: str) -> list:
    """
    加载评估数据集
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        包含评估数据的列表
    """
    logger.info(f"正在加载评估数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"成功加载 {len(data)} 条评估数据")
    return data


def convert_to_ragas_format(data: list) -> Dataset:
    """
    将原始数据转换为 Ragas 要求的 Dataset 格式
    
    Ragas 要求的字段：
        - question: 用户的问题
        - contexts: 检索到的上下文列表 (注意是复数形式)
        - answer: 模型生成的回答
        - ground_truth: 标准答案/参考答案
    
    Args:
        data: 原始数据列表
        
    Returns:
        Ragas Dataset 对象
    """
    logger.info("正在转换数据格式为 Ragas Dataset...")
    
    # 初始化各字段列表
    questions = []
    contexts = []
    answers = []
    ground_truths = []
    
    # 遍历每条数据，提取字段
    for item in data:
        questions.append(item["question"])
        # Ragas 要求 contexts 是列表，我们的数据中 context 本身就是列表
        contexts.append(item["context"])
        answers.append(item["answer"])
        ground_truths.append(item["ground_truth"])
    
    # 构建 Ragas 兼容的字典格式
    ragas_data = {
        "question": questions,
        "contexts": contexts,  # Ragas 要求复数形式
        "answer": answers,
        "ground_truth": ground_truths
    }
    
    # 转换为 Hugging Face datasets.Dataset 对象
    dataset = Dataset.from_dict(ragas_data)
    logger.info(f"数据格式转换完成，共 {len(dataset)} 条记录")
    
    return dataset


def run_evaluation(dataset: Dataset, llm, embeddings) -> dict:
    """
    执行 Ragas 评估
    
    Args:
        dataset: Ragas Dataset 对象
        llm: 用于评估的语言模型
        embeddings: 用于评估的嵌入模型
        
    Returns:
        评估结果字典
    """
    logger.info("开始执行 Ragas 评估...")
    logger.info("评估指标: faithfulness, answer_relevancy, context_precision, context_recall")
    
    # 定义评估指标列表
    metrics = [
        faithfulness,       # 忠实度
        answer_relevancy,   # 答案相关性
        context_precision,  # 上下文精确率
        context_recall      # 上下文召回率
    ]
    
    # 调用 Ragas evaluate 函数执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    logger.info("Ragas 评估完成!")
    return result


def save_results(result, output_path: str):
    """
    保存评估结果到 CSV 文件
    
    Args:
        result: Ragas 评估结果
        output_path: 输出文件路径
    """
    logger.info(f"正在保存评估结果到: {output_path}")
    
    # 将结果转换为 DataFrame
    df = result.to_pandas()
    
    # 保存为 CSV 文件
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"评估结果已保存到: {output_path}")


def print_summary(result):
    """
    打印评估结果摘要
    
    Args:
        result: Ragas 评估结果
    """
    print("\n" + "=" * 60)
    print("📊 Ragas 评估结果摘要")
    print("=" * 60)
    
    # 打印各指标得分
    for metric_name, score in result.items():
        if isinstance(score, float):
            # 根据分数给出评级提示
            if score >= 0.8:
                emoji = "🟢"  # 优秀
                level = "优秀"
            elif score >= 0.6:
                emoji = "🟡"  # 良好
                level = "良好"
            elif score >= 0.4:
                emoji = "🟠"  # 一般
                level = "一般"
            else:
                emoji = "🔴"  # 需改进
                level = "需改进"
            
            print(f"{emoji} {metric_name}: {score:.4f} ({level})")
    
    print("=" * 60)
    print("\n📝 指标说明:")
    print("  - faithfulness: 回答是否忠于检索到的上下文 (越高越好)")
    print("  - answer_relevancy: 回答与问题的相关程度 (越高越好)")
    print("  - context_precision: 检索到的上下文有多精准 (越高越好)")
    print("  - context_recall: 需要的信息被检索到了多少 (越高越好)")
    print()


def main():
    """
    主函数：执行完整的 Ragas 评估流程
    """
    print("\n🚀 启动 Ragas RAG 评估系统...\n")
    
    # ========== 1. 加载评估数据 ==========
    json_path = os.path.join(current_dir, "rag_evaluate_data.json")
    data = load_evaluate_data(json_path)
    
    # ========== 2. 转换为 Ragas 格式 ==========
    dataset = convert_to_ragas_format(data)
    
    # ========== 3. 配置评估模型 ==========
    # 使用本地 Ollama 模型进行评估
    # 注意：确保 Ollama 服务已启动，且已下载 qwen2.5:7b 模型
    logger.info("正在初始化评估模型 (Ollama qwen2.5:7b)...")
    
    llm = ChatOllama(
        model="qwen2.5:7b",
        base_url='http://localhost:11434'
    )
    
    embeddings = OllamaEmbeddings(
        model="qwen2.5:7b",
        base_url='http://localhost:11434'
    )
    
    logger.info("评估模型初始化完成")
    
    # ========== 4. 执行评估 ==========
    result = run_evaluation(dataset, llm, embeddings)
    
    # ========== 5. 打印结果摘要 ==========
    print_summary(result)
    
    # ========== 6. 保存详细结果 ==========
    output_csv_path = os.path.join(current_dir, "ragas_evaluation_result.csv")
    save_results(result, output_csv_path)
    
    print(f"✅ 评估完成！详细结果已保存至: {output_csv_path}")


if __name__ == "__main__":
    main()
