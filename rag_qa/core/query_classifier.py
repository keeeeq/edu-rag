# -*- coding:utf-8-*-
# 导入标准库
import json
import os
import torch
import sys

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
# 导入日志
from base import logger
# 导入numpy
import numpy as np
# 导入 Transformers 库
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
# 导入train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


class QueryClassifier:
    def __init__(self, model_path="bert_query_classifier"):
        # 初始化模型路径
        self.model_path = model_path
        # 加载 BERT 分词器
        bert_path = os.path.join(rag_qa_path, 'models', 'bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained("../models/bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 初始化模型
        self.model = None
        # 确定设备（GPU 或 CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 记录设备信息
        # logger.info(f"使用设备: {self.device}")
        # 定义标签映射
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        # 加载模型
        self.load_model()

    # 目的:是加载原始的预训练模型(通义:中文文本分类)

    def load_model(self):
        # 检查模型路径是否存在
        if os.path.exists(self.model_path):  # 加载微调之后的模型
            # 加载预训练模型
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录加载成功的日志
            logger.info(f"加载模型: {self.model_path}")
        else: #  初始化新模型
            # 初始化新模型,num_labels=2 表示2分类({"通用知识": 0, "专业咨询": 1})
            self.model = BertForSequenceClassification.from_pretrained("../models/bert-base-chinese", num_labels=2)
            # print(f'self.model--》{self.model}')
            # 将模型移到指定设备
            self.model.to(self.device)
            # 记录初始化模型的日志
            logger.info("初始化新 BERT 模型")

    def save_model(self):
        pass
        """保存模型:save_pretrained"""
        self.model.save_pretrained(self.model_path)
        """保存分词器"""
        self.tokenizer.save_pretrained(self.model_path)
        # 日志打印
        logger.info('任务跑完,模型保存完毕.....')
    def train_model(self, data_file="model_generic_2000.json"):
        pass
        """训练 BERT 分类模型"""
        '''
            大的步骤:
                1.数据预处理 : 分解成训练集和测试集
                2.模型训练 : Trainer(模型训练对象,封装参数和数据), TrainingArguments(设置超参数)
                3.模型保存 : 本地话存储,存储的是微调之后的模型
        '''

        # 加载数据集
        # 1.判断文件是否存在
        if not os.path.exists(data_file):
            # FileNotFoundError 内置异常对象
            raise FileNotFoundError(f'没有文件:{data_file}..')  # raise:关键字,表示向上抛异常

        # 2.读取文件,并转成json
        with open(data_file,'r',encoding='utf-8') as rf:
            # 字符串转json
            data = [json.loads(i) for i in rf.readlines()]
        print('====='*20)
        # 3.获取问题和标签
        # 3.1获取所有的问题(特征)
        questions = [i['query'] for i in data]
        print(questions)
        # 3.2获取所有的标签
        labels = [i['label'] for i in data]
        print(labels)
        # 4.数据划分
        x_train,x_test,y_train,y_test = train_test_split(questions,labels,test_size=0.2,random_state=20)

        # 5. 数据预处理:训练和测试,分词 + 获取标签类别[0,1]
        encodings_train, labels_train = self.preprocess_data(x_train,y_train) # 训练集
        encodings_test, labels_test = self.preprocess_data(x_test,y_test) # 训练集

        # 6.获取dataset对象: create_dataset
        dataset_train = self.create_dataset(encodings_train, labels_train)
        dataset_test = self.create_dataset(encodings_test, labels_test)

        # 7.设置训练参数
        training_args = TrainingArguments(
            output_dir="./bert_results",  # 模型（检查点）以及日志保存的路径等，
            num_train_epochs=3,  # 训练的轮次
            per_device_train_batch_size=8,  # 训练的批次
            per_device_eval_batch_size=8,  # 验证批次
            warmup_steps=20,  # 学习率预热的步数
            weight_decay=0.01,  # 权重衰减系数
            logging_dir="./bert_logs",  # 日志保存路径:如果想生成这个文件夹，需要安装tensorboard
            logging_steps=10,  # 每隔多少步打印日志
            eval_strategy= "epoch",  # 每轮都进行评估
            save_strategy= "epoch",  # 每轮都进行检查点的模型保存
            load_best_model_at_end=True,  # 加载最优的模型
            save_total_limit=1,  # 只保存一个检查点，其他被覆盖
            metric_for_best_model="eval_loss",  # 评估最优模型的指标（验证集损失）
            fp16=False,  # 禁用混合精度
        )

        # 8. 初始化Trainer
        trainer = Trainer(
            model=self.model, # 加载的模型->原始模型
            args= training_args ,
            train_dataset= dataset_train, # 训练集
            eval_dataset= dataset_test,  # 测试集(验证集)
            compute_metrics= self.compute_metrics
        )

        # 9. 训练模型
        logger.info("开始训练BERT模型")
        trainer.train()
        # 10. 模型保存
        # self.save_model()
        # 11. 对验证集进行训练好的模型验证
        # self.evaluate_model(x_test,labels_test)
    def preprocess_data(self, texts, labels):
        """预处理数据为 BERT 输入格式"""
        encodings = self.tokenizer(
            texts, # 原始文本
            truncation=True, # 截断
            padding='max_length', #最大长度,当不足128,填充0
            max_length=128,
            return_tensors="pt" # 张量pytorch -> pt
        )
        labels = [self.label_map[label] for label in labels]
        # print(f'encodings--》{encodings}')
        # print(f'encodings--》{encodings["input_ids"].shape}')
        # print(f'labels--》{labels}')
        return encodings, labels

    def create_dataset(self, encodings, labels):
        # 自定义Dataset类
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                super().__init__()
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                dicts = {key: value[idx] for key, value in self.encodings.items()}
                dicts["labels"] = torch.tensor(self.labels[idx])
                return dicts

        return Dataset(encodings, labels)

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1) # 按行取最大值对应的下标id
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    """评估模型性能"""

    def evaluate_model(self, x_test, labels):

        # 1.分词器
        encodings = self.tokenizer(
            x_test,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors="pt"
        )

        # 2.获取数据集:create_dataset
        dataset = self.create_dataset(encodings, labels)

        # 3. 加载训练模型:Trainer
        trainer = Trainer(model = self.model) # 加载训练好的模型
        # 4.模型预测
        predict = trainer.predict(dataset)
        # print(predict.predictions)
        # 5. 获取标签:argmax
        pred_labels = np.argmax(predict.predictions, axis=1) # 获取的是预测类别
        # print('获取的标签:',pred_labels) # 0或者1
        # 6. 模型分类报告,混淆矩阵 : 比较真实标签和预测的标签
        true_labels=labels #真实标签
        logger.info('分类报告打印:')
        logger.info(classification_report(true_labels,pred_labels,target_names=['通用知识','专业咨询']))

        logger.info('混淆矩阵打印:')
        logger.info(confusion_matrix(true_labels,pred_labels))

    # 模型分类预测
    def predict_category(self, query): # query:提示词
        # 检查模型是否加载
        if self.model is None:
            # 模型未加载，记录错误
            logger.error("模型未训练或加载")
            # 默认返回通用知识
            return "通用知识"
        # 对查询进行编码
        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # 将编码移到指定设备
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        # 不计算梯度，进行预测
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(**encoding)
            # print(f'outputs--》{outputs}')
            # logits = outputs.logits
            # print(f'logits--》{logits}')
            # # 获取预测结果
            prediction = torch.argmax(outputs.logits, dim=1).item()
        # 根据预测结果返回类别
        return "专业咨询" if prediction == 1 else "通用知识"


if __name__ == '__main__':
    # 1. 模型加载初始化
    query_classify = QueryClassifier()
    # 2. 训练模型
    data_file = r'D:\workspace\workspace_python\python_1022\dev07_rag\integrated_qa_system\rag_qa\data\model\model_generic_2000.json'
    query_classify.train_model(data_file) # 模型训练+ 评估
    # 3.模型预测
    # result = query_classify.predict_category(query="AI的课程大纲是什么")
    # result = query_classify.predict_category(query="写一个函数判断一个数是否是素数")
    # print(result)
