# retrieval/bm25_search.py
# 导入 BM25 算法
import os.path
import sys

from rank_bm25 import BM25Okapi

# 导入数值计算库
import numpy as np

# 将路径添加到环境变量里面
dir_cache = os.path.dirname(__file__)  # 当前文件所在的文件夹
qa_dir = os.path.dirname(dir_cache)  # 上一级路径
sys_dir = os.path.dirname(qa_dir)
# 路径添加到系统环境里面
sys.path.insert(0, qa_dir)
sys.path.insert(0, sys_dir)
# 导入日志
from base import logger
# 导入文本预处理
from utils.preprocess import preprocess_text

class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志

        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data() # 并不是私有方法,只是为了让其他开发者,后续不要对此方法做修改

    def _load_data(self):
        logger.info('BM25检索开始.....')
        # 加载数据
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从 Redis 获取原始问题
        self.original_questions = self.redis_client.get_data(original_key)
        # 从 Redis 获取分词问题
        self.tokenized_questions = self.redis_client.get_data(tokenized_key)

        # 如果 Redis 中没有数据，从 MySQL 加载
        if not self.original_questions or not self.tokenized_questions: #表示查询为None
            logger.info('redis查询,无结果')
            # Reids没有查询到数据,所以需要从 MySQL 获取问题
            self.original_questions = self.mysql_client.fetch_questions() # 原始问题存到self.original_questions
            # 记录无问题 -> 警告
            if not self.original_questions:
                logger.info('MySQL查询无数据')
                return
            # 对每一行问题分词,# 对原始问题分词存到self.tokenized_questions
            self.tokenized_questions = [preprocess_text(doc[0]) for doc in self.original_questions]
            # 存储原始问题到 Redis
            self.redis_client.set_data(original_key,[doc[0] for doc in self.original_questions])
            # 存储分词问题到 Redis
            self.redis_client.set_data(tokenized_key,self.tokenized_questions)

        # 初始化 BM25 模型:BM25Okapi
        self.bm25 = BM25Okapi(self.tokenized_questions)
        # 记录 BM25 初始化成功
        logger.info('bm25初始化成功!')

    def _softmax(self, scores):
        # 计算 Softmax 分数
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数,把数值映射到0-1之间,然后判断是否大于0.85
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        logger.info('BM25检索开始....')
        # 搜索查询 ->判断 ,需要做非空判断
        if not query or not isinstance(query,str): # 字符类型判断
            # 记录无效查询
            logger.info('无效查询....')
            # 返回 None
            return None, False

        # 检查 Redis 缓存:
        cache_answer = self.redis_client.get_data(f'answer:{query}')
        # 返回缓存答案
        if cache_answer:
            logger.info('答案已查询,并返回...')
            return cache_answer, False
        # 查询-> 分词
        tokenized_query_doc = preprocess_text(query)
        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenized_query_doc)
        # 计算 Softmax 分数, 归一化
        softmax_score = self._softmax(scores)
        # 获取最高分索引,目的是获取最大概率
        argmax_id = softmax_score.argmax()
        # 获取最高分
        max_score = softmax_score[argmax_id]
        # 检查是否超过阈值
        if max_score > threshold:
            # 获取原始问题
            original_question = self.original_questions[argmax_id]
            # 获取答案:mysql
            answer = self.mysql_client.fetch_answer(original_question)
            if answer: # 有值才能进入if语句
                # 缓存答案 key: answer:{query}
                self.redis_client.set_data(f'answer:{query}',answer)
                # 记录搜索成功
                logger.info('搜索成功,并缓存写入Redis...')
                # 返回答案
                return answer, False

        # 记录无可靠答案
        logger.info('本次查询,无合适的答案...')
        # 返回 None
        return None, True
if __name__ == '__main__':

    search = BM25Search()
