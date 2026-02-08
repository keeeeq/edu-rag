# 导入 MySQL 客户端
import os.path
import sys

from db.MySQLClient import MySQLClient
# 导入 Redis 客户端
from cache.RedisClient import RedisClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search

# 将路径添加到环境变量里面
qa_dir = os.path.dirname(__file__) #当前文件所在的文件夹
sys_dir = os.path.dirname(qa_dir) # 上一级路径
# 路径添加到系统环境里面
sys.path.insert(0,qa_dir)
sys.path.insert(0,sys_dir)

# 导入日志
from base import logger

# 导入时间库
import time


# 作用:是模拟前端问答页面,模拟提问的
def main():
    # 初始化 MySQL 系统
    mysql_system = MySQLQASystem()
    # 打印欢迎信息
    print("\n欢迎使用 MySQL 问答系统！")
    print("输入查询进行问答，输入 'exit' 退出。")
    # 循环
    while True:
        # 获取用户输入
        question = input('请输入您的问题:').strip() # strip字符两端去除空格
        # 执行查询
        answer = mysql_system.query(question)
        # 打印答案
        logger.info(f'查询答案为:{answer}')

if __name__ == "__main__":
    # 运行主程序
    main()
class MySQLQASystem:
    def __init__(self):
        logger.info('主类测试初始化....')
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        # 记录查询信息
        logger.info('用户查询开始.....')
        # 查询 MySQL 系统: 开始时间
        start_time = time.time()

        # 执行 BM25 搜索
        self.bm25_search.search(query,threshold=0.85)
        # 结束时间
        end_time = time.time()
        # 记录处理时间
        process_time = end_time - start_time
        logger.info(f'用户查询耗时为:{process_time:.2f}')
        # 返回答案
