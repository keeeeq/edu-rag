# db/mysql_client.py
# 导入 MySQL 连接库
import os.path
import sys

import pymysql
# 导入pandas
import pandas as pd

# 将路径添加到环境变量里面
dir_cache = os.path.dirname(__file__)  # 当前文件所在的文件夹
qa_dir = os.path.dirname(dir_cache)  # 上一级路径
sys_dir = os.path.dirname(qa_dir)
# 路径添加到系统环境里面
sys.path.insert(0, qa_dir)
sys.path.insert(0, sys_dir)

# 导入配置和日志
from base import Config, logger


class MySQLClient:
    def __init__(self):
        logger.info('创建数据库初始化连接......')
        # 创建初始化连接
        try:
            self.connect = pymysql.connect(
                user=Config().MYSQL_USER,
                password=Config().MYSQL_PASSWORD,
                host=Config().MYSQL_HOST,
                database=Config().MYSQL_DATABASE,
                charset='utf8mb4'  # 支持emoji和特殊字符
            )
            # 获取cursor(游标)对象 ,通过该对象,可以实现对表的增删改查
            self.cursor = self.connect.cursor()
            logger.info('数据库客户端连接对象初始化成功....')
        except pymysql.MySQLError as e:
            logger.info(f'数据库初始化异常:{e}')

    def create_table(self):
        logger.info('创建表结构.....')
        sql = '''
            create table if not exists jpkb(
                    id           int auto_increment primary key,
                    subject_name varchar(20)   ,
                    question     varchar(1000) ,
                    answer       varchar(1000) 
            );
        '''
        try:
            self.cursor.execute(sql)
            logger.info('表创建成功....')
        except pymysql.MySQLError as e:
            logger.info(f'表创建失败:{e}')

    def insert_data(self, csv_path):
        logger.info('插入数据.......')
        sql = 'insert into jpkb(subject_name, question, answer)  values (%s,%s,%s);'
        # 读取本地知识文件
        try:
            df = pd.read_csv(csv_path)
            for id, row in df.iterrows():
                # 数据插入
                self.cursor.execute(sql, (row['学科名称'], row['问题'], row['答案']))
            self.connect.commit()  # 提交
            logger.info('数据插入成功....')
        except pymysql.MySQLError as e:
            logger.info(f'数据插入失败:{e}')

    def fetch_questions(self):
        # 获取所有问题
        logger.info('查询所有的问题...')
        try:
            self.cursor.execute('select question from jpkb')
            tuple_questions = self.cursor.fetchall()
            logger.info('所有的问题查询完毕...')
            return tuple_questions
        except pymysql.MySQLError as e:
            logger.info(f'所有的问题查询失败:{e}')

    def fetch_answer(self, question):
        # 获取指定问题的答案
        logger.info('查询所有的问题...')
        try:
            self.cursor.execute('select answer from jpkb where question = %s',question)
            tuple_answer = self.cursor.fetchone()
            logger.info('答案查询完毕...')
            return tuple_answer
        except pymysql.MySQLError as e:
            logger.info(f'答案查询失败:{e}')

    def close(self):
        # 关闭数据库连接
        try:
            # 关闭连接
            self.connect.close()
            # 记录关闭成功
            logger.info("MySQL 连接已关闭")
        except pymysql.MySQLError as e:
            # 记录关闭失败
            logger.error(f"关闭连接失败: {e}")


if __name__ == '__main__':
    pass
    client = MySQLClient()
    # client.create_table()
    # path = '../data/JP学科知识问答.csv'
    # client.insert_data(path)
    # questions = client.fetch_questions()
    # print(type(questions))
    # logger.info(f'查询结果为:{questions}')
    answer = client.fetch_answer('关联子查询的执行顺序是什么')
    logger.info(f'根据问题查询结果为:{answer}')
    # 程序执行完毕,关闭数据库连接
    client.close()