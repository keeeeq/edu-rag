
import os # 获取文件路径
import sys # 系统环境,可以把模块的路径添加到系统环境里面

# D:\workspace\workspace_python\python_1022\dev07_rag\integrated_qa_system\base
print(__file__) # 打印的是该文件的绝对路径
cur_dir = os.path.dirname(__file__) # 当前文件所在文件夹的路径
print(cur_dir) #
qa_dir = os.path.dirname(cur_dir) # 再上一级文件夹的路径
print(qa_dir)#
# 添加到系统环境里面
# path = sys.path
# print(path,type(path))
sys.path.insert(0,cur_dir)
sys.path.insert(0,qa_dir)

from config import Config
from logger import logger

