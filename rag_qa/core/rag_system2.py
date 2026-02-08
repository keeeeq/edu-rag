# -*-coding:utf-8-*-
# core/rag_system.py 源码
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base import logger, Config
from query_classifier import QueryClassifier  #   导入查询分类器
from strategy_selector import StrategySelector  #   导入策略选择器
from vector_store import VectorStore # 导入向量数据库对象

conf = Config()

#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        classifier_path = os.path.join(rag_qa_path, 'core', 'bert_query_classifier')
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector()

    # 3.3 提示词:回溯问题
    #   定义类似私有方法，使用回溯问题进行检索
    def _retrieve_with_backtracking(self, query, source_filter):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        try:
            #   获取回溯问题生成的 Prompt 模板
            prompt = RAGPrompts.backtracking_prompt()
            # 格式化模板,参数替换
            prompt_input = prompt.format(query=query)
            #   调用大语言模型生成回溯问题
            # 扩展: ctrl +G :快速定位
            simple_query = self.llm(prompt_input)
            #   使用回溯问题进行混合检索，并返回检索结果
            docs = self.vector_store.hybrid_search_with_rerank(simple_query, k=conf.RETRIEVAL_K, source_filter=source_filter)
            logger.info(f'检索策略-回溯问题查询完成!')
            return docs
        except Exception as e:
            logger.error(f'检索策略-回溯问题查询异常:{e}!')
            return []

    # 3.2 提示词:子查询进行检索
    #   定义类似私有方法，使用子查询进行检索
    def _retrieve_with_subqueries(self, query, source_filter):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        prompt = RAGPrompts.subquery_prompt()
        prompt_input = prompt.format(query=query)
        #   调用大语言模型生成子查询列表,并按照\n分割
        sub_queries = self.llm(prompt_input)
        # 按照\n 把字符串分割成两个子查询 - > []
        sub_queries = [i.strip() for i in sub_queries.split('\n')]
        # print(sub_queries)
        #   判断子查询是否有返回数据,如果无,返回[]
        if not sub_queries :
            logger.info('子查询通过LLM,无返回数据...')
            return []
        #   初始化空列表，用于存储所有子查询的检索结果
        all_docs = []

        #   遍历每个子查询
        for sub_query in sub_queries:
            # 使用子查询混合检索,并添加入列表
            sub_docs = self.vector_store.hybrid_search_with_rerank(sub_query, k=conf.RETRIEVAL_K, source_filter=source_filter)
            # 把子查询的结果存储到all_docs
            all_docs.extend(sub_docs)
        #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
        #  采用字典去重,字典的key是唯一的,整个结果就是唯一的
        #   遍历 -> dict , key = 文档内容
        dict_docs = { doc.page_content:doc for doc in all_docs}

        #  获取字典values ,并转列表
        docs = list(dict_docs.values())

        # 返回文档
        return docs

    # 3.1 假设文档提示词处理
    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")

        try:
            #   获取假设问题生成的 Prompt 模板
            prompt = RAGPrompts.hyde_prompt()
            #   调用大语言模型生成假设答案
            hyde_query = self.llm(prompt.format(query=query))
            #   使用假设答案进行检索，并返回检索结果
            docs = self.vector_store.hybrid_search_with_rerank(hyde_query, k=conf.RETRIEVAL_K,source_filter=source_filter)
            logger.info(f'假设问题,通过LLM,查询完毕...')
            return docs
        except Exception as e:
            logger.error(f'假设问题,通过LLM,查询异常:{e}')
            return []

    # 2.策略选择,根据传输的策略,来选择具体的查询策略,并做后续的向量数据检索
    # 并基于具体的查询策略,从向量库里面返回Top-k
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        #   如果未指定检索策略，则使用策略选择器选择
        if not strategy: # 如果策略为空
            logger.info('无查询策略,可供选择,直接返回...')
            return []

        # 根据检索策略选择不同的检索方式
        ranked_chunks = [] # 初始化

        # 回溯问题检索
        if strategy == '回溯问题检索':
            ranked_chunks = self._retrieve_with_backtracking(query,source_filter)
        # 子查询检索
        elif strategy == '子查询检索':
            ranked_chunks = self._retrieve_with_subqueries(query, source_filter)
        # 假设问题检索
        elif strategy == '假设问题检索':
            ranked_chunks = self._retrieve_with_hyde(query, source_filter)
        # 直接检索
        else:
            # 直接检索：self.vector_store
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(query, k=conf.RETRIEVAL_K, source_filter=source_filter)
        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_chunks)} 个候选文档 (可能已是父文档)")
        # 查询从排名之后的前N个
        final_context_docs = ranked_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        # 返回数据
        return final_context_docs

    # 1. 生成答案
    def generate_answer(self, query, source_filter=None):
        pass
        #   记录查询开始时间,目的是记录程序的消耗时间,当记录好耗时之后,方便日后性能调优
        start_time = time.time()
        #   查询问题类型
        query_category = self.query_classifier.predict_category(query)
        logger.info(f'查询类别为:{query_category}')
        #   如果查询属于“通用知识”类别，则直接使用 LLM 回答
        if   query_category == '通用知识':
            # 设置提示词
            prompt = RAGPrompts.rag_prompt()
            # 变量替换
            prompt_input = prompt.format(context='', question=query, phone='123456')
            # 模型调用
            try:
                answer = self.llm(prompt_input)
                logger.info(f'通用知识问答,已回答完毕.....')
                process_time = time.time() - start_time
                logger.info(f'通用知识问答,耗时为:{process_time:.2f}')
                return answer
            except Exception as e:
                logger.error(f'通用知识检索失败....')
                return None

        #   否则，进行 RAG 检索并生成答案
        logger.info('查询为专业咨询,执行RAG流程....')
        #   选择检索策略
        template = self.strategy_selector.strategy_prompt_template
        # 格式化模版,替换变量
        template_prompt = template.format(query=query)
        # LLM模型调用,由大模型帮我们选择具体的查询策略
        try:
            strategy = self.llm(template_prompt)
            logger.info(f'模型调用,查询策略为:{strategy}')
        except Exception as e :
            logger.error(f'在模型策略选择的时候,发生了模型调用异常....')

        # 根据检索策略,选择具体的查询策略,来执行后续的处理
        #  检索相关文档: retrieve_and_merge
        final_docs = self.retrieve_and_merge(query, source_filter, strategy)
        # print(final_docs)
        #  准备上下文:doc.page_content
        # 遍历文档, 使用换行符拼接文档 -> 把final_docs 里面的多条语句,通过\n,拼接成一个context
        context = '\n'.join([doc.page_content for doc in final_docs])

        #   构造 Prompt，调用大语言模型生成答案
        try:
            rag_prompt = RAGPrompts.rag_prompt()
            # ["context", "question", "phone"],
            prompt_format = rag_prompt.format(context=context, question=query, phone=conf.CUSTOMER_SERVICE_PHONE)
            answer = self.llm(prompt_format)
            #   记录查询处理完成的日志
            logger.info(f'生成的最终结果为:{answer}')
            process_time = time.time() - start_time
            logger.info(f'专业知识问答,耗时为:{process_time:.2f}')
            # 返回结果
            return answer
        except Exception as e:
            logger.error(f'生成结果失败,异常为:{e}')
            return f'查询失败,请联系人工服务,手机号为:{conf.CUSTOMER_SERVICE_PHONE}'

if __name__ == '__main__':
    vector_store = VectorStore()
    llm = StrategySelector().call_dashscope
    # print(llm(prompt="什么是AI"))
    rag_system = RAGSystem(vector_store, llm)
    # print('=='*20)

    answer = rag_system.generate_answer(query="AI学科的课程大纲内容有什么",source_filter="ai")
    # answer = rag_system.generate_answer(query="你是谁?",source_filter="ai")
    print(answer)
    # 回溯问题
    # answer = rag_system._retrieve_with_backtracking(query="我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？", source_filter="ai")
    # print(answer)
    # 子查询
    # answer=rag_system._retrieve_with_subqueries(query="AI和JAVA的区别是什么？", source_filter="ai")
    # print(answer)
    # 假设问题
    # result = rag_system._retrieve_with_hyde(query="AI课程的NLP的技术有哪些?",source_filter="ai")
    # print(result)

