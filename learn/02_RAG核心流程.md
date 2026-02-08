# 知识点 2: RAG 核心流程

> 📍 **核心文件**: `new_main.py` (IntegratedQASystem) + `rag_qa/core/rag_system.py`  
> ⏱️ **学习时间**: 约 50-70 分钟 (增加BM25集成部分)  
> 🎯 **重要性**: ⭐⭐⭐⭐⭐ (优化版的核心编排层)

---

## 🎯 RAG 的本质：三步曲

```
RAG = Retrieval (检索) + Augment (增强) + Generate (生成)
```

### 整体流程图

```
用户提问: AI课程的学费是多少?
         ↓
    ┌─────────┐
    │查询分类  │
    └─────────┘
       ↓      ↓
   通用知识  专业咨询
       ↓      ↓
   直接LLM  进入RAG流程
              ↓
    ┌──────────────────┐
    │ 1️⃣ RETRIEVE 检索 │
    └──────────────────┘
    - 选择检索策略
    - 执行向量检索
    - 返回相关文档
              ↓
    ┌──────────────────┐
    │ 2️⃣ AUGMENT 增强  │
    └──────────────────┘
    - 提取文档内容
    - 构建增强Prompt
              ↓
    ┌──────────────────┐
    │ 3️⃣ GENERATE 生成 │
    └──────────────────┘
    - LLM基于上下文生成答案
              ↓
         返回最终答案
```

---

## 🎯 优化版RAG流程:三层架构 🆕

```
用户提问: AI课程的学费是多少?
         ↓
    ┌─────────┐
    │查询分类  │ (可选,见知识点6)
    └─────────┘
       ↓      ↓
   通用知识  专业咨询
       ↓      ↓
   直接LLM  进入集成系统
              ↓
    ┌──────────────────┐
    │ 0️⃣ BM25检索(MySQL)│ ← 🆕 第一层
    └──────────────────┘
    - 关键词精确匹配
    - 查询结构化数据
    - 阈值: 0.85
              ↓
         找到答案? 
       是↓      否↓
    直接返回   进入RAG
              ↓
    ┌──────────────────┐
    │ 1️⃣ RAG检索流程   │ ← 第二层
    └──────────────────┘
       ↓
    RETRIEVE → AUGMENT → GENERATE
       ↓
    返回最终答案
```

**为什么这样设计?**
- ✅ **BM25快速**: 简单问题(如"学费多少")直接返回,无需RAG
- ✅ **RAG精准**: 复杂问题(如"课程怎么样")用语义检索
- ✅ **分层回退**: 兼顾速度和准确度

---

## 第零阶段:BM25检索(优先) ⭐⭐⭐⭐⭐ 🆕

### 🎯 为什么需要BM25?

**问题**: RAG虽然强大,但有成本:
- 向量检索需要时间(~1-2秒)
- LLM调用需要费用
- 对于简单问题(如"学费多少")过于复杂

**解决方案**: 分层检索
1. **第一层**: BM25在MySQL中快速查找(~0.1秒)
2. **第二层**: 如果BM25无结果,才调用RAG

### 📊 代码实现

**代码位置**: `new_main.py` 第188-240行

```python
def query(self, query, source_filter=None, session_id=None):
    # 步骤1: BM25检索
    answer, need_rag = self.bm25_search.search(query, threshold=0.85)
    
    if answer:
        # 找到答案,直接返回
        logger.info(f"MySQL答案: {answer}")
        if session_id:
            self.update_session_history(session_id, query, answer)
        yield answer, True  # 一次性返回完整答案
        
    elif need_rag:
        # 无答案,回退到RAG
        logger.info("无可靠MySQL答案,回退到RAG")
        collected_answer = ""
        for token in self.rag_system.generate_answer(query, ...):
            collected_answer += token
            yield token, False  # 流式返回
        
        if session_id:
            self.update_session_history(session_id, query, collected_answer)
        yield "", True  # 标记流结束
```

### 💡 BM25工作原理

**BM25算法**:
- 关键词匹配算法(类似搜索引擎)
- 计算查询与文档的相关性分数
- 分数 ≥ 0.85 认为匹配成功

**示例**:
```python
查询: "AI课程的学费是多少"
→ BM25分词: ["AI", "课程", "学费"]
→ 在MySQL中搜索包含这些词的问答对
→ 找到: "AI课程学费为19800元" (分数: 0.92)
→ 直接返回答案,不进入RAG
```

**BM25检索详情见**: [08_MySQL与BM25检索.md](./08_MySQL与BM25检索.md)

---

## 第一阶段：RETRIEVE（检索）⭐⭐⭐

### 🔍 核心代码：`retrieve_and_merge`

**代码位置**: 第 113-136 行

```python
def retrieve_and_merge(self, query, source_filter=None, strategy=None):
    # 步骤1: 如果未指定策略，自动选择
    if not strategy:
        strategy = self.strategy_selector.select_strategy(query)
    
    # 步骤2: 根据策略执行检索
    ranked_chunks = []
    if strategy == "回溯问题检索":
        ranked_chunks = self._retrieve_with_backtracking(query, source_filter)
    elif strategy == '子查询检索':
        ranked_chunks = self._retrieve_with_subqueries(query, source_filter)
    elif strategy == "假设问题检索":
        ranked_chunks = self._retrieve_with_hyde(query, source_filter)
    else:  # 直接检索
        ranked_chunks = self.vector_store.hybrid_search_with_rerank(
            query, k=conf.RETRIEVAL_K, source_filter=source_filter
        )
    
    # 步骤3: 取Top-M个文档
    final_context_docs = ranked_chunks[:conf.CANDIDATE_M]  # 通常M=2
    
    return final_context_docs
```

---

## 💡 四种检索策略详解

### 策略 1：直接检索（最常用）✅

```python
# 直接用用户问题检索
query = "AI课程的学费是多少?"
→ 直接调用 hybrid_search_with_rerank(query)
→ 返回最相关的文档
```

**适用场景**：问题明确、直接

---

### 策略 2：回溯问题检索 🔄

**代码位置**: 第 42-56 行

```python
def _retrieve_with_backtracking(self, query, source_filter):
    # 1. 用LLM简化问题
    backtrack_prompt = "将以下复杂查询简化为一个更简单的问题："
    simplified_query = self.llm(backtrack_prompt.format(query=query))
    
    # 2. 用简化后的问题检索
    return self.vector_store.hybrid_search_with_rerank(simplified_query, ...)
```

**例子**：
```
原问题：我有一个包含100亿条记录的数据集，想把它存储到Milvus中进行查询。可以吗？
       ↓ LLM简化
简化后：Milvus能处理多大规模的数据？
       ↓ 更容易检索到相关文档
```

**适用场景**：问题过于复杂、冗长

---

### 策略 3：子查询检索 🔀

**代码位置**: 第 59-95 行

```python
def _retrieve_with_subqueries(self, query, source_filter):
    # 1. 用LLM拆分问题
    subquery_prompt = "将以下复杂查询分解为多个简单子查询："
    subqueries_text = self.llm(subquery_prompt.format(query=query))
    subqueries = [q.strip() for q in subqueries_text.split("\n")]
    
    # 2. 分别检索每个子查询
    all_docs = []
    for sub_q in subqueries:
        docs = self.vector_store.hybrid_search_with_rerank(sub_q, k=M//2, ...)
        all_docs.extend(docs)
    
    # 3. 去重并返回
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return list(unique_docs)
```

**例子**：
```
原问题：比较AI和JAVA课程的学费、学时和就业前景
       ↓ LLM拆分
子查询1：AI课程的学费是多少？
子查询2：JAVA课程的学费是多少？
子查询3：AI课程的学时？
子查询4：JAVA课程的学时？
       ↓ 分别检索，合并结果
```

**适用场景**：问题涉及多个方面、需要比较

---

### 策略 4：HyDE（假设文档检索）💡

**代码位置**: 第 98-112 行

```python
def _retrieve_with_hyde(self, query, source_filter):
    # 1. 用LLM生成假设答案
    hyde_prompt = "假设你是用户，想了解以下问题，请生成一个简短的假设答案："
    hypo_answer = self.llm(hyde_prompt.format(query=query))
    
    # 2. 用假设答案检索（而不是用问题）
    return self.vector_store.hybrid_search_with_rerank(hypo_answer, ...)
```

**例子**：
```
问题：人工智能在教育领域的应用有哪些？
     ↓ LLM生成假设答案
假设答案：人工智能在教育领域的应用包括智能辅导系统、自动批改作业、个性化学习推荐...
     ↓ 用这个假设答案去检索
→ 能找到内容相似的真实文档
```

**为什么有效？**
- 问题和答案的语义空间不同
- 答案和答案更容易匹配

**适用场景**：问题抽象、开放性问题

---

## 第二阶段：AUGMENT（增强）⭐⭐⭐

### 🔧 核心代码：构建增强 Prompt

**代码位置**: 第 189-198 行

```python
# 步骤1: 提取文档内容
if context_docs:
    context = "\n\n".join([doc.page_content for doc in context_docs])
    logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
else:
    context = ""
    logger.info("未检索到相关文档，上下文为空")

# 步骤2: 构造增强Prompt
prompt_input = self.rag_prompt.format(
    context=context,            # 检索到的文档
    question=query,             # 用户问题
    phone=conf.CUSTOMER_SERVICE_PHONE  # 兜底信息
)
```

### 📝 Prompt 模板（来自 `prompts.py`）

```python
template="""  
你是一个智能助手，帮助用户回答问题。  
如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。  
如果答案来源于检索到的文档，请在回答中说明。

上下文: {context}  
问题: {question}  

如果无法回答，请回复："信息不足，无法回答，请联系人工客服，电话：{phone}。"  
回答:  
"""
```

### 💡 实际增强示例

```python
# 检索到的文档
doc1 = "AI课程学费为19800元，包含线上课程和实战项目。"
doc2 = "课程学时为6个月，每周学习20小时。"

# 用户问题
query = "AI课程的学费是多少？"

# 增强后的Prompt
"""
你是一个智能助手，帮助用户回答问题。
如果提供了上下文，请基于上下文回答。

上下文: 
AI课程学费为19800元，包含线上课程和实战项目。

课程学时为6个月，每周学习20小时。

问题: AI课程的学费是多少？

如果无法回答，请回复："信息不足，无法回答，请联系人工客服，电话：12345678。"
回答:
"""
```

**关键设计**：
- ✅ 明确指示基于上下文回答
- ✅ 提供兜底机制（无法回答时的提示）
- ✅ 上下文和问题分开，结构清晰

---

## 第三阶段：GENERATE（生成）⭐⭐⭐

### 🤖 核心代码：LLM 生成

**代码位置**: 第 202-207 行

```python
try:
    # 调用LLM生成答案
    answer = self.llm(prompt_input)
except Exception as e:
    logger.error(f"调用 LLM 生成最终答案失败: {e}")
    # 异常处理：返回兜底信息
    answer = f"抱歉，处理您的专业咨询问题时出错。请联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
```

### 🎯 LLM 的角色转变

**传统方式**：
```
用户：AI课程的学费是多少？
LLM：我不知道具体的课程学费，建议您咨询课程提供方。
```

**RAG 方式**：
```
用户：AI课程的学费是多少？
系统检索 → 找到："AI课程学费为19800元..."
增强Prompt → 把文档塞进去
LLM：根据课程资料，AI课程学费为19800元，包含线上课程和实战项目。
```

**核心差异**：
- ❌ **传统**：LLM 依赖训练数据（可能没有或过时）
- ✅ **RAG**：LLM 依赖检索到的实时文档（准确且可追溯）

---

## 🔄 完整流程实战演示

### 场景：用户询问 "AI课程的学费是多少？"

```python
# ====== 阶段0: 初始化 ======
rag_system = RAGSystem(vector_store, llm)

# ====== 阶段1: RETRIEVE ======
# 1.1 选择策略
strategy = strategy_selector.select_strategy("AI课程的学费是多少？")
→ 返回："直接检索"

# 1.2 执行检索
context_docs = vector_store.hybrid_search_with_rerank(
    query="AI课程的学费是多少？",
    k=3,
    source_filter="ai"
)
→ 返回2个父文档：
  - doc1: "AI课程学费为19800元，包含线上课程和实战项目。学习周期6个月..."
  - doc2: "报名AI课程可享受分期付款，首付3000元..."

# ====== 阶段2: AUGMENT ======
# 2.1 提取内容
context = doc1.page_content + "\n\n" + doc2.page_content

# 2.2 构建Prompt
prompt = f"""
你是一个智能助手，帮助用户回答问题。
上下文: {context}
问题: AI课程的学费是多少？
回答:
"""

# ====== 阶段3: GENERATE ======
# 3.1 调用LLM
answer = llm(prompt)
→ "根据课程资料，AI课程学费为19800元，包含线上课程和实战项目。课程支持分期付款，首付3000元即可开始学习。"

# 3.2 返回答案
return answer
```

---

## 🎓 核心概念总结

### ✅ 知识点检查清单

| 概念 | 核心理解 |
|------|---------|
| **Retrieve** | 用策略从知识库检索相关文档 |
| **Augment** | 把文档塞进 Prompt，增强 LLM 的上下文 |
| **Generate** | LLM 基于增强后的 Prompt 生成答案 |
| **策略选择** | 根据问题类型选择最佳检索策略 |
| **Prompt 工程** | 设计合理的模板，引导 LLM 输出 |

### 🔑 RAG 的核心价值

1. **知识实时性**：不依赖 LLM 训练时的数据
2. **可追溯性**：答案来自具体文档，可验证
3. **成本效率**：不需要重新训练模型
4. **灵活性**：随时更新知识库

### ⚠️ 常见误区

| 误区 | 正确理解 |
|------|---------|
| ❌ RAG 就是把文档全扔给 LLM | ✅ 需要精确检索 + 策略选择 + Prompt 设计 |
| ❌ 检索越多文档越好 | ✅ 过多无关文档会干扰，Top-2~3 足矣 |
| ❌ LLM 会自动找到答案 | ✅ 需要 Prompt 明确指示"基于上下文回答" |

---

## 📊 配置参数说明

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `RETRIEVAL_K` | config.ini | 3 | 初步检索的文档数量 |
| `CANDIDATE_M` | config.ini | 2 | 最终返回的文档数量 |
| `PARENT_CHUNK_SIZE` | config.ini | 1200 | 父文档大小（字符） |
| `CHILD_CHUNK_SIZE` | config.ini | 300 | 子文档大小（字符） |

---

## 📚 进阶学习资源

1. **RAG 综述论文**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. **检索策略对比**: 搜索 "Advanced RAG techniques"
3. **Prompt 工程**: [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

**上一个知识点**: [01_向量检索与存储.md](./01_向量检索与存储.md)  
**下一个知识点**: [03_检索策略选择器.md](./03_检索策略选择器.md)
