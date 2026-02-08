# 知识点 8: MySQL与BM25检索

> 📍 **核心文件**: `mysql_qa/retrieval/bm25_search.py` + `mysql_qa/db/MySQLClient.py`  
> ⏱️ **学习时间**: 约 40-50 分钟  
> 🎯 **重要性**: ⭐⭐⭐⭐⭐ (优化版的第一层检索)

---

## 🎯 核心概念:分层检索架构

### 为什么需要BM25?

**问题场景:**
```
用户问题: "AI课程的学费是多少?"

传统RAG方案:
1. 向量化查询 (~0.5秒)
2. Milvus检索 (~0.8秒)
3. LLM生成 (~2秒)
总耗时: ~3.3秒

优化方案(BM25优先):
1. BM25检索MySQL (~0.1秒)
2. 找到答案,直接返回
总耗时: ~0.1秒 ✅
```

**优势:**
- ✅ **速度快**: 简单问题0.1秒返回
- ✅ **成本低**: 不调用LLM,节省费用
- ✅ **准确度高**: 精确匹配,不会"幻觉"

---

## 第一部分:BM25算法原理

### 🔍 什么是BM25?

**BM25 (Best Matching 25)**: 一种基于关键词的相关性评分算法

**核心思想:**
```
相关性分数 = Σ (词频 × IDF × 归一化因子)

其中:
- 词频(TF): 词在文档中出现的次数
- IDF: 逆文档频率(稀有词权重更高)
- 归一化: 考虑文档长度
```

### 💡 实际例子

**查询:** "AI课程的学费"

**文档1:** "AI课程学费为19800元,包含线上课程和实战项目"
**文档2:** "JAVA课程学费为18800元"
**文档3:** "AI学科介绍:人工智能是..."

**BM25评分:**
```python
文档1: 
- "AI" 出现1次 × IDF(中) = 0.5
- "课程" 出现1次 × IDF(高) = 0.6
- "学费" 出现1次 × IDF(高) = 0.8
→ 总分: 1.9 ✅ 最高

文档2:
- "课程" 出现1次 = 0.6
- "学费" 出现1次 = 0.8
→ 总分: 1.4

文档3:
- "AI" 出现1次 = 0.5
→ 总分: 0.5
```

---

## 第二部分:MySQL数据结构

### 📊 jpkb表设计

**代码位置**: `mysql_qa/db/MySQLClient.py` 第43-48行

```sql
CREATE TABLE IF NOT EXISTS jpkb(
    id           INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(20),      -- 学科名称(ai/java/test等)
    question     VARCHAR(1000),    -- 问题
    answer       VARCHAR(1000)     -- 答案
);
```

**数据示例:**
```
| id | subject_name | question              | answer                    |
|----|-------------|-----------------------|---------------------------|
| 1  | ai          | AI课程的学费是多少?    | AI课程学费为19800元...    |
| 2  | java        | JAVA课程学时多长?     | JAVA课程学时为6个月...    |
| 3  | ai          | 如何报名AI课程?       | 报名方式:线上填表...      |
```

---

## 第三部分:Redis缓存机制

### 🚀 为什么需要Redis?

**问题:** BM25需要加载所有问题到内存进行分词和评分

**解决方案:** 使用Redis缓存

**代码位置**: `mysql_qa/cache/RedisClient.py`

```python
class RedisClient:
    def __init__(self):
        self.client = redis.Redis(
            host=Config().REDIS_HOST,
            port=Config().REDIS_PORT,
            password=Config().REDIS_PASSWORD,
            db=Config().REDIS_DB,
            decode_responses=True
        )
```

### 缓存策略

**键值设计:**
```python
# 缓存所有问题列表
key: "jpkb:questions"
value: ["AI课程的学费是多少?", "JAVA课程学时多长?", ...]

# 缓存问题→答案映射
key: "jpkb:qa:{question_hash}"
value: "AI课程学费为19800元..."
```

---

## 第四部分:BM25检索流程

### 🔍 核心代码详解

**代码位置**: `mysql_qa/retrieval/bm25_search.py`

```python
class BM25Search:
    def __init__(self, redis_client, mysql_client):
        self.redis = redis_client
        self.mysql = mysql_client
        self.bm25 = None
        self.questions = []
        self._load_questions()  # 初始化时加载问题
    
    def _load_questions(self):
        """从MySQL加载所有问题到内存"""
        # 1. 尝试从Redis获取
        cached = self.redis.get("jpkb:questions")
        if cached:
            self.questions = json.loads(cached)
        else:
            # 2. 从MySQL查询
            results = self.mysql.fetch_questions()
            self.questions = [q[0] for q in results]
            # 3. 缓存到Redis
            self.redis.set("jpkb:questions", json.dumps(self.questions))
        
        # 4. 初始化BM25模型
        tokenized = [jieba.lcut(q) for q in self.questions]
        self.bm25 = BM25Okapi(tokenized)
```

### 检索流程

**代码位置**: 第XX-XX行

```python
def search(self, query, threshold=0.85):
    """
    BM25检索
    
    Args:
        query: 用户问题
        threshold: 分数阈值(默认0.85)
    
    Returns:
        (answer, need_rag): 答案和是否需要RAG
    """
    # 步骤1: 分词
    query_tokens = jieba.lcut(query)
    
    # 步骤2: BM25评分
    scores = self.bm25.get_scores(query_tokens)
    
    # 步骤3: 找到最高分
    max_score = max(scores)
    max_idx = scores.argmax()
    
    # 步骤4: 判断是否超过阈值
    if max_score >= threshold:
        # 找到答案
        matched_question = self.questions[max_idx]
        answer = self.mysql.fetch_answer(matched_question)
        return answer, False  # 不需要RAG
    else:
        # 未找到答案
        return None, True  # 需要RAG
```

---

## 第五部分:与RAG的集成

### 🔗 调用流程

**代码位置**: `new_main.py` 第188-240行

```python
def query(self, query, source_filter=None, session_id=None):
    # 第一层: BM25检索
    answer, need_rag = self.bm25_search.search(query, threshold=0.85)
    
    if answer:
        # ✅ BM25找到答案
        logger.info(f"MySQL答案: {answer}")
        yield answer, True
    elif need_rag:
        # ❌ BM25无结果,回退到RAG
        logger.info("无可靠MySQL答案,回退到RAG")
        for token in self.rag_system.generate_answer(query, ...):
            yield token, False
```

### 完整流程图

```
用户查询: "AI课程的学费是多少?"
    ↓
┌─────────────────────────┐
│ 1. BM25分词             │
│ ["AI", "课程", "学费"]  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 2. 计算BM25分数         │
│ 文档1: 1.9 ✅           │
│ 文档2: 1.4             │
│ 文档3: 0.5             │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 3. 判断阈值             │
│ max_score(1.9) >= 0.85? │
│ → 是 ✅                 │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ 4. 返回答案             │
│ "AI课程学费为19800元..." │
└─────────────────────────┘
```

---

## 第六部分:性能优化

### ⚡ 优化点

1. **Redis缓存**
   - 问题列表缓存,避免重复查询MySQL
   - 答案缓存,提高响应速度

2. **分词预处理**
   - 初始化时完成所有问题的分词
   - 查询时只需分词一次

3. **阈值调优**
   - 默认0.85,可根据实际情况调整
   - 阈值越高,准确度越高但召回率越低

### 📊 性能对比

| 方案 | 平均耗时 | 准确度 | 成本 |
|------|---------|--------|------|
| **BM25检索** | ~0.1秒 | 高(精确匹配) | 极低 |
| **RAG检索** | ~3秒 | 高(语义理解) | 中 |
| **直接LLM** | ~2秒 | 中(可能幻觉) | 高 |

---

## ✅ 核心概念检查清单

- [x] **BM25算法**: 基于关键词的相关性评分
- [x] **MySQL存储**: jpkb表存储问答对
- [x] **Redis缓存**: 缓存问题列表和答案
- [x] **分层检索**: BM25优先,RAG回退
- [x] **阈值设置**: 0.85平衡准确度和召回率
- [x] **性能优势**: 简单问题快速返回

---

**上一个知识点**: [07_文档加载器.md](./07_文档加载器.md)  
**下一个知识点**: [09_会话管理与历史.md](./09_会话管理与历史.md)
