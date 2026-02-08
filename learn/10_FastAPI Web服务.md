# 知识点 10: FastAPI Web服务

> 📍 **核心文件**: `app.py` + `static/index.html`  
> ⏱️ **学习时间**: 约 40-50 分钟  
> 🎯 **重要性**: ⭐⭐⭐⭐ (Web应用的基础)

---

## 🎯 核心概念:为什么需要Web服务?

**问题:** 命令行交互不友好,无法多用户使用

**解决方案:** FastAPI + WebSocket
- ✅ Web界面友好
- ✅ 支持多用户并发
- ✅ 流式输出实时显示
- ✅ 跨平台访问

---

## 第一部分:FastAPI架构

### 🏗️ 应用结构

**代码位置**: `app.py`

```python
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 创建应用实例
app = FastAPI(title="问答系统API")

# 配置CORS(允许跨域)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 创建QA系统实例
qa_system = IntegratedQASystem()
```

---

## 第二部分:API端点

### 📡 RESTful API

#### 1. 创建会话

```python
@app.post("/api/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}
```

**用途:** 生成唯一会话ID

#### 2. 非流式查询

```python
@app.post("/api/query")
async def query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    
    # 检查是否为问候语
    greeting = check_greeting(request.query)
    if greeting:
        return {
            "answer": greeting,
            "is_streaming": False,
            "session_id": session_id
        }
    
    # BM25检索
    answer, need_rag = qa_system.bm25_search.search(request.query)
    if need_rag:
        return {
            "answer": "请使用WebSocket获取流式响应",
            "is_streaming": True,
            "session_id": session_id
        }
    
    return {
        "answer": answer,
        "is_streaming": False,
        "session_id": session_id
    }
```

#### 3. 获取历史

```python
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    history = qa_system.get_session_history(session_id)
    return {"session_id": session_id, "history": history}
```

#### 4. 清除历史

```python
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    success = qa_system.clear_session_history(session_id)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=500)
```

---

## 第三部分:WebSocket流式输出 ⭐⭐⭐

### 🌊 为什么需要WebSocket?

**HTTP vs WebSocket:**
```
HTTP (传统):
用户 → 请求 → 服务器
用户 ← 等待3秒 ← 服务器
用户 ← 完整答案 ← 服务器

WebSocket (流式):
用户 → 连接 → 服务器
用户 ← "根据" ← 服务器 (0.1秒)
用户 ← "课程" ← 服务器 (0.2秒)
用户 ← "资料" ← 服务器 (0.3秒)
...实时显示
```

### 💻 WebSocket实现

**代码位置**: `app.py` 第146-234行

```python
@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 接受连接
    
    try:
        while True:
            # 1. 接收客户端消息
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            query = request_data.get("query")
            session_id = request_data.get("session_id")
            
            # 2. 发送开始标志
            await websocket.send_json({
                "type": "start",
                "session_id": session_id
            })
            
            # 3. 检查问候语
            greeting = check_greeting(query)
            if greeting:
                await websocket.send_json({
                    "type": "token",
                    "token": greeting
                })
                await websocket.send_json({"type": "end"})
                break
            
            # 4. 调用QA系统(流式)
            collected_answer = ""
            for token, is_complete in qa_system.query(query, session_id=session_id):
                collected_answer += token
                
                if token:
                    # 发送token
                    await websocket.send_json({
                        "type": "token",
                        "token": token
                    })
                
                if is_complete:
                    # 发送结束标志
                    await websocket.send_json({
                        "type": "end",
                        "is_complete": True
                    })
                    break
                
                await asyncio.sleep(0.01)  # 控制速度
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "error": str(e)})
```

---

## 第四部分:前端界面

### 🖥️ HTML结构

**代码位置**: `static/index.html`

```html
<!DOCTYPE html>
<html>
<head>
    <title>智能问答系统</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="chat-container">
        <div id="chat-history"></div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="输入问题...">
            <button onclick="sendMessage()">发送</button>
        </div>
    </div>
</body>
</html>
```

### 🎨 CSS样式

```css
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.message {
    margin: 10px 0;
    padding: 10px;
    border-radius: 8px;
}

.user-message {
    background: #007bff;
    color: white;
    text-align: right;
}

.system-message {
    background: #f1f1f1;
    color: black;
}
```

### 📡 JavaScript交互

```javascript
let socket = null;
let currentSessionId = null;

// 创建会话
async function createSession() {
    const response = await fetch('/api/create_session', {
        method: 'POST'
    });
    const data = await response.json();
    currentSessionId = data.session_id;
}

// 发送消息
function sendMessage() {
    const input = document.getElementById('user-input');
    const query = input.value.trim();
    
    if (!query) return;
    
    // 显示用户消息
    addMessage('user', query);
    input.value = '';
    
    // 连接WebSocket
    socket = new WebSocket('ws://localhost:8003/api/stream');
    
    socket.onopen = () => {
        // 发送查询
        socket.send(JSON.stringify({
            query: query,
            session_id: currentSessionId
        }));
    };
    
    let accumulatedContent = '';
    const messageElement = createMessageElement('system');
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch(data.type) {
            case 'token':
                // 累积内容
                accumulatedContent += data.token;
                // 使用marked.js渲染Markdown
                messageElement.innerHTML = marked.parse(accumulatedContent);
                scrollToBottom();
                break;
                
            case 'end':
                console.log('流式响应结束');
                socket.close();
                break;
                
            case 'error':
                console.error('错误:', data.error);
                break;
        }
    };
}

// 添加消息到界面
function addMessage(type, content) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    messageDiv.innerHTML = marked.parse(content);
    chatHistory.appendChild(messageDiv);
    scrollToBottom();
}
```

---

## 第五部分:Markdown渲染

### 📝 为什么需要Markdown?

**LLM输出格式:**
```markdown
根据课程资料:

## AI课程信息
- **学费**: 19800元
- **学时**: 6个月
- **内容**: 
  1. Python基础
  2. 机器学习
  3. 深度学习

```python
# 示例代码
def hello():
    print("Hello AI!")
```
```

**使用marked.js渲染:**
- ✅ 标题、列表、代码块正确显示
- ✅ 格式美观易读
- ✅ 支持代码高亮

---

## 第六部分:问候语快速响应

### 👋 预定义问候语

**代码位置**: `app.py` 第37-54行

```python
GREETING_PATTERNS = [
    {
        "pattern": r"^(你好|您好|hi|hello)",
        "response": "你好!我是黑马程序员,很高兴为你服务!"
    },
    {
        "pattern": r"^(你是谁|您是谁)",
        "response": "我是黑马程序员,你的智能学习助手!"
    },
    {
        "pattern": r"^(在吗|在不在)",
        "response": "我在!随时为你解答问题!"
    }
]

def check_greeting(query: str) -> Optional[str]:
    for pattern_info in GREETING_PATTERNS:
        if re.match(pattern_info["pattern"], query, re.IGNORECASE):
            return pattern_info["response"]
    return None
```

**优势:**
- ✅ 无需调用BM25或RAG
- ✅ 毫秒级响应
- ✅ 节省资源

---

## 第七部分:完整交互流程

### 🔄 用户提问流程

```
1. 用户打开页面
   ↓
2. JavaScript创建会话
   POST /api/create_session
   ← session_id
   ↓
3. 用户输入问题
   ↓
4. 建立WebSocket连接
   ws://localhost:8003/api/stream
   ↓
5. 发送查询
   → {query: "AI课程学费?", session_id: "..."}
   ↓
6. 服务器处理
   - 检查问候语 → 否
   - BM25检索 → 无结果
   - RAG检索 → 开始
   ↓
7. 流式返回
   ← {type: "token", token: "根据"}
   ← {type: "token", token: "课程"}
   ← {type: "token", token: "资料"}
   ...
   ← {type: "end"}
   ↓
8. 前端实时显示
   Markdown渲染
   ↓
9. 保存历史
   INSERT INTO conversations
```

---

## ✅ 核心概念检查清单

- [x] **FastAPI**: 现代Python Web框架
- [x] **WebSocket**: 实时双向通信
- [x] **流式输出**: token逐个返回
- [x] **Markdown渲染**: marked.js美化显示
- [x] **会话管理**: session_id标识用户
- [x] **CORS**: 允许跨域访问
- [x] **静态文件**: 提供HTML/CSS/JS

---

**上一个知识点**: [09_会话管理与历史.md](./09_会话管理与历史.md)  
**下一个知识点**: [11_配置管理系统.md](./11_配置管理系统.md)
