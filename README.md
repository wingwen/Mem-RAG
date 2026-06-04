# Mem-RAG

基于 **Milvus + BM25 + RRF** 的混合检索增强对话系统，集成**结构化主题记忆**、**LangGraph Agentic RAG 状态机**、流式 RAG、多格式知识库管理与**可量化检索评估**。

> 适用场景：私有知识库问答、多轮对话助手、RAG 检索效果对比实验。

---

## 目录

- [核心亮点](#核心亮点)
- [系统架构](#系统架构)
- [结构化记忆](#结构化记忆)
- [知识库入库 Pipeline](#知识库入库-pipeline)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [检索评估](#检索评估)
- [API 接口](#api-接口)
- [安全说明](#安全说明)
- [故障排除](#故障排除)

---

## 核心亮点

| 模块 | 能力 |
|------|------|
| **混合检索** | Milvus 向量 + BM25 稀疏 + 自实现 RRF 融合 |
| **结构化记忆** | 主题分类、主题摘要、同主题优先召回，提升多轮连贯性 |
| **知识库 Pipeline** | TXT / PDF / Word → 混合分块 → 向量 + BM25 双索引 |
| **全栈交付** | FastAPI 流式 SSE + Vue3 前端 + MySQL 持久化 |
| **安全基线** | `.env` 配置隔离、bcrypt 哈希、会话归属校验 |
| **评估闭环** | Hit@K / MRR / 来源命中率，动画片集 Hit@3 **91.7%** |
| **LangGraph 编排** | `AgentState` 状态机：Router → Memory → Retrieval → Fusion → Generation |

---

## LangGraph Agentic RAG

对话链路已从 LangChain `RunnableWithMessageHistory` 链升级为 **LangGraph `StateGraph`**：

```mermaid
flowchart LR
    START([START]) --> R[Router]
    R --> M[Memory Recall]
    M --> RET[Retrieval]
    RET --> RR[Rerank 精排]
    RR --> F[Context Fusion]
    F --> G[Generation]
    G --> END([END])
```

**核心状态**（`app/graph/state.py` → `AgentState`）：

| 字段 | 说明 |
|------|------|
| `query` / `rewritten_query` | 用户问题 / Router 改写后检索问句 |
| `memories` / `memory_block` | 结构化主题记忆 |
| `retrieved_docs` | 混合检索文档块 |
| `final_context` | 记忆 + 检索融合上下文 |
| `answer` | 生成结果 |

**目录**（对应设计中的 `mem-rag/` 模块，实现在 `app/` 包内）：

```plain
app/
├── graph/           # LangGraph 状态机
│   ├── state.py
│   ├── workflow.py
│   └── nodes/       # router / memory / retrieval / rerank / fusion / generation
├── memory/          # 结构化记忆服务
├── retrieval/       # Milvus + BM25 + RRF + Rerank
├── llm/             # DashScope 模型工厂
└── api/             # FastAPI 路由
```

---

## 系统架构

### 总体架构

```mermaid
flowchart TB
    subgraph Client["客户端"]
        WEB["Vue3 前端<br/>index.html"]
    end

    subgraph API["FastAPI 服务层"]
        AUTH["认证<br/>bcrypt + Cookie"]
        RBAC["会话归属校验"]
        CHAT["POST /chat 流式"]
        KB["POST /kb/upload"]
        MEM["GET /sessions/.../memory"]
    end

    subgraph Core["RAG 核心"]
        SM["StructuredMemory<br/>主题记忆构建"]
        VS["VectorStoreService<br/>Milvus + BM25 + RRF"]
        RAG["RagService<br/>LangGraph astream"]
        LLM["通义千问 qwen3-8b"]
    end

    subgraph Ingest["知识库入库"]
        PARSE["DocumentParser<br/>TXT/PDF/DOCX"]
        SPLIT["HybridTextSplitter<br/>semantic/recursive/hybrid"]
        EMB["DashScope Embedding v4"]
    end

    subgraph Store["持久化"]
        MYSQL[("MySQL<br/>用户/会话/消息/主题")]
        MILVUS[("Milvus Lite<br/>向量索引")]
        BM25F[("bm25_corpus.pkl<br/>稀疏语料")]
        MD5F[("md5.text<br/>去重")]
    end

    WEB --> AUTH
    AUTH --> RBAC
    RBAC --> CHAT & KB & MEM
    KB --> PARSE --> SPLIT --> EMB
    EMB --> MILVUS
    SPLIT --> BM25F
    PARSE --> MD5F

    CHAT --> SM
    SM --> MYSQL
    CHAT --> VS
    VS --> MILVUS & BM25F
    VS --> RAG
    SM --> RAG
    RAG --> LLM
    LLM --> CHAT
    MEM --> MYSQL
```

### 对话请求时序

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant F as Vue 前端
    participant A as FastAPI
    participant SM as 结构化记忆
    participant VS as 混合检索
    participant L as 通义千问
    participant DB as MySQL

    U->>F: 输入问题
    F->>A: POST /chat (Cookie + session_uuid)
    A->>A: 校验会话归属 (user_id)
    A->>SM: build_context(session, query)
    SM->>DB: 加载主题 & 历史消息
    SM-->>A: memory_block + 精选 history
    A->>VS: hybrid_search(query)
    VS-->>A: Top-K 文档块 (RRF)
    A->>L: prompt(context + memory + history)
    loop 流式输出
        L-->>A: token chunk
        A-->>F: text/plain stream
        F-->>U: 逐字渲染 + 思考步骤
    end
    A->>DB: 异步 save_chat_history
    A->>SM: 异步 attach_message_to_topic
    F->>A: GET /sessions/.../memory
    A-->>F: 更新「当前主题」标签
```

---

## 结构化记忆

传统 RAG 仅线性压缩历史消息，长对话容易丢失主题上下文。Mem-RAG 引入 **按主题组织的结构化记忆**：

```mermaid
flowchart LR
    subgraph Input
        Q[用户新问题]
    end

    subgraph Classify["主题路由"]
        KW[关键词快匹配]
        LLM_C[LLM JSON 分类]
    end

    subgraph Memory["记忆注入 Prompt"]
        IDX[主题索引摘要]
        SAME[同主题近期轮次 x4]
        OTHER[其他主题概要 x3]
        RECENT[跨主题最近轮次 x2]
    end

    Q --> KW
    KW -->|未命中| LLM_C
    KW --> Memory
    LLM_C --> Memory
    Memory --> PROMPT[RAG System Prompt]
```

| 组件 | 文件 | 说明 |
|------|------|------|
| 主题模型 | `app/models/models.py` | `MemoryTopic` 表 |
| 记忆服务 | `app/core/structured_memory.py` | 分类、摘要、上下文构建 |
| 前端展示 | `html/index.html` | 当前主题标签 + 主题列表面板 |
| API | `GET /sessions/{uuid}/memory` | 返回主题列表与摘要 |

---

## 知识库入库 Pipeline

```mermaid
flowchart LR
    FILE["上传文件<br/>TXT/PDF/DOCX"] --> PARSE["document_parser"]
    PARSE --> DEDUP{"MD5 去重?"}
    DEDUP -->|已存在| SKIP["跳过"]
    DEDUP -->|新内容| SPLIT["HybridTextSplitter"]
    SPLIT --> VEC["embed_documents"]
    VEC --> MV["Milvus insert"]
    SPLIT --> BM["BM25 corpus 追加"]
    MV --> DONE["入库完成"]
    BM --> DONE
```

**分块策略**（`CHUNK_STRATEGY`，默认 `hybrid`）：

| 策略 | 行为 |
|------|------|
| `hybrid` | 语义分块 → 超大块二次切分 → 过小块合并 |
| `semantic` | 纯 SemanticChunker + 大小约束 |
| `recursive` | 递归字符分块（超长文档自动降级） |

---

## 项目结构

```plain
Mem-RAG-main/
├── app/
│   ├── graph/                    # LangGraph 状态机
│   │   ├── state.py              # AgentState
│   │   ├── workflow.py           # StateGraph 编译
│   │   └── nodes/                # router / memory / retrieval / fusion / generation
│   ├── memory/                   # 结构化记忆封装
│   ├── retrieval/                # 混合检索封装
│   ├── llm/                      # DashScope 模型工厂
│   ├── api/                      # FastAPI 路由层
│   │   ├── api_service.py        # 主入口 & 路由
│   │   ├── database.py           # 异步 DB 连接
│   │   ├── deps.py               # get_current_user / get_owned_session
│   │   └── schemas.py            # Pydantic 请求体
│   ├── core/
│   │   ├── rag.py                # LangGraph 流式 RAG 入口
│   │   ├── vector_stores.py      # Milvus + BM25 + RRF
│   │   ├── structured_memory.py  # 结构化主题记忆
│   │   ├── knowledge_base.py     # 知识库入库
│   │   ├── document_parser.py    # 多格式文档解析
│   │   ├── text_splitter.py      # 混合分块
│   │   ├── security.py           # bcrypt 密码
│   │   └── config_data.py        # 读取 .env
│   ├── eval/
│   │   ├── retrieval_eval.py     # 检索评估 CLI
│   │   └── seed_data.py          # 批量导入 data/
│   └── models/models.py          # User / ChatSession / MemoryTopic / ChatMessage
├── eval/
│   ├── retrieval_dataset.json    # 动画片 12 条
│   ├── nfs_dataset.json          # NFS 文档 8 条
│   ├── results.json              # 评估报告（自动生成）
│   └── nfs_results.json
├── data/                         # 示例知识库
├── html/index.html               # Vue3 前端
├── database/                     # 运行时：Milvus / BM25 / MD5
├── .env.example
└── requirements.txt
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排 | **LangGraph** StateGraph, LangChain Core |
| 后端 | FastAPI, SQLAlchemy, aiomysql |
| 检索 | Milvus Lite, rank-bm25, 自研 RRF |
| Embedding | DashScope `text-embedding-v4` |
| LLM | 通义千问 `qwen3-8b`（对话 / 摘要 / 分类，`.env` 可配置） |
| 前端 | Vue 3, Tailwind CSS, Fetch ReadableStream |
| 数据库 | MySQL 8.0+ |
| 安全 | bcrypt, HttpOnly Cookie, 会话归属校验, CORS 白名单 |

---

## 快速开始

### 1. 环境

```bash
conda create -n memrag python=3.10
conda activate memrag
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DATABASE_URL=mysql+aiomysql://root:your_password@localhost:3306/memrag_db
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
COOKIE_SECURE=false
COOKIE_SAMESITE=lax
```

创建数据库：

```sql
CREATE DATABASE memrag_db CHARACTER SET utf8mb4;
```

### 3. 启动

```bash
python -m app.api.api_service
```

- 前端：http://localhost:8000/html/index.html
- API 文档：http://localhost:8000/docs

可选 Streamlit 上传工具：

```bash
streamlit run ./app/core/app_file_uploder.py
```

---

## 使用说明

1. **注册 / 登录**：密码通过 JSON Body 提交
2. **上传知识库**：侧边栏绿色上传按钮（TXT / PDF / DOCX）
3. **新建对话**：点击 `+` 创建会话
4. **提问**：Enter 发送，流式输出 + 思考步骤可视化
5. **当前主题**：聊天区顶部橙色标签，点击展开主题记忆面板

---

## 检索评估

### 评估目标

评估脚本 `app/eval/retrieval_eval.py` 针对 **检索阶段**（不含 LLM 生成），在标注问答集上量化混合检索质量。

### 评估指标

| 指标 | 含义 | 越高越好 |
|------|------|----------|
| **Hit@K (任一)** | Top-K 中至少命中一个期望关键词 | ✓ |
| **Hit@K (全部)** | Top-K 合并文本包含全部期望关键词 | ✓ |
| **MRR** | 第一个命中结果的倒数排名均值 | ✓ |
| **关键词召回率** | 期望关键词在 Top-K 中的平均命中比例 | ✓ |
| **来源 Hit@K** | 是否检索到期望来源文件 | ✓ |

### 评估流程

```mermaid
flowchart TD
    A["准备知识库"] --> B{"数据集匹配?"}
    B -->|动画片集| C["--import-data 导入 data/*.txt"]
    B -->|NFS 文档| D["前端/API 上传 PDF"]
    C --> E["停止 API 服务"]
    D --> E
    E --> F["python -m app.eval.retrieval_eval"]
    F --> G["控制台报告 + eval/results.json"]
```

> **注意**：Milvus Lite 同一时间只允许一个进程访问 `database/milvus_db.db`，运行评估前需停止 API 服务。

### 运行命令

```bash
# 导入 data/ 并评估动画片数据集（Top-K=3）
python -m app.eval.retrieval_eval --import-data

# 指定 K 值并保存 JSON 报告
python -m app.eval.retrieval_eval --k 5 --output eval/results.json

# 评估 NFS 文档（需先上传 day29-第三章NFS.pdf）
python -m app.eval.retrieval_eval --dataset eval/nfs_dataset.json --output eval/nfs_results.json

# 仅批量导入示例数据
python -m app.eval.seed_data
```

### 消融实验（Ablation Study）

对比不同检索策略的 Hit@K / MRR，验证 Rerank 等环节的实际增益：

```bash
# 停止 API 后运行全部实验 A0–A4
python -m app.eval.ablation_eval --import-data

# 保存报告
python -m app.eval.ablation_eval --k 3 --output eval/ablation_results.json

# 只跑混合宽召回 vs Rerank
python -m app.eval.ablation_eval --only A2_hybrid_wide,A3_hybrid_rerank
```

| 实验 ID | 配置 |
|---------|------|
| A0_dense | 仅 Milvus 向量 Top-K |
| A1_hybrid_legacy | RRF + 97% 阈值（旧版） |
| A2_hybrid_wide | 仅消融：宽召回直接截 Top-K（**生产已禁用**） |
| A3_hybrid_rerank | **生产默认**：宽召回 + DashScope Rerank 精排 |
| A4_dense_rerank | 向量宽召回 + Rerank |

`.env` 可调：`RETRIEVAL_RECALL_K`、`RERANK_ENABLED`、`RERANK_MODEL`（如 `gte-rerank-v2`）、`RERANK_TOP_N`。

### 评估数据集

| 文件 | 场景 | 用例数 | 知识库前置条件 |
|------|------|--------|----------------|
| `eval/retrieval_dataset.json` | 海绵宝宝 / 宝可梦 / 猫和老鼠等 | 12 | `data/` 下 4 个 txt |
| `eval/nfs_dataset.json` | NFS 共享存储实战 | 8 | 对应 NFS PDF 已上传 |

单条用例格式：

```json
{
  "id": "sponge-001",
  "query": "海绵宝宝住在哪里？",
  "expected_keywords": ["菠萝", "比基尼海滩"],
  "expected_source": "海绵宝宝.txt"
}
```

### 实测结果

**动画片数据集**（12 条，Top-K=3，2026-05-28）

| 指标 | 结果 |
|------|------|
| Hit@3 (任一关键词) | **91.7%** (11/12) |
| Hit@3 (全部关键词) | **91.7%** |
| MRR | **0.917** |
| 平均关键词召回率 | **91.7%** |
| 来源 Hit@3 | **83.3%** |

未通过用例：`pokemon-002`（「皮卡丘配音演员」— 关键词「大谷育江」未出现在 Top-3）

**NFS 数据集**（8 条，Top-K=3，2026-05-28）

| 指标 | 结果 |
|------|------|
| Hit@3 (任一关键词) | **100%** (8/8) |
| Hit@3 (全部关键词) | **100%** |
| MRR | **0.833** |
| 平均关键词召回率 | **100%** |
| 来源 Hit@3 | **100%** |

完整逐条结果见 `eval/results.json` 与 `eval/nfs_results.json`。

### 常见误区

| 现象 | 原因 |
|------|------|
| Hit@K = 0% | 知识库内容与评估集不匹配（如只有 NFS PDF 却跑动画片数据集） |
| Milvus 连接失败 | API 与评估脚本同时运行导致文件锁 |
| 来源 Hit 低 | RRF 返回跨文件块，关键词命中但 filename 不一致 |

---

## API 接口

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| POST | `/auth/register` | 注册 | 否 |
| POST | `/auth/login` | 登录 | 否 |
| POST | `/sessions` | 创建会话 | Cookie |
| GET | `/sessions` | 会话列表 | Cookie |
| GET | `/sessions/{uuid}/memory` | 结构化主题记忆 | Cookie + 归属 |
| GET | `/chat/{uuid}` | 聊天历史 | Cookie + 归属 |
| POST | `/chat` | 流式对话 | Cookie + 归属 |
| DELETE | `/delete/{uuid}` | 删除会话 | Cookie + 归属 |
| GET | `/kb/formats` | 支持格式 | Cookie |
| POST | `/kb/upload` | 上传知识库 | Cookie |

**登录**

```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"123456"}' \
  -c cookies.txt
```

**流式对话**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"session_uuid":"your-uuid","input_text":"海绵宝宝住在哪里？"}' \
  --no-buffer
```

---

## 安全说明

- 敏感配置通过 `.env` 管理，**切勿提交 `.env`**
- 密码 **bcrypt** 存储，旧 SHA256 用户登录后自动升级
- 会话接口校验 **user_id 归属**，防止 UUID 越权
- Cookie：`HttpOnly`；本地 `COOKIE_SECURE=false`，生产环境 HTTPS 下设为 `true`
- CORS 白名单：`CORS_ORIGINS` 环境变量配置

---

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| Milvus 锁冲突 | 停止 API 或评估脚本，二者不可同时访问 `milvus_db.db` |
| 端口 8000 占用 | `netstat -ano \| findstr :8000` → `taskkill /PID xxx /F` |
| Cookie 登录失败 | 确认 `COOKIE_SECURE=false`，清除浏览器 Cookie 后重登 |
| 评估 0% | 确认知识库与数据集匹配，或先 `--import-data` |
| `DASHSCOPE_API_KEY` 未设置 | 检查 `.env` 文件是否在项目根目录 |

---

## 日志

- 控制台：INFO+
- 文件：`logs/rag_system.log`（DEBUG，10MB 轮转，保留 5 份）

---

## 许可证

MIT License
