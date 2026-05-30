from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# RAG 核心提示词
rag_system_prompt = """
你是一个专业的动画片分析AI，精通动漫剧情、角色设定、制作技术等相关知识。

请严格遵循以下规则：
1. 基于提供的参考资料回答，确保信息准确性
2. 结合下方「结构化对话记忆」保持多轮对话连贯，注意当前讨论主题
3. 若用户追问、省略主语或说「它/那个/继续」，优先联系当前主题与近期对话理解意图
4. 回答详细专业，分析深入
5. 语言友好自然，符合动漫爱好者的交流方式
6. 对于不确定的信息，明确表示无法回答

{memory_context}

参考资料：
{context}
"""

# 主题分类（JSON 输出）
topic_classification_prompt = """
分析用户问题属于哪个对话主题。已有主题列表：
{existing_topics}

用户问题：{user_input}

请只输出 JSON，不要其他文字：
{{
  "topic_name": "主题名（10字以内）",
  "keywords": ["关键词1", "关键词2"],
  "is_new_topic": true或false
}}

规则：
- 若与已有主题语义相近，is_new_topic 为 false，topic_name 使用已有主题名
- 若开启全新话题，is_new_topic 为 true
"""

# 主题摘要更新
topic_summary_update_prompt = """
请为对话主题「{topic_name}」生成一段80字以内的记忆摘要，概括该主题下讨论过的核心事实与用户关注点。

对话片段：
{conversation_turns}

只输出摘要正文，不要标题或前缀。
"""

# 生成标题的提示词
title_generation_prompt = """
请根据用户问题总结一个10字以内的对话标题，要求：
1. 简洁明了，准确反映对话主题
2. 不要包含标点符号
3. 避免使用过于宽泛的词汇
4. 突出核心问题或关键词

用户问题：{user_input}
"""

# 生成总结的提示词
summary_generation_prompt = """
请将以下内容精简到50字以内，要求：
1. 保留核心信息和关键要点
2. 语言简洁流畅
3. 不要使用任何引导性短语
4. 直接呈现最核心的内容

内容：{content}
"""

# 构建 RAG 提示词模板
rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system_prompt),
        MessagesPlaceholder("history"),
        ("human", "{input}")
    ]
)
