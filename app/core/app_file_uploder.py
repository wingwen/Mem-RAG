"""
基于 Streamlit 的知识库上传服务，支持 TXT / PDF / Word(docx)。
"""
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from app.core.knowledge_base import KnowledgeBaseService

st.title("知识库更新服务")
st.caption("支持格式：TXT、PDF、Word (.docx)")

if 'service' not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

supported = st.session_state["service"].supported_formats()
upload_file = st.file_uploader(
    "请上传文档",
    type=supported,
    accept_multiple_files=False,
    help="支持 UTF-8/GBK 文本、PDF 文字版、Word 2007+ (.docx)"
)

if upload_file:
    file_name = upload_file.name
    file_size = upload_file.size / 1024
    file_type = upload_file.type
    st.subheader(f"文件名: {file_name}")
    st.write(f"格式 {file_type} ｜ 大小: {file_size:.2f} KB")

    with st.spinner("正在解析文档并载入知识库..."):
        time.sleep(0.5)
        result = st.session_state["service"].upload_by_bytes(upload_file.getvalue(), file_name)
        if result.startswith("【成功】"):
            st.success(result)
        elif result.startswith("【跳过】"):
            st.warning(result)
        else:
            st.error(result)
