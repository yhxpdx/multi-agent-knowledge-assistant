"""
Streamlit 前端 — 多智能体知识助手

功能：
1. 聊天界面：用户输入问题，流式展示 Agent 回答
2. 文档管理：上传、查看、删除文档
3. 会话管理：切换、新建会话
4. Agent 状态：展示路由决策和工具调用
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="多智能体知识助手",
    page_icon="🤖",
    layout="wide",
)

# 初始化 session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def create_session():
    """创建新会话"""
    resp = requests.post(f"{API_BASE}/api/sessions")
    if resp.status_code == 200:
        st.session_state.session_id = resp.json()["session_id"]
        st.session_state.messages = []


def send_message(message: str):
    """发送消息并获取流式响应"""
    st.session_state.messages.append({"role": "user", "content": message})

    # 流式请求
    full_response = ""
    agent_used = ""

    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            resp = requests.post(
                f"{API_BASE}/api/chat",
                json={
                    "message": message,
                    "session_id": st.session_state.session_id,
                    "stream": True,
                },
                stream=True,
            )

            for line in resp.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("done"):
                            st.session_state.session_id = data.get("session_id")
                        elif data.get("error"):
                            st.error(data["error"])
                        else:
                            content = data.get("content", "")
                            node = data.get("node", "")
                            if node and node != agent_used:
                                agent_used = node
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            st.error(f"请求失败: {e}")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "agent": agent_used,
    })


# 侧边栏
with st.sidebar:
    st.title("🤖 多智能体知识助手")

    # 会话管理
    st.subheader("会话管理")
    if st.button("新建对话", use_container_width=True):
        create_session()
        st.rerun()

    if st.session_state.session_id:
        st.info(f"当前会话: {st.session_state.session_id}")

    # 文档管理
    st.subheader("文档管理")
    uploaded_file = st.file_uploader("上传文档", type=["pdf", "txt", "md", "docx"])
    if uploaded_file:
        if st.button("上传并处理"):
            with st.spinner("处理中..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{API_BASE}/api/documents", files=files)
                if resp.status_code == 200:
                    st.success(f"上传成功！{resp.json()['chunk_count']} 个 chunks")
                else:
                    st.error(f"上传失败: {resp.json().get('detail', '未知错误')}")

    # 文档列表
    try:
        resp = requests.get(f"{API_BASE}/api/documents")
        if resp.status_code == 200:
            docs = resp.json()
            if docs:
                st.write(f"已上传 {len(docs)} 个文档:")
                for doc in docs:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"📄 {doc['filename']}")
                    if col2.button("🗑", key=f"del_{doc['doc_id']}"):
                        requests.delete(f"{API_BASE}/api/documents/{doc['doc_id']}")
                        st.rerun()
    except:
        pass

    # 记忆管理
    st.subheader("记忆管理")
    try:
        category_filter = st.selectbox(
            "按分类筛选",
            ["全部", "preference", "fact", "conclusion", "instruction"],
            key="memory_category",
        )
        params = {}
        if category_filter != "全部":
            params["category"] = category_filter
        resp = requests.get(f"{API_BASE}/api/memories", params=params)
        if resp.status_code == 200:
            memories = resp.json()
            if memories:
                st.write(f"共 {len(memories)} 条记忆:")
                for mem in memories:
                    col1, col2 = st.columns([4, 1])
                    category = mem.get("category", "")
                    content = mem.get("content", "")
                    col1.caption(f"[{category}]")
                    col1.write(content)
                    if col2.button("🗑", key=f"del_mem_{mem.get('id', '')}"):
                        requests.delete(f"{API_BASE}/api/memories/{mem.get('id', '')}")
                        st.rerun()
            else:
                st.info("暂无记忆")
    except Exception:
        st.info("记忆服务未就绪")

    # 服务状态
    st.subheader("服务状态")
    try:
        resp = requests.get(f"{API_BASE}/api/health")
        if resp.status_code == 200:
            health = resp.json()
            services = health.get("services", {})
            for name, status in services.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {name}")
    except:
        st.error("后端服务未启动")

# 主聊天区域
st.title("💬 对话")

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("agent"):
            st.caption(f"Agent: {msg['agent']}")

# 用户输入
if prompt := st.chat_input("输入你的问题..."):
    if not st.session_state.session_id:
        create_session()
    send_message(prompt)
