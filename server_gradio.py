# server_gradio.py
import os
import gradio as gr

# 让 LangGraph 的打印更干净（可选）
os.environ.setdefault("TERMINAL_SIMPLE", "1")

from langchain_core.messages import HumanMessage, AIMessage
from LangGraph import build_agent  # 直接用你现有的 build_agent()

# 预加载一次 Agent（避免每次请求重复构建）
AGENT = build_agent()

def run_agent(user_text: str, history):
    """
    Gradio 的聊天回调：
    - 输入: 用户文本 + 历史（不强依赖历史，因为你的图自带上下文拼装）
    - 输出: 模型返回文本
    """
    if not user_text or not user_text.strip():
        return "⚠️ 请输入内容"

    # 送入一轮对话（你的图是从 profile 节点开始，会自动拉齐上下文）
    final_state = AGENT.invoke({"messages": [HumanMessage(content=user_text.strip())]})
    # final_state 是一个 dict，包含 "messages" 列表，里面夹杂了 System/Human/AI
    msgs = final_state.get("messages", [])
    # 收集本轮产生的所有 AIMessage（尤其是 synthesize + teach）
    ai_chunks = []
    for m in msgs:
        if isinstance(m, AIMessage):
            txt = (m.content or "").strip()
            if txt:
                ai_chunks.append(txt)

    if not ai_chunks:
        return "（本轮没有生成可显示的回答）"

    # 通常最后一个就是教学卡片；我们把所有 AI 段落合并，保证你“教学段落”也在
    output = "\n\n".join(ai_chunks)
    return output

# 简洁优雅的 UI
with gr.Blocks(title="LangGraph 投资助手（含教学RAG）") as demo:
    gr.Markdown(
        """
        # LangGraph 投资助手（含教学 RAG）
        直接对话，例如：`I would like to buy some technology stocks`  
        **结尾会自动附上 📚 教学卡片（来源：Investopedia / CFA / Morningstar）**
        """
    )

    chatbot = gr.Chatbot(label="对话区", height=520)
    msg = gr.Textbox(placeholder="请输入问题...", label="输入")

    # 用户提交消息时触发的函数
    def user_submit(user_text, history):
        bot_reply = run_agent(user_text, history)
        history = history + [(user_text, bot_reply)]
        return "", history

    # 绑定回车提交事件
    msg.submit(user_submit, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)