from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# 定义全局变量（为了在工具中传递状态）
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 更新工具
@tool
def update(content: str) -> str:   # content 参数将由后台的 LLM 提供
    """Updates the document with the provided content."""    # 文档字符串，用提供的内容更新文档
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

# 保存工具
@tool
def save(filename: str) -> str:   # 文件名也由 LLM 提供
    """Save the current document to a text file and finish the process.
    Args:
        filename: Name for the text file.   # 强调文本文件，这样 LLM 就知道它需要传递的文件名末尾必须有 .txt
    """
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:    # 将保存在全局变量中的内容，以文本文件的形式，在指定文件名之下保存
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:    # 异常处理
        return f"Error saving document: {str(e)}"

# 工具列表
tools = [update, save]

# 调用模型 + 绑定工具
model = ChatOpenAI(model="mimo-v2.5-pro").bind_tools(tools)

# 初始化智能体（智能体是图中的一个节点）
def our_agent(state: AgentState) -> AgentState:
    # 系统消息
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:    # 第一次使用
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:    # 更新状态
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}   # 返回更新后的状态

# 条件边函数，条件边是从 tools 节点出来的，要么指向 Agent，要么结束流程
def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and    # 如果是更新工具，肯定得走 continue；如果是保存工具，直接走 END
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"   # goes to the end edge which leads to the endpoint
    return "continue"

# 为了让打印的消息在终端上格式更易读
def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)

# 有智能体节点和工具节点
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")    # 起点
graph.add_edge("agent", "tools")   # 智能体要调用工具，有向边

# 条件边
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,    # 终点
    },
)

app = graph.compile()

# 调用图
def run_document_agent():
    print("\n ===== DRAFTER =====")
    state = {"messages": []}    # 从空列表初始化，也可以传入一些现有的内容（邮件或文档内容）
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()