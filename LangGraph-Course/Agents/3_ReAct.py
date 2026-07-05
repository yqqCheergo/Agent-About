from typing import Annotated    # Annotated: provides additional context without affecting the type itself
from typing import Sequence     # Sequence: to automatically handle the state updates for sequences such as by adding new messages to a chat history
from typing import TypedDict
from dotenv import load_dotenv   # 存储 API 密钥
from langchain_core.messages import BaseMessage       # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage       # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage     # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages    # add_messages 是一个reducer函数（通过追加而不是覆盖来保留状态）
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode    # 工具节点，会将工具的输出连接回状态（State），以便其他节点可以使用这些信息

load_dotenv()

# 定义智能体的状态
class AgentState(TypedDict):
    # 消息序列（Sequence）是数据类型。这里提供了元数据（add_messages），因此使用 Annotated 关键字
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 创建工具。使用装饰器
@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""    # 函数必须要有文档字符串，如果没有会报错。文档字符串的作用是告诉LLM这个工具是用来干什么的
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

# 将工具融入LLM
tools = [add, subtract, multiply]

model = ChatOpenAI(model = "mimo-v2.5-pro").bind_tools(tools)   # 此时 LLM 可以访问所有的工具了

# 创建节点
def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."    # 系统消息
    )
    response = model.invoke([system_prompt] + state["messages"])   # 调用模型，传入系统消息 + 查询（人类消息的形式）
    return {"messages": [response]}   # 更新状态

# 条件边
def should_continue(state: AgentState):    # 传入状态
    messages = state["messages"]
    last_message = messages[-1]    # 获取最后一条消息，看看是否还需要运行更多工具
    if not last_message.tool_calls: 
        return "end"    # 如果没有更多工具调用了，就结束
    else:
        return "continue"   # 否则就去到工具节点，选择工具并执行所有操作

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

# 工具节点本质上就是一个单独的节点，它包含所有不同的工具
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")   # 设置入口点

# 添加条件边
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")   # 这就是创建循环连接的方式（条件边只提供了一个单向的有向边，从Agent到工具节点，或从Agent到终点），因此还需要另一条边（从工具节点到Agent）

app = graph.compile()

# 辅助函数（并非 LangGraph 的一部分），可以让每个工具调用等操作以更好的方式输出
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# 流式处理
# inputs = {"messages": [("user", "Add 3 + 4.")]}
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))