# 导入部分新增了 AI消息 和 Union类型注解
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]   # 允许在这个状态键中存储人类消息或AI消息
    # 或写成
    # messages: List[HumanMessage]
    # messages_ai: List[AIMessage]

llm = ChatOpenAI(model="mimo-v2.5-pro")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    # response.content 提取响应的内容部分，也就是LLM返回的答案或结果
    state["messages"].append(AIMessage(content=response.content))    # 添加AI消息
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])   # 当前状态的一个快照
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

conversation_history = []   # 初始化对话历史

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))   # 对话历史记录中添加了人类消息，即用户输入的内容
    result = agent.invoke({"messages": conversation_history})   # 编译后的图，包含了整个对话历史记录
    conversation_history = result["messages"]
    user_input = input("Enter: ")

# 文本文件存储
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    # 对话历史存储了 AI消息 和 人类消息（所有图外信息），状态被锁定在图内，对话历史是状态的一个副本
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")