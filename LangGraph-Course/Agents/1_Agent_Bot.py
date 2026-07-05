from typing import TypedDict, List
from langchain_core.messages import HumanMessage   # 使用HumanMessage
from langchain_openai import ChatOpenAI   # 使用LLM
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv    # 用来存储敏感信息（如API密钥或配置值）的文件，主要是出于安全考虑

load_dotenv()   # 加载 env 文件

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="mimo-v2.5-pro")

# 节点函数，传入状态并返回状态
def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
# 多轮对话
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
