from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader   # 加载 PDF 文件
from langchain_text_splitters import RecursiveCharacterTextSplitter   # langchain.text_splitter 在新版 langchain 中已迁移到独立包 langchain_text_splitters，用于切分 chunk
from langchain_chroma import Chroma   # 向量数据库，存储向量嵌入
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="mimo-v2.5-pro", temperature = 0)    # I want to minimize hallucination - temperature = 0 makes the model output more deterministic 趋于0 更确定；趋于1 更随机

# Our Embedding Model - 使用本地 HuggingFace 模型，避免依赖 OpenAI embeddings API
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

pdf_path = "Stock_Market_Performance_2024.pdf"   # 2024年股票市场表现

# Safety measure I have put for debugging purposes :)
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path) # This loads the PDF

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")   # 9页
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

# Chunking Process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 将文本分块过程应用到文档的所有页面
pages_split = text_splitter.split_documents(pages) # We now apply this to our pages

# 向量数据库所在位置
persist_directory = r"C:\Users\11842\Desktop\Agent-About\LangGraph-Course\Agents"
collection_name = "stock_market_hf"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# 创建向量嵌入（chroma向量数据库）
try:
    # Here, we actually create the chroma database using our embeddigns model
    vectorstore = Chroma.from_documents(
        documents=pages_split,   # 页面如何分割
        embedding=embeddings,    # 使用哪种嵌入
        persist_directory=persist_directory,   # 存储在哪里
        collection_name=collection_name    # 集合名称
    )
    print(f"Created ChromaDB vector store!")
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Now we create our retriever 
retriever = vectorstore.as_retriever(
    search_type="similarity",   # 默认设置
    search_kwargs={"k": 5} # K is the amount of chunks to return（默认值为4）
)

# 接收查询并输出一个字符串
@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")    # 存储所有找到的相似块
    return "\n\n".join(results)

tools = [retriever_tool]
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 条件边函数，检查最后一条消息是否包含任何工具调用
def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

# 系统提示词
# “请始终引用你在答案中使用的文档的具体部分”，为了确保它不会产生幻觉
system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools} # Creating a dictionary of our tools

# LLM Agent，调用LLM和当前状态
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages': [message]}

# Retriever Agent
# 如果有一个工具，且其名称是一个正确指定的工具，就会执行相关操作
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        # 检查 LLM 选择的工具是否有效
        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

graph = StateGraph(AgentState)

# 把两个AI智能体作为节点，并添加它们各自的动作
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

# 条件边，从LLM节点开始
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# 允许我们不断向 Graph 提问并接收答案
def running_agent():
    print("\n=== RAG AGENT===")
    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)] # converts back to a HumanMessage type
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)

running_agent()