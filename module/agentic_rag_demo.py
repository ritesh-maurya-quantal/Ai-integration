import json
import uuid
import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ─────────── Secrets from environment (DEV ONLY – remove before publishing) ────
OPENAI_API_KEY   = os.getenv('OPEN_API_KEY')
NOTION_MCP_TOKEN = os.getenv('NOTION_MCP_TOKEN')
NOTION_VERSION   = os.getend('NOTION_VERSION')  

# ───────────────────────── build LangGraph app once ───────────────────────────
async def build_app():
    notion_cfg = {
        "notion": {
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "transport": "stdio",
            "env": {
                "OPENAPI_MCP_HEADERS": json.dumps(
                    {
                        "Authorization": f"Bearer {NOTION_MCP_TOKEN}",
                        "Notion-Version": NOTION_VERSION,
                    }
                )
            },
        }
    }
    client = MultiServerMCPClient(notion_cfg)
    notion_tools = await client.get_tools()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
    ).bind_tools(notion_tools)

    async def agent_node(state: MessagesState):
        msgs = state["messages"]
        ai_msg = await llm.ainvoke(msgs)
        return {"messages": msgs + [ai_msg]}

    tool_node = ToolNode(notion_tools)

    wf = StateGraph(MessagesState)
    wf.add_node("agent", agent_node)
    wf.add_node("tools", tool_node)
    wf.add_edge(START, "agent")

    def need_tool(state: MessagesState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    wf.add_conditional_edges("agent", need_tool, {"tools": "tools", END: END})
    wf.add_edge("tools", "agent")

    return wf.compile(checkpointer=MemorySaver())

APP = asyncio.run(build_app())

# ──────────────────────────────── Chat handlers ───────────────────────────────
async def respond(message: str, thread_id: str):
    """
    Return assistant response string for the given user message.
    LangGraph memory is handled via thread_id + MemorySaver.
    """
    if not message:
        return "Empty message."

    input_messages = [HumanMessage(content=message)]
    result = await APP.ainvoke(
        {"messages": input_messages},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Extract last AI message
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            return m.content

    return "No response."

# # ───────────────────────────── Simple CLI interface ───────────────────────────
# async def chat_cli():
#     # print("🔗 Notion MCP Agent (CLI). Type 'exit' to quit.\n")
#     # thread_id = str(uuid.uuid4())
#     while True:
#         try:
#             user_in = input("You: ").strip()
#         except (EOFError, KeyboardInterrupt):
#             break
#         if user_in.lower() in {"exit", "quit"}:
#             break
#         reply = await respond(user_in, thread_id)
#         print(f"Assistant: {reply}\n")

async def run_chat(user_in, thread_id):
    reply = await respond(user_in, thread_id)
    return reply
    
if __name__ == "__main__":
    asyncio.run(chat_cli())
