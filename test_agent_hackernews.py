import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import agent
from MCPClient import MCPClient


async def test_hackernews_to_md():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_clients = []
    
    fetchMCP = MCPClient(
        name="fetch",
        command="uvx",
        args=["mcp-server-fetch"]
    )
    mcp_clients.append(fetchMCP)
    
    fileMCP = MCPClient(
        name="file",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", current_dir]
    )
    mcp_clients.append(fileMCP)
    
    # 系统提示词
    sys_prompt = """你是一个助手，可以使用工具来完成任务。
    available tools: fetch, file
    
    任务：
    1. 使用fetch工具从https://news.ycombinator.com/抓取最新新闻
    2. 提取新闻标题和链接
    3. 使用file工具将结果保存为markdown文件到当前目录，文件名为hackernews.md
    """
    
    # 创建Agent
    llm_model = os.getenv("MODEL", "z-ai/glm-4.7-flash")  # 使用环境变量或默认模型
    agent_instance = agent(
        model=llm_model,
        mcpClients=mcp_clients,
        sysprompt=sys_prompt
    )
    
    try:
        # 初始化Agent
        await agent_instance.init()
        
        # 执行任务
        prompt = """请帮我完成以下任务：
        1. 从 https://news.ycombinator.com/ 获取最新10条新闻
        2. 提取每条新闻的标题和链接
        3. 将结果保存为 hackernews.md 文件
        
        格式要求：
        # HackerNews 最新新闻
        
        ## 新闻列表
        
        1. [标题](链接)
        2. [标题](链接)
        ...
        """
        
        result = await agent_instance.invoke(prompt)
        print("\n=== 执行结果 ===")
        print(result)
        
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:

        await agent_instance.close()


async def test_simple():

    # 测试单个MCP客户端
    client = MCPClient(
        name="test",
        args=["python", "-c", "print('test')"],
        command="python"
    )
    
    try:
        await client.init()
        tools = client.get_tools()
        print(f"可用工具: {tools}")
    except Exception as e:
        print(f"连接失败: {e}")
    finally:
        await client.close()


if __name__ == "__main__":

    asyncio.run(test_hackernews_to_md())