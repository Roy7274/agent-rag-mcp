from MCPClient import MCPClient
from agent import agent
import os
from log import log
from embeddingRetriver import EmbeddingRetriever


os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890" 

emb_model = os.getenv("EMBEDDING_MODEL")
current_dir = os.getcwd()
fetchMCP = MCPClient("fetch", command="uvx", args=['mcp-server-fetch'])
fileMCP = MCPClient("file", command="npx", args=['-y', '@modelcontextprotocol/server-filesystem', current_dir])

print(f'current_dir is {current_dir}')
TASK = f"""
        告诉我 Chelsey Dietrich 的信息,先从我给你的文件中中找到相关信息,信息在 knowledge 目录下,总结后创作一个关于她的故事
        把故事和她的基本信息保存到{current_dir}/antonette.md,输出一个漂亮md文件
        """
async def main():
    # 简化流程，不再使用多层异常处理
    context = await retriveContext()
    log("init Agent")
    agent_instance = agent("z-ai/glm-4.5-air:free", [fetchMCP, fileMCP], context=context)
    await agent_instance.init()
    log("init Agent finish")
    response = await agent_instance.invoke(TASK)
    return response

async def retriveContext():
    # RAG
    embeddingRetriever = EmbeddingRetriever(model = emb_model)
    knowledgeDir = os.path.join(current_dir, 'knowledge');
    for file in os.listdir(knowledgeDir):
        with open(os.path.join(knowledgeDir, file), 'r', encoding='utf-8') as f:
            text = f.read()
            await embeddingRetriever.embedDocument(text)
            
    context = await embeddingRetriever.retrive(TASK, 3)
    log(context)
    return context
if __name__ == "__main__":
    import asyncio
    # 简单地运行主函数
    result = asyncio.run(main())
    print(result) 