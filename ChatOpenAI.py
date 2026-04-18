import openai
from openai import OpenAI
import dotenv
import os
from log import log
import asyncio

dotenv.load_dotenv()
# https://platform.openai.com/docs/api-reference/chat

class ChatOpenAI():
    def __init__(self, model:str,system_prompt: str="",tools = [],context: str=""):
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("BASE_URL")
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.context = context
        self.llm = openai.OpenAI(api_key=api_key, base_url = base_url)
        self.message = []
    async def chat(self, prompt = None):
        log(f"CHAT")
        if(prompt):
            self.message.append({"role":"user","content": prompt})

            stream = self.llm.chat.completions.create(
                model = self.model,
                messages = self.message,
                tools = self.getToolsDefinition(),
                stream = True
            )

            content = ""
            toolCalls = []
            log(f"RESPONSE")
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
                if delta.tool_calls:
                    # Chunk 工具调用情况
                    for toolCallChunk in delta.tool_calls:
                        if len(toolCalls) <= toolCallChunk.index:
                            toolCalls.append({"id":"","function": {"name":"","arguments":""}})
                        currentCall = toolCalls[toolCallChunk.index]
                        if(toolCallChunk.id):
                            currentCall["id"] +=toolCallChunk.id
                        if(toolCallChunk.function.name):
                            currentCall["function"]["name"] +=toolCallChunk.function.name
                        if(toolCallChunk.function.arguments):
                            currentCall["function"]["arguments"] +=toolCallChunk.function.arguments
            self.message.append(
                {
                    "role":"assistant",
                    "content":content,
                    "tool_calls": [{
                        "id":call["id"],
                        "type":"function",
                        "function":call["function"]
                    } for call in toolCalls] if toolCalls else None
                })
            return {
                "content":content,
                "toolCalls":toolCalls
            }   

    def appendToolResult(self,toolCallId:str,toolOutput:str):
        self.message.append({
            "role":"tool",
            "content":toolOutput,
            "tool_call_id":toolCallId
        })

    def getToolsDefinition(self):
        return [
            {
                "type":"function",
                "function": {
                    "name":tool['name'],
                    "description": tool['description'],
                    "parameters": tool['inputSchema']
                }
            } for tool in self.tools
        ]
if __name__ == "__main__":
    
    prompt = '你是什么模型'
    llm = ChatOpenAI('z-ai/glm-4.7-flash')
    res = asyncio.run(llm.chat(prompt=prompt)) 
    print(res)