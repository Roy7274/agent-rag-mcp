from ChatOpenAI import ChatOpenAI
from log import log
import json

class agent():
    def __init__(self,model,mcpClients,sysprompt="",context="") -> None:
        self.mcpClients = mcpClients  
        self.model = model
        self.sys_prompt = sysprompt
        self.context = context
        self.llm = None

    async def init(self):
        log("TOOLS")
        for mcp in self.mcpClients:
            await mcp.init()

        all_tools = []
        for client in self.mcpClients:
            all_tools.extend(client.get_tools())
        self.llm = ChatOpenAI(self.model, system_prompt=self.sys_prompt,tools=all_tools, context=self.context)

    async def close(self):
        for mcp in self.mcpClients:
            try:
                await mcp.close()
            except Exception as e:
                print(f"Error closing MCP Client: {e}")

    async def invoke(self, prompt: str):
        if not self.llm:
            raise Exception("Agent not ready")
        
        response = await self.llm.chat(prompt=prompt)
        while True:
            tool_calls = response.get("toolCalls") or []
            if len(tool_calls) > 0:
                for tool_call in tool_calls:
                    mcp = next(
                        (client for client in self.mcpClients if any(
                            t["name"] == tool_call["function"]["name"] for t in client.get_tools()
                        )),None
                    )

                    if mcp:
                        log("TOOL USE")
                        print(f"Calling tool: {tool_call['function']['name']}")
                        print(f"Arguments: {tool_call['function']['arguments']}")

                        result = await mcp.call_tool(
                            tool_call['function']['name'], 
                            json.loads(tool_call['function']['arguments'])
                        )
                        
                        result_str = ""
                        if hasattr(result, 'content') and result.content:
                            result_dict = {
                                "content": result.content[0].text if result.content else "",
                                "isError": getattr(result, 'isError', False)
                            }
                            result_str = json.dumps(result_dict)
                        else:
                            result_str = str(result)
                            
                        print(f"Result: {result_str}")
                        self.llm.appendToolResult(tool_call['id'], result_str)
                    else: 
                        self.llm.appendToolResult(tool_call['id'], 'Tool not found')

                response = await self.llm.chat()
                continue
            #工具调用完成
            await self.close()
            return response['content']