from openai import OpenAI
API_BASE_URL = "http://localhost:3000/v1"  # 修改为你的New API服务地址
API_KEY = "sk-8ZPOgmEUD7fZYBsQoTPcGm3CPDBZ8Hszop87ryzY998E2uAi"  # 请填入你在New API控制台创建的API密钥

class LLMClient:
    def __init__(self, api_key=API_KEY, base_url=API_BASE_URL):
        """初始化LLM客户端"""
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 模型映射表，将游戏中使用的模型名映射到New API支持的模型名
        self.model_mapping = {
            "deepseek-r1": "deepseek-chat",  # 根据New API支持情况调整
            "o3-mini": "o3-mini",  # OpenAI模型
            "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",  # Claude模型
            "gemini-2.0-flash-thinking": "gemini-2.0-flash-thinking-exp"  # Gemini模型
        }
        
    def chat(self, messages, model="o3-mini"):
        """与LLM交互
        
        Args:
            messages: 消息列表
            model: 使用的LLM模型
        
        Returns:
            tuple: (content, reasoning_content)
        """
        try:
            # 将游戏中的模型名转换为API支持的模型名
            api_model = self.model_mapping.get(model, model)
            
            print(f"LLM请求: {messages}")
            print(f"使用模型: {model} -> {api_model}")
            
            response = self.client.chat.completions.create(
                model=api_model,
                messages=messages,
            )
            if response.choices:
                message = response.choices[0].message
                content = message.content if message.content else ""
                reasoning_content = getattr(message, "reasoning_content", "")
                print(f"LLM响应: {content}")
                return content, reasoning_content
            
            return "", ""
                
        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            return "", ""

# 使用示例
if __name__ == "__main__":
    llm = LLMClient()
    
    # 测试各个模型
    for model_name in ["deepseek-r1", "o3-mini", "claude-3.7-sonnet", "gemini-2.0-flash-thinking"]:
        print(f"\n测试模型: {model_name}")
        messages = [
            {"role": "user", "content": "你好，请用一句话介绍自己"}
        ]
        response = llm.chat(messages, model=model_name)
        print(f"响应: {response}")