import json
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search, calculate

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下 JSON 格式进行回应，每个键都要包含，但值可以为空，具体规则会在下面的注释中说明。不要包含任何额外内容：

{{
  "thought": "你的思考过程，用于分析问题、拆解任务和规划下一步行动。",
  "action": {{
    "type": "tool", // 或 "finish"
    "tool_name": "工具名称", // 当 type 为 "tool" 时必填
    "tool_input": "工具输入", // 当 type 为 "tool" 时必填
    "final_answer": "最终答案" // 当 type 为 "finish" 时必填
  }}
}}

当你收集到足够的信息，能够回答用户的最终问题时，你必须使用 type: "finish" 格式。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""
class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        """
        运行ReAct智能体来回答一个问题。
        """
        self.history = [] # 每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            # ... (后续的解析、执行、整合步骤)
            # (这段逻辑在 run 方法的 while 循环内)
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            # 4. 执行Action
            tool_name, tool_input = self._parse_action(action)
            
            if tool_name == 'Finish':
                # 如果是Finish指令，提取最终答案并结束
                final_answer = tool_input
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            if not tool_name or not tool_input:
                # ... 处理无效Action格式 ...
                print("警告:未能解析出有效的Action，流程终止。")
                break

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
            # (这段逻辑紧随工具调用之后，在 while 循环的末尾)
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            action_str = f"{tool_name}[{tool_input}]"
            self.history.append(f"Action: {action_str}")
            self.history.append(f"Observation: {observation}")
        # 循环结束
        print("已达到最大步数，流程终止。")
        return None
        
    # (这些方法是 ReActAgent 类的一部分)
    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取Thought和Action。
        """
        thought = None
        action = {}
        try:
            # 尝试解析 JSON
            data = json.loads(text)
            thought = data.get('thought', '').strip()
            action = data.get('action', {})
            return thought, action
        except json.JSONDecodeError as e:
            print(f"警告: JSON 解析失败: {e}")
            return thought, action

    def _parse_action(self, action_data):
        """
        解析Action数据，提取工具名称和输入。
        """
        if not isinstance(action_data, dict):
            return None, None,false
        
        action_type = action_data.get('type')
        if action_type == 'finish':
            final_answer = action_data.get('final_answer', '').strip()
            return 'Finish', final_answer
        elif action_type == 'tool':
            tool_name = action_data.get('tool_name', '').strip()
            tool_input = action_data.get('tool_input', '').strip()
            return tool_name, tool_input
        return None, None



if __name__ == '__main__':
    # 1. 初始化LLM客户端和工具执行器
    llm_client = HelloAgentsLLM()
    toolExecutor = ToolExecutor()
    react_agent = ReActAgent(llm_client, toolExecutor)
    # 注册搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    # 注册计算器工具
    calculate_description = "一个数学计算器。当你需要进行数学计算时，应使用此工具。"
    toolExecutor.registerTool("Calculate", calculate_description, calculate)
    # 运行测试
    react_agent.run("计算 (123 + 456) × 789 / 12 = ? 的结果")