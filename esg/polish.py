import asyncio
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
import json
import time
import re


# --- 1. 定义版本快照模型 ---
class ReportVersion(BaseModel):
    version_id: int
    timestamp: str
    content: str = Field(description="当前版本的全文内容")
    instruction: str = Field(description="生成此版本时的用户指令")
    changes_summary: List[str] = Field(description="相比上一版的主要修改点", default=[])
    risk_check: str = Field(description="合规性检查结果", default="")


# --- 2. 定义 LLM 输出结构 ---
class ReportInfoRulesBody(BaseModel):
    rewritten_content: str
    changes_summary: List[str]
    risk_check: str


# --- 配置 ---
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-8c7192b42d6a44649157d4769aaadf12"
MODEL_MAX = 'qwen-max'
MODEL_PLUS = 'qwen-plus'
LLM = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_MAX,
            temperature=0
        )

class ESGPolishingService:
    def __init__(self):
        self.llm_max = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_MAX,
            streaming=True,
            temperature=0.3,
            extra_body={"enable_thinking": True}
        )
        self.llm_plus = ChatOpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_PLUS,
            streaming=True,
            temperature=0.3,
            extra_body={"enable_thinking": True}
        )
        self.parser = JsonOutputParser(pydantic_object=ReportInfoRulesBody)

        # --- 版本控制核心属性 ---
        self.versions: List[ReportVersion] = []  # 存储所有版本历史
        self.current_index: int = -1  # 当前指针位置

    @property
    def current_text(self) -> str:
        """获取当前指针指向的文本内容"""
        if self.current_index >= 0 and self.versions:
            return self.versions[self.current_index].content
        return ""

    def initialize_text(self, text: str):
        """初始化 V0 版本"""
        v0 = ReportVersion(
            version_id=0,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            content=text,
            instruction="[原始素材导入]",
            changes_summary=["初始版本"],
            risk_check="N/A"
        )
        self.versions = [v0]
        self.current_index = 0
        print(f"✅ 系统初始化完成 (V0)，字数: {len(text)}")

    def rollback(self, rollback_sign) -> bool:
        """回滚到上一个版本 (Undo)"""
        if rollback_sign:
            if self.current_index > 0:
                self.current_index -= 1
                prev_ver = self.versions[self.current_index]
                print(f"⏪ 已回滚至 V{prev_ver.version_id}。当前指令状态: {prev_ver.instruction}")
                return True
            else:
                print("⚠️ 已经是初始版本，无法回滚。")
                return False
        else:
            return False

    def rollback_to_initial(self, rollback_sign) -> bool:
        """回滚到上一个版本 (Undo)"""
        if rollback_sign:
            if self.current_index > 0:
                self.current_index = 0
                prev_ver = self.versions[self.current_index]
                print(f"⏪ 已回滚到初始状态。当前指令状态: {prev_ver.instruction}")
                return True
            else:
                print("⚠️ 已经是初始版本，无法回滚。")
                return False
        else:
            return False

    def forward(self) -> bool:
        """重做/前进 (Redo)"""
        if self.current_index < len(self.versions) - 1:
            self.current_index += 1
            next_ver = self.versions[self.current_index]
            print(f"⏩ 已前进至 V{next_ver.version_id}。")
            return True
        else:
            print("⚠️ 已经是最新版本，无法前进。")
            return False

    def show_history(self):
        """打印版本树"""
        print("\n--- 版本历史记录 ---")
        for idx, ver in enumerate(self.versions):
            marker = "👈 (Current)" if idx == self.current_index else ""
            print(f"V{ver.version_id} [{ver.timestamp}] - {ver.instruction[:20]}... {marker}")
        print("--------------------\n")

    def instruction_check(self, user_instruction):
        system_prompt = f'''
                你是一名专业的 ESG 报告顾问，具备：
                - 上市公司 ESG 披露与年报撰写经验
                - 咨询公司风险管理与内控方法论
                - 对监管合规、漂绿风险高度敏感

                你的任务：判断【用户指令】是否属于以下类型之一：
                - 对 ESG 报告原文的扩写、缩写
                - 对 ESG 报告原文的润色、改写、优化、提升专业性、合规性或表达质量
                - 对 ESG 披露语言语气的风格、结构、逻辑、规范性调整
                
                若属于以上情况，输出：是  
                否则（如非报告相关需求、预测、闲聊、无关问题），输出：不是
                输出要求：只允许输出以下两个词之一：
                - 是
                - 不是
                禁止输出任何解释、标点或其他文字。
                '''
        user_prompt = f"用户指令：{user_instruction}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        match_result = LLM.invoke(messages)
        answer = match_result.content.strip()

        # 只保留“是”或“不是”
        if '是' in answer and '不是' not in answer:
            return True
        else:
            return False



    def polish_sync(self, user_instruction):
        """
        同步执行润色任务
        :param user_instruction: 用户指令
        :param on_failure: 失败回调函数(异常对象, 上下文信息)
        """
        check = self.instruction_check(user_instruction)
        if not check:
            print("[指令不属于合理润色要求]")
            return "[指令不属于合理润色要求]"
        format_instructions = self.parser.get_format_instructions()

        system_prompt = f'''
        你是一名专业的 ESG 报告顾问，具备：
        - 上市公司 ESG 披露与年报撰写经验
        - 咨询公司风险管理与内控方法论
        - 对监管合规、漂绿风险高度敏感
        
        你的任务是根据用户指令要求对 ESG 原文进行专业润色。
        
        【强制约束】
        1. 不新增任何事实、数据、案例或结论
        2. 不虚构管理成效或量化结果
        3. 不扩大承诺范围或时间边界
        4. 不改变原文披露口径与含义
        5. 对前瞻性内容必须使用审慎、有限的表达
        6. 输出内容应可直接用于正式 ESG 报告
        
        如原文存在表述风险，请降低语气，而非强化表述。
        
        关键限制条件 (Key Constraints)
        严格的内容边界： 你的撰写必须严格基于提供的原始素材，严禁从网络检索或杜撰任何原始素材中未包含的信息、数据、案例、承诺或管理措施。
        原始文字输出： 如果原文是中文，请按中文输出，如果是英文，请按英文输出。
        标点符号： 必须100%遵循原始文字标点符号的正确使用规范。
        
        输出必须严格按照要求的结构返回，不得额外发挥。
        {format_instructions}
        '''

        user_prompt = f"原文：{self.current_text}\n用户指令：{user_instruction}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # --- 策略配置 ---
        max_retries = 3
        # 定义尝试队列：前 3 次用 llm_max，最后 1 次用 llm_plus
        model_attempts = [self.llm_max] * max_retries + [self.llm_plus]

        last_exception = None

        for i, model in enumerate(model_attempts):
            is_fallback = (i >= max_retries)
            model_label = "llm_plus" if is_fallback else f"llm_max (Attempt {i + 1})"

            print(
                f"\n>>> [Processing] 正在使用 {model_label} 润色 V{self.current_index} -> V{self.current_index + 1}...")

            try:
                full_response_text = ""
                # 同步流式调用
                for chunk in model.stream(messages):
                    if chunk.content:
                        print(chunk.content, end='', flush=True)
                        full_response_text += chunk.content

                # 1. 提取 JSON 内容
                cleaned_text = full_response_text.strip()
                if "```" in cleaned_text:
                    match = re.search(r"```(?:json)?(.*?)```", cleaned_text, re.DOTALL)
                    if match: cleaned_text = match.group(1)

                # 2. 解析 JSON (如果格式不对会抛出异常进入 next loop)
                parsed_result = self.parser.parse(cleaned_text)

                # --- 验证通过，更新版本树 ---
                if self.current_index < len(self.versions) - 1:
                    self.versions = self.versions[:self.current_index + 1]

                new_ver = ReportVersion(
                    version_id=self.versions[-1].version_id + 1,
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    content=parsed_result['rewritten_content'],
                    instruction=user_instruction
                )
                self.versions.append(new_ver)
                self.current_index += 1

                print(f"\n\n✅ [Success] V{new_ver.version_id} 生成完毕。")
                return parsed_result

            except Exception as e:
                last_exception = e
                print(f"\n⚠️ [Attempt {i + 1} Failed] 错误类型: {type(e).__name__}")

                # 如果还有重试机会，则继续循环
                if i < len(model_attempts) - 1:
                    print(f"🔄 准备进行下一次尝试...")
                    time.sleep(1)  # 适当延迟避免 429 持续触发
                    continue
                else:
                    # 所有尝试均已用尽
                    break

        # --- 最终失败处理：输出原文 ---
        print(f"\n❌ [Final Error] 所有模型尝试均失败。取消润色，返回原文。")

        # 返回原文构造的假结果，确保调用方逻辑不中断，同时保持版本不变
        return {
            "rewritten_content": self.current_text,
            "changes_summary": ["由于系统异常，润色未成功，已保留原文"],
            "risk_check": "由于润色失败，未进行合规性扫描"
        }


# if __name__ == "__main__":
#     asyncio.run(main_with_rollback())

if __name__ == '__main__':
    with open(r'./title_write_515.json', 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    for i in data_list[1:2]:
        content_list = i['writing_content'].split('\n\n')
        print(content_list)
        for content in content_list[:1]:
            if content:
                print(content)
                service = ESGPolishingService()
                service.initialize_text(content)
                satisfy = False
                while not satisfy:
                    polish_guide = input("润色要求： ")
                    print(service.instruction_check(polish_guide))
                    service.polish_sync(polish_guide)
                    print(f"当前润色效果: {service.current_text}")
                    rollback_sign_raw = input("是否需要将当前结果回退?(y/n): ")
                    rollback_sign = True if rollback_sign_raw == "y" else False
                    if rollback_sign:
                        service.rollback(rollback_sign)
                        print(f"当前内容: {service.current_text}")
                        rollback_to_initial_sign_raw = input("是否需要将回退到初始化?(y/n): ")
                        rollback_to_initial_sign = True if rollback_to_initial_sign_raw == "y" else False
                        if rollback_to_initial_sign:
                            service.rollback(rollback_to_initial_sign)
                            print(f"当前内容: {service.current_text}")
                    else:
                        satisfy_raw = input('是否达到满意的润色效果？（y/n）： ')
                        satisfy = True if satisfy_raw == "y" else False
                    print("历史版本信息： ", service.versions)
                    print("指针位置： ", service.current_index)
                print(f"最终润色效果: {service.current_text}")
                # main(content)
                print('___________________________________')