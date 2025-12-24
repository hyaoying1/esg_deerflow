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

class ESGTranslateService:
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

    def translate_sync(self):
        """
        同步执行润色任务
        :param user_instruction: 用户指令
        :param on_failure: 失败回调函数(异常对象, 上下文信息)
        """
        format_instructions = self.parser.get_format_instructions()

        system_prompt = f'''
        你是一名资深 ESG 报告英文撰写与翻译专家，长期为港股、美股上市公司及跨国企业提供 ESG 披露支持。

        你的任务不是进行逐句直译，而是基于以下原则，将【中文 ESG 报告内容】的待翻译原文{self.current_text}转化为【符合国际资本市场披露习惯的专业英文 ESG 报告文本】：
        
        【翻译与改写原则】
        1. 保持事实一致，不新增未经原文支持的信息；
        2. 优先采用国际通行的 ESG 披露语言与结构（参考 GRI、ISSB、TCFD、UNGC 等框架）；
        3. 对中文中偏政策性、口号化、宣传性的表述进行专业重写，而非直译；
        4. 强调治理结构、职责分工、管理机制、风险与机遇，而非态度性表述；
        5. 对指标、成效、成果类内容，使用清晰、可核查、国际通行的表达方式；
        6. 对中国特有制度、法律或治理安排，采用“功能性翻译”，确保海外读者可理解；
        7. 英文整体语气应：客观、克制、专业，避免 marketing 或 PR 语言。
        
        【必须遵守的硬性风格规则】
        1. 统一使用国际商务英语（International Business English）        
        2. 词汇与拼写默认美式（American English）        
        3. 避免虚拟语气、夸张修辞、情绪化表达        
        4. 禁止推测性 / 假设性承诺（unless supported by data）        
        5. 优先使用主动语态 + 客观陈述        
        6. 时态严格对应事实（过去 / 现在 / 计划）        
        7. 避免绝对化表述（always / fully / completely）       
        8. 语气克制、分析导向，不使用 PR / Marketing 语言
        
        【输出要求】
        - 输出为完整、连贯的英文 ESG 报告段落；
        - 不需要逐句对照，不标注“翻译说明”；
        - 不添加中文原文中不存在的承诺或目标；
        - 语言风格符合国际上市公司 ESG 报告正文。
        {format_instructions}
        '''

        user_prompt = f"待翻译原文：{self.current_text}"

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
                f"\n>>> [Processing] 正在使用 {model_label} 翻译 V{self.current_index} -> V{self.current_index + 1}...")

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
                # if self.current_index < len(self.versions) - 1:
                #     self.versions = self.versions[:self.current_index + 1]
                #
                # new_ver = ReportVersion(
                #     version_id=self.versions[-1].version_id + 1,
                #     timestamp=datetime.now().strftime("%H:%M:%S"),
                #     content=parsed_result['rewritten_content']
                # )
                # self.versions.append(new_ver)
                # self.current_index += 1

                # print(f"\n\n✅ [Success] V{new_ver.version_id} 生成完毕。")
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
        print(f"\n❌ [Final Error] 所有模型尝试均失败。取消翻译，返回原文。")

        # 返回原文构造的假结果，确保调用方逻辑不中断，同时保持版本不变
        return {
            "rewritten_content": self.current_text,
            "changes_summary": ["由于系统异常，翻译未成功，已保留原文"],
            "risk_check": "由于翻译失败，未进行合规性扫描"
        }

    def translate(self):
        """
        同步执行润色任务
        :param user_instruction: 用户指令
        :param on_failure: 失败回调函数(异常对象, 上下文信息)
        """
        format_instructions = self.parser.get_format_instructions()

        system_prompt = f'''
        You are an ESG disclosure reviewer specializing in international sustainability reporting standards and ESG risk assessment.

        You are a senior ESG disclosure and compliance expert with experience in:
        - ESG report drafting for listed companies
        - Regulatory review by international investors and ESG rating agencies
        - Greenwashing and overstatement risk assessment
        
        Your task is to perform a closed-loop ESG compliance revision on the provided English ESG disclosure text.
        
        You must internally complete the following steps:
        1. Identify language that may be considered non-compliant, overly promotional, absolute, or insufficiently supported, including but not limited to:
           - Absolute or guarantee-based expressions (e.g. "ensures", "fully", "significantly")
           - Subjective or evaluative adjectives (e.g. "efficient", "strong", "robust") without explanation
           - Implicit performance conclusions without disclosed mechanisms
           - Strategy or commitment statements that imply outcomes rather than governance processes
        2. Assess the risk level of such expressions (high / medium / low).
        3. Revise the text to:
           - Remove or downgrade absolute or promotional language
           - Prioritize governance structures, processes, and accountability mechanisms
           - Use neutral, verifiable, and disclosure-appropriate wording
           - Maintain factual accuracy and original meaning without introducing new claims
        
        IMPORTANT OUTPUT RULES:
        - Only output the revised, compliance-enhanced ESG disclosure text
        - Do NOT explain your reasoning
        - Do NOT list issues or risk levels
        - Do NOT add any content not implied by the original text
        - Use formal, neutral ESG reporting language suitable for international disclosure

        Pay special attention to overstatement and commitment-related phrases such as:
        "committed to", "actively", "fully", "ensure", "significantly", "continuously".
        
        Output Requirements:
        {format_instructions}
        '''

        user_prompt = f"English ESG disclosure text：{self.translate_sync()}"

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
                f"\n>>> [Processing] 正在使用 {model_label} 英文润色 V{self.current_index} -> V{self.current_index + 1}...")

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
                    content=parsed_result['rewritten_content']
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
                service = ESGTranslateService()
                service.initialize_text(content)
                satisfy = False
                while not satisfy:
                    service.translate()
                    print(f"当前翻译效果: {service.current_text}")
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
                        satisfy_raw = input('是否达到满意的翻译效果？（y/n）： ')
                        satisfy = True if satisfy_raw == "y" else False
                    print("历史版本信息： ", service.versions)
                    print("指针位置： ", service.current_index)
                print(f"最终翻译效果: {service.current_text}")
                # main(content)
                print('___________________________________')