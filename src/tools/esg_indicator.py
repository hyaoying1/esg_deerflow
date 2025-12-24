import asyncio
import os
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from src.utils.token_usage import _log_token_usage

max_concurrent = 20

# 创建信号量控制并发数量
semaphore = asyncio.Semaphore(max_concurrent)

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")

llm_max = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="qwen-max",
    temperature=0,
    top_p=1,
)
llm_plus = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="qwen-plus",
    temperature=0,
    top_p=1,
)
llm_flash = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="qwen-flash",
    temperature=0,
    top_p=1,
)

import logging
logger = logging.getLogger(__name__)

async def invoke_with_retry(messages, parser, max_retries=5, initial_delay=2):
    """
    调用主模型 (llm_max)，在失败后自动重试并切换到备用模型 (llm_plus)。
    增加 token 使用统计。
    """
    retries = 0

    while retries <= max_retries:
        try:
            async with semaphore:
                # 尝试使用主模型
                result = await llm_max.ainvoke(messages)

                # ✅ 统计主模型 token 使用
                try:
                    _log_token_usage(result, model_name="主模型 max")
                except Exception as e:
                    print(f"无法记录 token 使用: {e}")

                return parser.parse(result.content)

        except Exception as e:
            retries += 1
            delay = initial_delay * (2 ** (retries - 1))

            if retries > max_retries:
                logger.warning(f"主模型多次调用失败，切换到备用模型: {e}")
                async with semaphore:
                    result = await llm_plus.ainvoke(messages)

                    # ✅ 统计备用模型 token 使用
                    _log_token_usage(result, model_name="备用模型 plus")

                    return parser.parse(result.content)

            logger.warning(f"主模型调用失败（第{retries}次重试）: {e}，{delay}秒后重试...")
            await asyncio.sleep(delay)

    # 理论上不会执行到这里
    raise Exception("所有调用尝试均失败")


model_thinking = 'qwen3-235b-a22b-thinking-2507'
THINKING_BUDGET = 5000

client = AsyncOpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=API_KEY,
    base_url=BASE_URL,
)

