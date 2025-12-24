def _log_token_usage(response, model_name="未知模型"):
    """
    通用 token 使用日志打印函数，兼容多种 LLM 响应格式
    """
    print(1)
    try:
        usage = None

        # ✅ LangChain ChatOpenAI 格式
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage")

        # ✅ OpenAI Python SDK 格式
        elif hasattr(response, "usage_metadata"):
            usage = response.usage_metadata

        # ✅ Qwen / vLLM / 通义千问 格式
        elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
            usage = response.additional_kwargs.get("usage", {})

        # ✅ 通义千问兼容模式的标准字段
        elif hasattr(response, "usage"):
            usage = getattr(response, "usage")
        print(usage)
        if usage:
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens", "N/A")
                output_tokens = usage.get("output_tokens") or usage.get("completion_tokens", "N/A")
                total_tokens = usage.get("total_tokens", "N/A")
            else:
                # CompletionUsage 对象的典型结构
                input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", "N/A")
                output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", "N/A")
                total_tokens = getattr(usage, "total_tokens", "N/A")
            print(
                f"📊 [{model_name}] Token使用情况 - 输入: {input_tokens}, 输出: {output_tokens}, 总计: {total_tokens}"
            )
        else:
            print(f"⚠️ [{model_name}] 当前接口未返回 token 使用信息。")

    except Exception as e:
        print(f"❌ 解析 token 使用信息失败: {e}")