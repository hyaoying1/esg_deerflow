from datetime import datetime
import json
import asyncio
import os
import time
from pathlib import Path
from src.workflow import run_agent_workflow_async


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "__dict__"):
        return {k: to_json_safe(v) for k, v in obj.__dict__.items()}
    else:
        return obj


async def run_esg_writer(report_id, tenant_id, title_id):
    input = f"[ESG_WRITE] report_id={report_id}, tenant_id={tenant_id}, title_id={title_id}"
    return await run_agent_workflow_async(user_input=input)


if __name__ == "__main__":
    report_id = 545
    tenant_id = 2
    title_id = 11
    result = asyncio.run(run_esg_writer(report_id, tenant_id, title_id))

    if not isinstance(result, dict):
        raise TypeError(f"Workflow must return dict, got {type(result)}")

    sections = result.get("final_report", [])
    if not isinstance(sections, list):
        raise TypeError("final_report must be a list")


    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = output_dir / f"output_{now_str}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
