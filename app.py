import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from esg_main import run_esg_writer  


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ESG Report Runner")

# 挂载静态文件：让浏览器能直接访问 /static/index.html
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class GenerateReq(BaseModel):
    report_id: int = Field(..., description="报告ID")
    tenant_id: int = Field(..., description="租户ID")
    title_id: int = Field(..., description="章节/标题ID")


@app.get("/", response_class=HTMLResponse)
def home():
    # 直接返回前端页面
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h3>static/index.html not found</h3><p>Please create static/index.html</p>",
            status_code=500,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/generate")
async def generate(req: GenerateReq):
    """
    接收三个数字 -> 在本机执行 run_esg_writer -> 返回结果
    """
    try:
        # 你给的逻辑等价于：
        # if __name__ == "__main__":
        #     report_id = 545
        #     tenant_id = 2
        #     title_id = 14
        #     result = asyncio.run(run_esg_writer(report_id, tenant_id, title_id))

        result = await run_esg_writer(req.report_id, req.tenant_id, req.title_id)

        # result 可能是 dict / list / str / pydantic / 其他对象
        # 这里尽量把它变成可 JSON 序列化的结构
        safe_result: Any
        if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
            safe_result = result
        else:
            # 尝试用 __dict__，不行就转成字符串
            safe_result = getattr(result, "__dict__", None) or str(result)

        return JSONResponse(
            {
                "ok": True,
                "inputs": req.model_dump(),
                "result": safe_result,
            }
        )
    except Exception as e:
        return JSONResponse(
            {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            },
            status_code=500,
        )


@app.get("/health")
def health():
    return {"ok": True}


# uvicorn app:app --host 0.0.0.0 --port 8000
