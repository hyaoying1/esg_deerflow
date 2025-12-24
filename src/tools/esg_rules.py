import os
from pydantic import BaseModel, Field


class IndustaryInfoRules(BaseModel):
    industry: str = Field(..., description="行业名称，总结公司主营业务，得到公司行业结论，最多不超过20个字。")
    main_business: str = Field(..., description="总结公司主营业务，最多不超过20个字，属行业信息收集的详细指引。")


class ReportInfoRules(BaseModel):
    part1: str = Field(..., description="第一部分报告正文。")
    part2: str = Field(..., description="第二部分报告正文。")
    part3: str = Field(..., description="第三部分报告正文。")
    part4: str = Field(..., description="第四部分报告正文。")
    part5: str = Field(..., description="第五部分报告正文。")