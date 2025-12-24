from dataclasses import field
from langgraph.graph import MessagesState

from src.prompts.planner_model import Plan
from src.rag import Resource


class State(MessagesState):
    """Extended global workflow state"""

    # === Default LLM workflow variables ===
    locale: str = "en-US"
    research_topic: str = ""
    clarified_research_topic: str = (
        ""  # Complete/final clarified topic with all clarification rounds
    )
    observations: list[str] = []
    resources: list[Resource] = []
    plan_iterations: int = 0
    current_plan: Plan | str = None
    final_report: str = ""
    auto_accepted_plan: bool = False
    enable_background_investigation: bool = True
    background_investigation_results: str = None

    # Clarification state tracking (disabled by default)
    enable_clarification: bool = (
        False  # Enable/disable clarification feature (default: False)
    )
    clarification_rounds: int = 0
    clarification_history: list[str] = field(default_factory=list)
    is_clarification_complete: bool = False
    max_clarification_rounds: int = 3

    # === ESG WRITING SPECIFIC FIELDS ===
    workflow_stage: str = ""
    esg_report_id: int | None = None
    esg_tenant_id: int | None = None
    esg_title_id: int | None = None
    esg_company_name: str = ""
    esg_title_datas: str = ""
    esg_title_contents_raw: str = ""
    esg_company_industry: str = ""
    current_sub_title_topics: str = ""
    law_info_text: str = ""
    esg_writing_tag: str = ""
    esg_part_report_result: str = ""
    esg_part_tables_markdown: list[str] = field(default_factory=list)
    result: str = ""

    # === Workflow next pointer ===
    goto: str = "planner"
