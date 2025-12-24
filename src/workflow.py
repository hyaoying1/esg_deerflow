# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging

from src.config.configuration import get_recursion_limit
from src.graph import build_graph
from src.graph.utils import build_clarified_topic_from_history
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph()


async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
    enable_clarification: bool | None = None,
    max_clarification_rounds: int | None = None,
    initial_state: dict | None = None,
):
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting async workflow with user input: {user_input}")

    if initial_state is None:
        initial_state = {
            "messages": [{"role": "user", "content": user_input}],
            "auto_accepted_plan": True,
            "enable_background_investigation": enable_background_investigation,
            "workflow_stage": "esg_prepare",
            "research_topic": user_input,
            "clarified_research_topic": user_input,
        }

        if enable_clarification is not None:
            initial_state["enable_clarification"] = enable_clarification

        if max_clarification_rounds is not None:
            initial_state["max_clarification_rounds"] = max_clarification_rounds

    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
        },
        "recursion_limit": get_recursion_limit(default=100),
    }

    # ---------- Phase 1: planning ----------
    async for s in graph.astream(
        input=initial_state,
        config=config,
        stream_mode="values",
    ):
        final_state = s
    print("introductory part end")

    writing_tag = final_state.get("esg_writing_tag")
    if writing_tag not in ['董事会声明', '管理层致辞', '关于#企业#', '关于本报告']:
        titles_content = final_state.get("esg_title_contents_raw", [])
        company_name = final_state.get("esg_company_name", "")
        industry = final_state.get("esg_company_industry", "")
        report_id = final_state.get("esg_report_id")
        tenant_id = final_state.get("esg_tenant_id")

        # ---------- Phase 2: section writing ----------
        tasks = [
            process_single_title(
                report_id=report_id,
                tenant_id=tenant_id,
                company_name=company_name,
                content=content,
                industry=industry,
            )
            for content in titles_content
        ]

        final_results = await asyncio.gather(*tasks)

        sections = []
        for raw in final_results:
            state = safe_parse_dict(raw)
            section = assemble_esg_section(state)
            sections.append(section)

        return {
            "final_report": sections
        }
    else:
        report = final_state.get("esg_part_report_result")
        return {
            "final_report": [report]
        }


async def process_single_title(
    report_id,
    tenant_id,
    company_name: str,
    content: str,
    industry: str,
):
    """
    Write a single ESG section and return final raw text
    (正文 + table)
    """
    writing_graph = build_graph()

    initial_state = {
        "workflow_stage": "esg_part_write",
        "esg_report_id": report_id,
        "esg_tenant_id": tenant_id,
        "esg_company_name": company_name,
        "esg_company_industry": industry,
        "esg_title_contents_raw": content,
    }

    async for s in writing_graph.astream(
        input=initial_state,
        config={},
        stream_mode="values",
    ):
        final_state = s

    # === 1. 正文（来自 write_report node）
    section_text = final_state.get("report_part_text", "").strip()

    # === 2. 表格（来自 create_table_of_contents node）
    tables = final_state.get("esg_indicator_tables", [])

    if tables:
        section_text += "\n\n"
        section_text += "\n\n".join(tables)

    # === 兜底，防止空
    if not section_text:
        section_text = str(final_state)

    return section_text

import ast

def safe_parse_dict(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception as e:
            raise ValueError(f"Failed to parse string dict: {e}")
    raise TypeError(f"Unsupported type: {type(raw)}")


def assemble_esg_section(state: dict) -> dict:
    title_raw = state["esg_title_contents_raw"]
    report_result = state.get("esg_part_report_result", {})
    tables = state.get("esg_part_tables_markdown", [])

    # 1. 正文（part1 + part2 + part3）
    text_parts = [
        report_result.get("part1", ""),
        report_result.get("part2", ""),
        report_result.get("part3", ""),
    ]
    text_body = "\n\n".join(p for p in text_parts if p.strip())

    # 2. 表格拼接
    table_text = "\n\n".join(tables)

    writing_content = text_body
    if table_text:
        writing_content += "\n\n\n" + table_text

    return {
        "report_id": title_raw["report_id"],
        "tenant_id": title_raw["tenant_id"],
        "title_id": title_raw["title_id"],
        "title": title_raw["title"],
        "parent_id": title_raw["parent_id"],
        "writing_content": writing_content,
    }


'''async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
    enable_clarification: bool | None = None,
    max_clarification_rounds: int | None = None,
    initial_state: dict | None = None,
):
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning to enhance context
        enable_clarification: If None, use default from State class (False); if True/False, override
        max_clarification_rounds: Maximum number of clarification rounds allowed
        initial_state: Initial state to use (for recursive calls during clarification)

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting async workflow with user input: {user_input}")

    # Use provided initial_state or create a new one
    if initial_state is None:
        initial_state = {
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "auto_accepted_plan": True,
            "enable_background_investigation": enable_background_investigation,
        }
        initial_state["research_topic"] = user_input
        initial_state["clarified_research_topic"] = user_input

        # Only set clarification parameter if explicitly provided
        # If None, State class default will be used (enable_clarification=False)
        if enable_clarification is not None:
            initial_state["enable_clarification"] = enable_clarification

        if max_clarification_rounds is not None:
            initial_state["max_clarification_rounds"] = max_clarification_rounds

    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "mcp_settings": {
                "servers": {
                    "mcp-github-trending": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-github-trending"],
                        "enabled_tools": ["get_github_trending_repositories"],
                        "add_to_agents": ["researcher"],
                    }
                }
            },
        },
        "recursion_limit": get_recursion_limit(default=100),
    }
    last_message_cnt = 0
    final_state = None
    async for s in graph.astream(
        input=initial_state, config=config, stream_mode="values"
    ):
        try:
            final_state = s
            if isinstance(s, dict) and "messages" in s:
                if len(s["messages"]) <= last_message_cnt:
                    continue
                last_message_cnt = len(s["messages"])
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
            else:
                print(f"Output: {s}")
        except Exception as e:
            logger.error(f"Error processing stream output: {e}")
            print(f"Error processing output: {str(e)}")

    # # Check if clarification is needed using centralized logic
    # if final_state and isinstance(final_state, dict):
    #     from src.graph.nodes import needs_clarification

    #     if needs_clarification(final_state):
    #         print("########################################")
    #         logger.info("Clarification needed, prompting user for input")
    #         # Wait for user input
    #         print()
    #         clarification_rounds = final_state.get("clarification_rounds", 0)
    #         max_clarification_rounds = final_state.get("max_clarification_rounds", 3)
    #         user_response = input(
    #             f"Your response ({clarification_rounds}/{max_clarification_rounds}): "
    #         ).strip()

    #         if not user_response:
    #             logger.warning("Empty response, ending clarification")
    #             return final_state

    #         # Continue workflow with user response
    #         current_state = final_state.copy()
    #         current_state["messages"] = final_state["messages"] + [
    #             {"role": "user", "content": user_response}
    #         ]
    #         for key in (
    #             "clarification_history",
    #             "clarification_rounds",
    #             "clarified_research_topic",
    #             "research_topic",
    #             "locale",
    #             "enable_clarification",
    #             "max_clarification_rounds",
    #         ):
    #             if key in final_state:
    #                 current_state[key] = final_state[key]

    #         return await run_agent_workflow_async(
    #             user_input=user_response,
    #             max_plan_iterations=max_plan_iterations,
    #             max_step_num=max_step_num,
    #             enable_background_investigation=enable_background_investigation,
    #             enable_clarification=enable_clarification,
    #             max_clarification_rounds=max_clarification_rounds,
    #             initial_state=current_state,
    #         )

    logger.info("Async workflow completed successfully")
    return final_state'''




if __name__ == "__main__":
    print(graph.get_graph(xray=True).draw_mermaid())
