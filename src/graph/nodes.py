# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
import re
from collections import defaultdict, deque
from decimal import Decimal
from functools import partial
from typing import Any, Annotated, Literal


# LangChain / LangGraph
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command, interrupt


# MCP / External Adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

# Project: Agents & Config
from src.agents import create_agent
from src.config.agents import AGENT_LLM_MAP
from src.config.configuration import Configuration
from ..config import SELECTED_SEARCH_ENGINE, SearchEngine

# Project: LLMs
from src.llms.llm import (
    get_llm_by_type,
    get_llm_token_limit_by_type,
)


# Project: Prompts & Planning
from src.prompts.planner_model import Plan
from src.prompts.template import apply_prompt_template


# Project: Tools
from src.tools import (
    crawl_tool,
    get_retriever_tool,
    get_web_search_tool,
    python_repl_tool,
)
from src.tools.connect_sql import get_connection, get_title_contents
from src.tools.search import LoggedTavilySearch
from src.tools.introductory_report_writing import main_report_writing_begining


# Project: ESG Rules & Logic
from src.tools.esg_rules import IndustaryInfoRules, ReportInfoRules
from src.esg_logic.writing_title_type import title_framework

# Project: Utils
from src.utils.context_manager import (
    ContextManager,
    validate_message_content,
)
from src.utils.json_utils import (
    repair_json_output,
    sanitize_tool_response,
)

# Project: Graph State & Helpers
from .types import State
from .utils import (
    build_clarified_topic_from_history,
    get_message_content,
    is_user_message,
    reconstruct_clarification_history,
)

logger = logging.getLogger(__name__)

@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return


@tool
def handoff_after_clarification(
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
    research_topic: Annotated[
        str, "The clarified research topic based on all clarification rounds."
    ],
):
    """Handoff to planner after clarification rounds are complete. Pass all clarification history to planner for analysis."""
    return


def needs_clarification(state: dict) -> bool:
    """
    Check if clarification is needed based on current state.
    Centralized logic for determining when to continue clarification.
    """
    if not state.get("enable_clarification", False):
        return False

    clarification_rounds = state.get("clarification_rounds", 0)
    is_clarification_complete = state.get("is_clarification_complete", False)
    max_clarification_rounds = state.get("max_clarification_rounds", 3)

    # Need clarification if: enabled + has rounds + not complete + not exceeded max
    # Use <= because after asking the Nth question, we still need to wait for the Nth answer
    return (
        clarification_rounds > 0
        and not is_clarification_complete
        and clarification_rounds <= max_clarification_rounds
    )


def preserve_state_meta_fields(state: State) -> dict:
    """
    Extract meta/config fields that should be preserved across state transitions.
    
    These fields are critical for workflow continuity and should be explicitly
    included in all Command.update dicts to prevent them from reverting to defaults.
    
    Args:
        state: Current state object
        
    Returns:
        Dict of meta fields to preserve
    """
    return {
        "locale": state.get("locale", "en-US"),
        "research_topic": state.get("research_topic", ""),
        "clarified_research_topic": state.get("clarified_research_topic", ""),
        "clarification_history": state.get("clarification_history", []),
        "enable_clarification": state.get("enable_clarification", False),
        "max_clarification_rounds": state.get("max_clarification_rounds", 3),
        "clarification_rounds": state.get("clarification_rounds", 0),
        "resources": state.get("resources", []),
    }


def validate_and_fix_plan(plan: dict, enforce_web_search: bool = False) -> dict:
    """
    Validate and fix a plan to ensure it meets requirements.

    Args:
        plan: The plan dict to validate
        enforce_web_search: If True, ensure at least one step has need_search=true

    Returns:
        The validated/fixed plan dict
    """
    if not isinstance(plan, dict):
        return plan

    steps = plan.get("steps", [])

    # ============================================================
    # SECTION 1: Repair missing step_type fields (Issue #650 fix)
    # ============================================================
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        
        # Check if step_type is missing or empty
        if "step_type" not in step or not step.get("step_type"):
            # Infer step_type based on need_search value
            # Default to "analysis" for non-search steps (Issue #677: not all processing needs code)
            inferred_type = "research" if step.get("need_search", False) else "analysis"
            step["step_type"] = inferred_type
            logger.info(
                f"Repaired missing step_type for step {idx} ({step.get('title', 'Untitled')}): "
                f"inferred as '{inferred_type}' based on need_search={step.get('need_search', False)}"
            )

    # ============================================================
    # SECTION 2: Enforce web search requirements
    # ============================================================
    if enforce_web_search:
        # Check if any step has need_search=true (only check dict steps)
        has_search_step = any(
            step.get("need_search", False) 
            for step in steps 
            if isinstance(step, dict)
        )

        if not has_search_step and steps:
            # Ensure first research step has web search enabled
            for idx, step in enumerate(steps):
                if isinstance(step, dict) and step.get("step_type") == "research":
                    step["need_search"] = True
                    logger.info(f"Enforced web search on research step at index {idx}")
                    break
            else:
                # Fallback: If no research step exists, convert the first step to a research step with web search enabled.
                # This ensures that at least one step will perform a web search as required.
                if isinstance(steps[0], dict):
                    steps[0]["step_type"] = "research"
                    steps[0]["need_search"] = True
                    logger.info(
                        "Converted first step to research with web search enforcement"
                    )
        elif not has_search_step and not steps:
            # Add a default research step if no steps exist
            logger.warning("Plan has no steps. Adding default research step.")
            plan["steps"] = [
                {
                    "need_search": True,
                    "title": "Initial Research",
                    "description": "Gather information about the topic",
                    "step_type": "research",
                }
            ]

    return plan


def background_investigation_node(state: State, config: RunnableConfig):
    logger.info("background investigation node is running.")
    configurable = Configuration.from_runnable_config(config)
    query = state.get("clarified_research_topic") or state.get("research_topic")
    background_investigation_results = []
    
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        searched_content = LoggedTavilySearch(
            max_results=configurable.max_search_results
        ).invoke(query)
        # check if the searched_content is a tuple, then we need to unpack it
        if isinstance(searched_content, tuple):
            searched_content = searched_content[0]
        
        # Handle string JSON response (new format from fixed Tavily tool)
        if isinstance(searched_content, str):
            try:
                parsed = json.loads(searched_content)
                if isinstance(parsed, dict) and "error" in parsed:
                    logger.error(f"Tavily search error: {parsed['error']}")
                    background_investigation_results = []
                elif isinstance(parsed, list):
                    background_investigation_results = [
                        f"## {elem.get('title', 'Untitled')}\n\n{elem.get('content', 'No content')}" 
                        for elem in parsed
                    ]
                else:
                    logger.error(f"Unexpected Tavily response format: {searched_content}")
                    background_investigation_results = []
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Tavily response as JSON: {searched_content}")
                background_investigation_results = []
        # Handle legacy list format
        elif isinstance(searched_content, list):
            background_investigation_results = [
                f"## {elem['title']}\n\n{elem['content']}" for elem in searched_content
            ]
            return {
                "background_investigation_results": "\n\n".join(
                    background_investigation_results
                )
            }
        else:
            logger.error(
                f"Tavily search returned malformed response: {searched_content}"
            )
            background_investigation_results = []
    else:
        background_investigation_results = get_web_search_tool(
            configurable.max_search_results
        ).invoke(query)
    
    return {
        "background_investigation_results": json.dumps(
            background_investigation_results, ensure_ascii=False
        )
    }


def planner_node(
    state: State, config: RunnableConfig
) -> Command[Literal["human_feedback", "reporter"]]:
    """Planner node that generate the full plan."""
    logger.info("Planner generating full plan with locale: %s", state.get("locale", "en-US"))
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0

    # For clarification feature: use the clarified research topic (complete history)
    # (stop using the whole chat history and instead plan from a sigle, clean, clarified task statement
    if state.get("enable_clarification", False) and state.get(
        "clarified_research_topic"
    ):
        # Modify state to use clarified research topic instead of full conversation
        modified_state = state.copy()
        modified_state["messages"] = [
            {"role": "user", "content": state["clarified_research_topic"]}
        ]
        modified_state["research_topic"] = state["clarified_research_topic"]
        messages = apply_prompt_template("planner", modified_state, configurable, state.get("locale", "en-US"))

        logger.info(
            f"Clarification mode: Using clarified research topic: {state['clarified_research_topic']}"
        )
    else:
        # Normal mode: use full conversation history
        messages = apply_prompt_template("planner", state, configurable, state.get("locale", "en-US"))

    # if already have background information
    if state.get("enable_background_investigation") and state.get(
        "background_investigation_results"
    ):
        messages += [
            {
                "role": "user",
                "content": (
                    "background investigation results of user query:\n"
                    + state["background_investigation_results"]
                    + "\n"
                ),
            }
        ]

    # decide which llm to use
    if configurable.enable_deep_thinking:
        llm = get_llm_by_type("reasoning")
    elif AGENT_LLM_MAP["planner"] == "basic":
        llm = get_llm_by_type("basic")
    else:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])

    # if the plan iterations is greater than the max plan iterations, return the reporter node
    if plan_iterations >= configurable.max_plan_iterations:
        return Command(
            update=preserve_state_meta_fields(state),
            goto="reporter"
        )

    full_response = ""
    # if is basic model, use invoke
    # 非流式，一次性调用基础模型
    if AGENT_LLM_MAP["planner"] == "basic" and not configurable.enable_deep_thinking:
        response = llm.invoke(messages)
        # return jason format or just content
        if hasattr(response, "model_dump_json"):
            full_response = response.model_dump_json(indent=4, exclude_none=True)
        else:
            full_response = get_message_content(response) or ""
    # 流式调用resoning模型
    # 流式（stream）：模型一边想、一边吐字 → 边生成边返回
    else:
        response = llm.stream(messages)
        # 拼接流式输出
        for chunk in response:
            full_response += chunk.content
    logger.debug(f"Current state messages: {state['messages']}")
    logger.info(f"Planner response: {full_response}")

    # Validate explicitly that response content is valid JSON before proceeding to parse it
    # json外观校验
    if not full_response.strip().startswith('{') and not full_response.strip().startswith('['):
        logger.warning("Planner response does not appear to be valid JSON")
        if plan_iterations > 0:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__"
            )

    try:
        curr_plan = json.loads(repair_json_output(full_response))
        # Need to extract the plan from the full_response
        curr_plan_content = extract_plan_content(curr_plan)
        # load the current_plan
        curr_plan = json.loads(repair_json_output(curr_plan_content))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__"
            )

    # Validate and fix plan to ensure web search requirements are met
    if isinstance(curr_plan, dict):
        curr_plan = validate_and_fix_plan(curr_plan, configurable.enforce_web_search)

    if isinstance(curr_plan, dict) and curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(curr_plan)
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
                **preserve_state_meta_fields(state),
            },
            goto="reporter",
        )
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
            **preserve_state_meta_fields(state),
        },
        goto="human_feedback",
    )


def extract_plan_content(plan_data: str | dict | Any) -> str:
    """
    Safely extract plan content from different types of plan data.
    
    Args:
        plan_data: The plan data which can be a string, AIMessage, or dict
        
    Returns:
        str: The plan content as a string (JSON string for dict inputs, or 
    extracted/original string for other types)
    """
    if isinstance(plan_data, str):
        # If it's already a string, return as is
        return plan_data
    elif hasattr(plan_data, 'content') and isinstance(plan_data.content, str):
        # If it's an AIMessage or similar object with a content attribute
        logger.debug(f"Extracting plan content from message object of type {type(plan_data).__name__}")
        return plan_data.content
    elif isinstance(plan_data, dict):
        # If it's already a dictionary, convert to JSON string
        # Need to check if it's dict with content field (AIMessage-like)
        if "content" in plan_data:
            if isinstance(plan_data["content"], str):
                logger.debug("Extracting plan content from dict with content field")
                return plan_data["content"]
            if isinstance(plan_data["content"], dict):
                logger.debug("Converting content field dict to JSON string")
                return json.dumps(plan_data["content"], ensure_ascii=False)
            else:
                logger.warning(f"Unexpected type for 'content' field in plan_data dict: {type(plan_data['content']).__name__}, converting to string")
                return str(plan_data["content"])
        else:
            logger.debug("Converting plan dictionary to JSON string")
            return json.dumps(plan_data)
    else:
        # For any other type, try to convert to string
        logger.warning(f"Unexpected plan data type {type(plan_data).__name__}, attempting to convert to string")
        return str(plan_data)


def human_feedback_node(
    state: State, config: RunnableConfig
) -> Command[Literal["planner", "research_team", "reporter", "__end__"]]:
    current_plan = state.get("current_plan", "")
    # check if the plan is auto accepted
    auto_accepted_plan = state.get("auto_accepted_plan", False)
    if not auto_accepted_plan:
        feedback = interrupt("Please Review the Plan.")

        # Handle None or empty feedback
        if not feedback:
            logger.warning(f"Received empty or None feedback: {feedback}. Returning to planner for new plan.")
            return Command(
                update=preserve_state_meta_fields(state),
                goto="planner"
            )

        # Normalize feedback string
        feedback_normalized = str(feedback).strip().upper()

        # if the feedback is not accepted, return the planner node
        if feedback_normalized.startswith("[EDIT_PLAN]"):
            logger.info(f"Plan edit requested by user: {feedback}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(content=feedback, name="feedback"),
                    ],
                    **preserve_state_meta_fields(state),
                },
                goto="planner",
            )
        elif feedback_normalized.startswith("[ACCEPTED]"):
            logger.info("Plan is accepted by user.")
        else:
            logger.warning(f"Unsupported feedback format: {feedback}. Please use '[ACCEPTED]' to accept or '[EDIT_PLAN]' to edit.")
            return Command(
                update=preserve_state_meta_fields(state),
                goto="planner"
            )

    # if the plan is accepted, run the following node
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    goto = "research_team"
    try:
        # Safely extract plan content from different types (string, AIMessage, dict)
        original_plan = current_plan
        
        # Repair the JSON output
        current_plan = repair_json_output(current_plan)
        # parse the plan to dict
        current_plan = json.loads(current_plan)
        current_plan_content = extract_plan_content(current_plan)
        
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(repair_json_output(current_plan_content))
        # Validate and fix plan to ensure web search requirements are met
        configurable = Configuration.from_runnable_config(config)
        new_plan = validate_and_fix_plan(new_plan, configurable.enforce_web_search)
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Failed to parse plan: {str(e)}. Plan data type: {type(current_plan).__name__}")
        if isinstance(current_plan, dict) and "content" in original_plan:
            logger.warning(f"Plan appears to be an AIMessage object with content field")
        if plan_iterations > 1:  # the plan_iterations is increased before this check
            return Command(
                update=preserve_state_meta_fields(state),
                goto="reporter"
            )
        else:
            return Command(
                update=preserve_state_meta_fields(state),
                goto="__end__"
            )

    # Build update dict with safe locale handling
    update_dict = {
        "current_plan": Plan.model_validate(new_plan),
        "plan_iterations": plan_iterations,
        **preserve_state_meta_fields(state),
    }
    
    # Only override locale if new_plan provides a valid value, otherwise use preserved locale
    if new_plan.get("locale"):
        update_dict["locale"] = new_plan["locale"]
    
    return Command(
        update=update_dict,
        goto=goto,
    )


def coordinator_node(
    state: State, config: RunnableConfig
) -> Command[
    Literal["planner", "background_investigator", "coordinator", "get_content", "__end__"]
]:
    logger.info("Coordinator talking.")

    # 1. ESG 总控阶段（来自 run_agent_workflow_async 的第一个 graph）
    if state.get("workflow_stage") == "esg_prepare":
        logger.info("Detected ESG writing task → routing to get_content")

        messages = state.get("messages", [])
        user_msg = messages[-1].content.strip()

        m1 = re.search(r"report_id\s*=\s*(\d+)", user_msg)
        m2 = re.search(r"tenant_id\s*=\s*(\d+)", user_msg)
        m3 = re.search(r"title_id\s*=\s*(\d+)", user_msg)
        m4 = re.search(r"title_text\s*=\s*([^,]+)", user_msg)

        report_id = int(m1.group(1)) if m1 else None
        tenant_id = int(m2.group(1)) if m2 else None
        title_id = int(m3.group(1)) if m3 else None
        title_text = m4.group(1).strip() if m4 else ""

        logger.info(
            f"Parsed ESG args: report_id={report_id}, tenant_id={tenant_id}, title_id={title_id}"
        )

        return Command(
            update={
                "task_type": "esg_write",
                "esg_report_id": report_id,
                "esg_tenant_id": tenant_id,
                "esg_title_id": title_id,
                "esg_title_text": title_text,
                "raw_esg_input": user_msg,
            },
            goto="get_content",
        )

    # 2. ESG 分段写作阶段（来自 write_esg_report_part）
    if state.get("workflow_stage") == "esg_part_write":
        logger.info("Detected ESG part writing stage → skip get_content")
        return Command(
            update={},
            goto="process_single_title",  
        )

def get_content_node(state: State, config: RunnableConfig):
    report_id = state.get("esg_report_id")
    title_id = state.get("esg_title_id")
    tenant_id = state.get("esg_tenant_id")

    text_results, company_name = get_title_contents(report_id, tenant_id, title_id)
    return Command(
        update={
            "esg_company_name": company_name,
            "esg_title_datas": text_results,
            "goto": "check_title_tag",
        },
        goto="check_title_tag",
    )

async def check_title_tag_node(state: State, config: RunnableConfig):
    title_datas = state.get("esg_title_datas", [])
    report_id = state.get("esg_report_id")
    tenant_id = state.get("esg_tenant_id")
    
    title_in_framework = await title_framework(title_datas[0]['title'],
                                                   [title_datas[i]['title'] for i in range(1, len(title_datas))],
                                                   [title_datas[i]['topic_info_rules'] for i in range(len(title_datas))])
    writing_tag = title_in_framework['title_in_framework']
    logger.info(f"writing_tag： {writing_tag}")
    if writing_tag not in ['董事会声明', '管理层致辞', '关于#企业#', '关于本报告']:
        title_ids = []
        for title_data in title_datas:
            if title_data['title_id'] not in title_ids:
                title_ids.append(title_data['title_id'])
        title_contents = []
        for title_id in title_ids:
            title_content = {}
            title_content['report_id'] = report_id
            title_content['tenant_id'] = tenant_id
            title_content['title_id'] = title_id
            title_content['sub_title'] = []
            for title_data in title_datas:
                if title_data['title_id'] == title_id:
                    title_content['title'] = title_data['title']
                    title_content['parent_id'] = title_data['parent_id']
                    title_content['topic_info_rules'] = title_data['topic_info_rules']
                    sub_title = {'sub_title_name': title_data['sub_title_name'],
                                    'sub_title_content': title_data['sub_title_content'],
                                    'sub_title_topic': title_data['sub_title_topic'],
                                    'sub_title_raw_data': title_data['sub_title_raw_data'] if title_data['sub_title_raw_data'] else ''}
                    title_content['sub_title'].append(sub_title)
            title_contents.append(title_content)
        print(title_contents)
        return Command(
            update={
                "esg_title_contents_raw": title_contents,
                "esg_writing_tag": writing_tag,
            },
            goto="industrial_researcher"
        )
    else:
        title_contents = []
        return Command(
            update = {
                "esg_writing_tag": writing_tag,
            },
            goto="write_introductory"
        )

async def write_introductory_node(state: State, config: RunnableConfig):
    title_datas = state.get("esg_title_datas", [])
    writing_tag = state.get("esg_writing_tag")
    company_name = state.get("esg_company_name")
    report_id = state.get("esg_report_id")
    tenant_id = state.get("esg_tenant_id")
    content = ''
    for title_data in title_datas:
        content += title_data['content'] + '/n'
    writing = await main_report_writing_begining(title_datas[0]['title'], writing_tag, company_name, content)
    result = []
    exist_title = []
    for i, title_data in enumerate(title_datas):
        result_i = {}
        result_i['report_id'] = report_id
        result_i['tenant_id'] = tenant_id
        result_i['title_id'] = title_data['title_id']
        result_i['title'] = title_data['title']
        result_i['parent_id'] = title_data['parent_id']
        if i == 0:
            result_i['writing_content'] = writing
        else:
            result_i['writing_content']= ''
        if result_i['title_id'] not in exist_title:
            result.append(result_i)
            exist_title.append(result_i['title_id'])
    return Command(
        update={
            "esg_part_report_result": result
        },
        goto="__end__"
    )

def reporter_node(state: State, config: RunnableConfig):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    configurable = Configuration.from_runnable_config(config)
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "en-US"),
    }
    invoke_messages = apply_prompt_template("reporter", input_, configurable, input_.get("locale", "en-US"))
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
            name="system",
        )
    )

    observation_messages = []
    for observation in observations:
        observation_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )

    # Context compression
    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP["reporter"])
    compressed_state = ContextManager(llm_token_limit).compress_messages(
        {"messages": observation_messages}
    )
    invoke_messages += compressed_state.get("messages", [])

    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm_by_type(AGENT_LLM_MAP["reporter"]).invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}

def researcher_node(state: State, config: RunnableConfig):
    pass

def research_team_node(state: State):
    """Research team node that collaborates on tasks."""
    logger.info("Research team is collaborating on tasks.")
    logger.debug("Entering research_team_node - coordinating research and coder agents")
    pass


def validate_web_search_usage(messages: list, agent_name: str = "agent") -> bool:
    """
    Validate if the agent has used the web search tool during execution.
    
    Args:
        messages: List of messages from the agent execution
        agent_name: Name of the agent (for logging purposes)
        
    Returns:
        bool: True if web search tool was used, False otherwise
    """
    web_search_used = False
    
    for message in messages:
        # Check for ToolMessage instances indicating web search was used
        if isinstance(message, ToolMessage) and message.name == "web_search":
            web_search_used = True
            logger.info(f"[VALIDATION] {agent_name} received ToolMessage from web_search tool")
            break
            
        # Check for AIMessage content that mentions tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('name') == "web_search":
                    web_search_used = True
                    logger.info(f"[VALIDATION] {agent_name} called web_search tool")
                    break
            # break outer loop if web search was used
            if web_search_used:
                break
                    
        # Check for message name attribute
        if hasattr(message, 'name') and message.name == "web_search":
            web_search_used = True
            logger.info(f"[VALIDATION] {agent_name} used web_search tool")
            break
    
    if not web_search_used:
        logger.warning(f"[VALIDATION] {agent_name} did not use web_search tool")
        
    return web_search_used


async def _execute_agent_step(
    state: State, agent, agent_name: str, config: RunnableConfig = None
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    logger.debug(f"[_execute_agent_step] Starting execution for agent: {agent_name}")
    
    current_plan = state.get("current_plan")
    plan_title = current_plan.title
    observations = state.get("observations", [])
    logger.debug(f"[_execute_agent_step] Plan title: {plan_title}, observations count: {len(observations)}")

    # Find the first unexecuted step
    current_step = None
    completed_steps = []
    for idx, step in enumerate(current_plan.steps):
        if not step.execution_res:
            current_step = step
            logger.debug(f"[_execute_agent_step] Found unexecuted step at index {idx}: {step.title}")
            break
        else:
            completed_steps.append(step)

    if not current_step:
        logger.warning(f"[_execute_agent_step] No unexecuted step found in {len(current_plan.steps)} total steps")
        return Command(
            update=preserve_state_meta_fields(state),
            goto="research_team"
        )

    logger.info(f"[_execute_agent_step] Executing step: {current_step.title}, agent: {agent_name}")
    logger.debug(f"[_execute_agent_step] Completed steps so far: {len(completed_steps)}")

    # Format completed steps information
    completed_steps_info = ""
    if completed_steps:
        completed_steps_info = "# Completed Research Steps\n\n"
        for i, step in enumerate(completed_steps):
            completed_steps_info += f"## Completed Step {i + 1}: {step.title}\n\n"
            completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

    # Prepare the input for the agent with completed steps info
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"# Research Topic\n\n{plan_title}\n\n{completed_steps_info}# Current Step\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }

    # Add citation reminder for researcher agent
    if agent_name == "researcher":
        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

    # Invoke the agent
    default_recursion_limit = 25
    try:
        env_value_str = os.getenv("AGENT_RECURSION_LIMIT", str(default_recursion_limit))
        parsed_limit = int(env_value_str)

        if parsed_limit > 0:
            recursion_limit = parsed_limit
            logger.info(f"Recursion limit set to: {recursion_limit}")
        else:
            logger.warning(
                f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit
    except ValueError:
        raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
        logger.warning(
            f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
            f"Using default value {default_recursion_limit}."
        )
        recursion_limit = default_recursion_limit

    logger.info(f"Agent input: {agent_input}")
    
    # Validate message content before invoking agent
    try:
        validated_messages = validate_message_content(agent_input["messages"])
        agent_input["messages"] = validated_messages
    except Exception as validation_error:
        logger.error(f"Error validating agent input messages: {validation_error}")
    
    # Apply context compression to prevent token overflow (Issue #721)
    llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_name])
    if llm_token_limit:
        token_count_before = sum(
            len(str(msg.content).split()) for msg in agent_input.get("messages", []) if hasattr(msg, "content")
        )
        compressed_state = ContextManager(llm_token_limit, preserve_prefix_message_count=3).compress_messages(
            {"messages": agent_input["messages"]}
        )
        agent_input["messages"] = compressed_state.get("messages", [])
        token_count_after = sum(
            len(str(msg.content).split()) for msg in agent_input.get("messages", []) if hasattr(msg, "content")
        )
        logger.info(
            f"Context compression for {agent_name}: {len(compressed_state.get('messages', []))} messages, "
            f"estimated tokens before: ~{token_count_before}, after: ~{token_count_after}"
        )
    
    try:
        result = await agent.ainvoke(
            input=agent_input, config={"recursion_limit": recursion_limit}
        )
    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = f"Error executing {agent_name} agent for step '{current_step.title}': {str(e)}"
        logger.exception(error_message)
        logger.error(f"Full traceback:\n{error_traceback}")
        
        # Enhanced error diagnostics for content-related errors
        if "Field required" in str(e) and "content" in str(e):
            logger.error(f"Message content validation error detected")
            for i, msg in enumerate(agent_input.get('messages', [])):
                logger.error(f"Message {i}: type={type(msg).__name__}, "
                            f"has_content={hasattr(msg, 'content')}, "
                            f"content_type={type(msg.content).__name__ if hasattr(msg, 'content') else 'N/A'}, "
                            f"content_len={len(str(msg.content)) if hasattr(msg, 'content') and msg.content else 0}")

        detailed_error = f"[ERROR] {agent_name.capitalize()} Agent Error\n\nStep: {current_step.title}\n\nError Details:\n{str(e)}\n\nPlease check the logs for more information."
        current_step.execution_res = detailed_error

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=detailed_error,
                        name=agent_name,
                    )
                ],
                "observations": observations + [detailed_error],
                **preserve_state_meta_fields(state),
            },
            goto="research_team",
        )

    # Process the result
    response_content = result["messages"][-1].content
    
    # Sanitize response to remove extra tokens and truncate if needed
    response_content = sanitize_tool_response(str(response_content))
    
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Validate web search usage for researcher agent if enforcement is enabled
    web_search_validated = True
    should_validate = agent_name == "researcher"
    validation_info = ""

    if should_validate:
        # Check if enforcement is enabled in configuration
        configurable = Configuration.from_runnable_config(config) if config else Configuration()
        if configurable.enforce_researcher_search:
            web_search_validated = validate_web_search_usage(result["messages"], agent_name)
            
            # If web search was not used, add a warning to the response
            if not web_search_validated:
                logger.warning(f"[VALIDATION] Researcher did not use web_search tool. Adding reminder to response.")
                # Add validation information to observations
                validation_info = (
                    "\n\n[WARNING] This research was completed without using the web_search tool. "
                    "Please verify that the information provided is accurate and up-to-date."
                    "\n\n[VALIDATION WARNING] Researcher did not use the web_search tool as recommended."
                )

    # Update the step with the execution result
    current_step.execution_res = response_content
    logger.info(f"Step '{current_step.title}' execution completed by {agent_name}")

    # Include all messages from agent result to preserve intermediate tool calls/results
    # This ensures multiple web_search calls all appear in the stream, not just the final result
    agent_messages = result.get("messages", [])
    logger.debug(
        f"{agent_name.capitalize()} returned {len(agent_messages)} messages. "
        f"Message types: {[type(msg).__name__ for msg in agent_messages]}"
    )
    
    # Count tool messages for logging
    tool_message_count = sum(1 for msg in agent_messages if isinstance(msg, ToolMessage))
    if tool_message_count > 0:
        logger.info(
            f"{agent_name.capitalize()} agent made {tool_message_count} tool calls. "
            f"All tool results will be preserved and streamed to frontend."
        )

    return Command(
        update={
            "messages": agent_messages,
            "observations": observations + [response_content + validation_info],
            **preserve_state_meta_fields(state),
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
    state: State,
    config: RunnableConfig,
    agent_type: str,
    default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic for both researcher_node and coder_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_type: The type of agent ("researcher" or "coder")
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}
    
    # Get locale from workflow state to pass to agent creation
    # This fixes issue #743 where locale was not correctly retrieved in agent prompt
    locale = state.get("locale", "en-US")

    # Extract MCP server configuration for this agent type
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                server_config["enabled_tools"]
                and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env", "headers")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name

    # Create and execute agent with MCP tools if available
    if mcp_servers:
        client = MultiServerMCPClient(mcp_servers)
        loaded_tools = default_tools[:]
        all_tools = await client.get_tools()
        for tool in all_tools:
            if tool.name in enabled_tools:
                tool.description = (
                    f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
                )
                loaded_tools.append(tool)

        llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
        pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)
        agent = create_agent(
            agent_type,
            agent_type,
            loaded_tools,
            agent_type,
            pre_model_hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
            locale=locale,
        )
        return await _execute_agent_step(state, agent, agent_type, config)
    else:
        # Use default tools if no MCP servers are configured
        llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
        pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)
        agent = create_agent(
            agent_type,
            agent_type,
            default_tools,
            agent_type,
            pre_model_hook,
            interrupt_before_tools=configurable.interrupt_before_tools,
            locale=locale,
        )
        return await _execute_agent_step(state, agent, agent_type, config)


async def industrial_researcher_node(state: State, config: RunnableConfig, model='qwen-max') -> Command:
    logger.info("[researcher_node] Start ESG research")
    company_name = state.get("esg_company_name")
    
    llm = get_llm_by_type("basic")
    parser = JsonOutputParser(pydantic_object=IndustaryInfoRules)
    format_instructions = parser.get_format_instructions()

    system_prompt = f"""
    任务：根据公司名称和网络搜索主营业务文本，总结公司所属行业，输出公司所属行业和主营业务，如搜索到相应的上市公司主体名称和对应行业和主营业务，如’金蝶‘对应‘金蝶国际软件集团有限公司’再寻找主营业务和行业软件业，如未搜索到所属行业和业务信息，输出为：无特点行业。

    你需要按照以下格式返回结果：
    {format_instructions}
    """
    user_prompt = f"""以下是公司名称：{company_name}"""
    company_industary_info_result = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    try:
        parsed = parser.parse(company_industary_info_result.content)
        print("✅ JSON 解析成功，返回模型对象")
        print("解析结果：", parsed)
        # 强制转为模型对象（确保返回类型一致）
        if isinstance(parsed, dict):
            company_info = IndustaryInfoRules(**parsed)
    except Exception as e:
        print(f"⚠️ JSON 解析失败，返回原始内容: {e}")
        print("原始返回内容：", company_industary_info_result)
        # 兜底策略：直接返回空结构
        try:
            company_industary_info = parser.parse(company_industary_info_result.content)
            company_info = IndustaryInfoRules(industry=company_industary_info['industry'],
                                      main_business=company_industary_info['main_business'])
            print("尝试从解析后的内容中构建模型对象")
        except:
            company_info = IndustaryInfoRules(industry='', main_business='')
            print("最终返回空结构")
    company_information_texts = getattr(company_info, "industry", "") or ""
    return Command(
        update={
            "esg_company_industry": company_information_texts,
            "goto": "__end__",
        },
        goto="__end__",
    )

async def process_single_title_node(state: State) -> Command:
    """
    Node: Process a single ESG title content.
    Responsibility:
    - Extract title / raw_text / sub_title_topics
    - Update state with prepared writing inputs
    - Route to get_law_info node
    """
    logger.info("[process_single_title_node] Start processing single title")

    # 1. 从 state 里取 workflow 已经喂进来的数据
    title_content = state.get("esg_title_contents_raw")

    if not title_content:
        logger.warning("No esg_title_content found in state")
        return Command(goto="__end__")

    # 2. 提取 title / raw_text / sub_title_topics
    title = title_content.get("title")

    raw_text = ""
    sub_title_topics = set()

    for sub_title in title_content.get("sub_title", []):
        if sub_title.get("sub_title_raw_data"):
            raw_text += sub_title["sub_title_raw_data"] + "\n"
        else:
            raw_text += sub_title.get("sub_title_content", "")

        topic = sub_title.get("sub_title_topic")
        if topic:
            sub_title_topics.add(topic)

    logger.info(
        f"[process_single_title_node] title={title}, topics={sub_title_topics}"
    )

    # 3. 把“下一阶段需要的信息”写入 state
    return Command(
        update={
            # 写作输入（供后续 node 使用）
            "current_title": title,
            "current_raw_text": raw_text,
            "current_sub_title_topics": list(sub_title_topics),
        },
        goto="get_laws_info",
    )


async def get_laws_info_node(state: State) -> Command:
    """
    Node: Get laws and regulations information.
    """
    logger.info("[get_laws_info_node] Start getting laws information")

    sub_title_topics = state.get("current_sub_title_topics", [])
    company_information_texts = state.get("esg_company_industry", "")

    if not sub_title_topics:
        logger.warning("No sub_title_topics found in state")
        return Command(goto="__end__")

    llm = get_llm_by_type("basic")

    topics_text = "；".join(sub_title_topics)

    user_prompt = (
        f"请基于以下主题：{topics_text}，并结合企业行业背景："
        f"{company_information_texts}，"
        f"梳理国内外相关法律法规、条例或管理办法中"
        f"与这些主题直接相关的具体条目和内容。"
        f"请用中文回答，结构清晰。"
    )

    law_info_result = await llm.ainvoke(
        [{"role": "user", "content": user_prompt}]
    )
    return Command(
        update={
            "law_info_text": law_info_result.content,
        },
        goto="esg_write_report",  # 或下一个 node，如 "write_report_part"
    )


async def esg_write_report_node(state: State) -> Command:
    """
    Node: Write ESG report content for a single title.
    - Use prompt from src/prompts/write_report.md
    - Stream LLM output
    - Parse JSON result with ReportInfoRules
    """
    llm = get_llm_by_type("reasoning")
    parser = JsonOutputParser(pydantic_object=ReportInfoRules)
    format_instructions = parser.get_format_instructions()

    data_text = state.get("current_raw_text", "")
    laws_info = state.get("law_info_text", "")


    prompt_state = {
        **state,
        "format_instructions": format_instructions,
        "data_text": data_text,
        "laws_info": laws_info,
    }

    messages = apply_prompt_template(
        "write_report",          
        prompt_state,
        configurable=None,
        locale=state.get("locale", "zh-CN"),
    )

    output_text = ""
    logger.info("📘 开始生成报告内容（流式输出）")

    async for chunk in llm.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            output_text += chunk.content


    report_result = parser.parse(output_text)
    return Command(
        update={
            "esg_part_report_result": report_result,
        },
        goto="create_table_of_contents",  
    )

async def create_table_of_contents_node(state: State) -> Command:
    sub_title_topics = state.get("current_sub_title_topics", [])
    report_id = state.get("esg_report_id")
    tenant_id = state.get("esg_tenant_id")

    tables = []
    connection = get_connection()
    with connection.cursor() as cursor:
        for sub_title_topic in sub_title_topics:
            if sub_title_topic.upper() != 'SASB':
                table = []
                sql_indicator_topic = f'''SELECT DISTINCT indicator_name FROM t_esg_indicator 
                WHERE report_id = {abs(report_id)} AND tenant_id = {tenant_id} and deleted = 0 
                AND topic_tags LIKE '%{sub_title_topic}%'
                '''
                cursor.execute(sql_indicator_topic)
                raw_indicator_results = cursor.fetchall()

                if not raw_indicator_results:
                    print(f"⚠️ 未找到匹配议题：{sub_title_topics}")
                    continue

                indicator_results = [list(d.values())[0] for d in raw_indicator_results]

                for indicator_result in indicator_results:
                    sql_data = f'''SELECT * FROM t_esg_indicator_data 
                    WHERE report_id = {abs(report_id)} AND tenant_id = {tenant_id} and deleted = 0 AND name = '{indicator_result}'
                    '''
                    cursor.execute(sql_data)
                    raw_data_results = cursor.fetchall()

                    if not raw_data_results:
                        print(f"⚠️ 议题 {indicator_result} 无数据")
                        continue

                    for data_result in raw_data_results:
                        table.append({'name': indicator_result, 'year': data_result['year'], 'value': data_result['value']})

                if not table:
                    print(f"⚠️ 议题 {sub_title_topic} 无可用表格数据")
                    return ""

                years = sorted(set(item['year'] for item in table), reverse=True)
                table_data = defaultdict(dict)

                def normalize_value(value):
                    if value is None:
                        return ""
                    if isinstance(value, (int, float, Decimal)):
                        return f"{float(value):.2f}"
                    return str(value)

                for item in table:
                    table_data[item['name']][item['year']] = normalize_value(item.get('value'))

                header = "| 指标名称 | " + " | ".join(str(y) for y in years) + " |"
                separator = "|-----------|" + "|".join(["-----------"] * len(years)) + "|"

                rows = []
                for name, values in table_data.items():
                    row = [name] + [values.get(y, "") for y in years]
                    rows.append("| " + " | ".join(row) + " |")

                markdown_table = "\n".join([header, separator] + rows)
                print(f"————————————{sub_title_topic}————————————")
                print(markdown_table)
                tables.append(markdown_table)
    connection.close()
    return Command(
        update={
            "esg_part_tables_markdown": tables,
        },
        goto="__end__",
    )




async def coder_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Coder node that do code analysis."""
    logger.info("Coder node is coding.")
    logger.debug(f"[coder_node] Starting coder agent with python_repl_tool")
    
    return await _setup_and_execute_agent_step(
        state,
        config,
        "coder",
        [python_repl_tool],
    )


async def analyst_node(
    state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Analyst node that performs reasoning and analysis without code execution.
    
    This node handles tasks like:
    - Cross-validating information from multiple sources
    - Synthesizing research findings
    - Comparative analysis
    - Pattern recognition and trend analysis
    - General reasoning tasks that don't require code
    """
    logger.info("Analyst node is analyzing.")
    logger.debug(f"[analyst_node] Starting analyst agent for reasoning/analysis tasks")
    
    # Analyst uses no tools - pure LLM reasoning
    return await _setup_and_execute_agent_step(
        state,
        config,
        "analyst",
        [],  # No tools - pure reasoning
    )