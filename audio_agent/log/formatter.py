"""
Markdown formatting utilities for run logging.

Provides functions to format various data structures as Markdown content.
"""

from datetime import datetime
from typing import Any


def format_metadata(
    timestamp: datetime,
    question: str,
    status: str,
    step_count: int,
    log_file: str,
) -> str:
    """Format run metadata as Markdown."""
    lines = [
        "## Metadata",
        "",
        f"- **Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Question**: {question}",
        f"- **Status**: {status}",
        f"- **Steps**: {step_count}",
        f"- **Log File**: {log_file}",
        "",
    ]
    return "\n".join(lines)


def format_input_section(original_audios: list[str], temp_dir: str) -> str:
    """Format input section as Markdown."""
    lines = [
        "## Input",
        "",
    ]
    
    # Handle multiple audio files
    if isinstance(original_audios, list) and original_audios:
        if len(original_audios) == 1:
            lines.append(f"- **Original Audio**: {original_audios[0]}")
        else:
            lines.append(f"- **Original Audios** ({len(original_audios)} files):")
            for i, audio_path in enumerate(original_audios):
                lines.append(f"  - audio_{i}: {audio_path}")
    elif isinstance(original_audios, str):
        # Backward compatibility for single string
        lines.append(f"- **Original Audio**: {original_audios}")
    else:
        lines.append(f"- **Original Audio**: Unknown")
    
    lines.append(f"- **Temp Directory**: {temp_dir}")
    lines.append("")
    return "\n".join(lines)


def format_question_oriented_prompt(prompt: str | None) -> str:
    """Format question-oriented prompt as Markdown."""
    if not prompt:
        return "## Question-Oriented Prompt\n\n*No question-oriented prompt generated*\n\n"
    
    lines = [
        "## Question-Oriented Prompt",
        "",
        "```",
        prompt,
        "```",
        "",
    ]
    return "\n".join(lines)


def format_frontend_output(caption: str | None) -> str:
    """Format frontend output as Markdown."""
    if not caption:
        return "## Frontend Output\n\n*No frontend output*\n\n"
    
    lines = [
        "## Frontend Output",
        "",
        "```",
        caption,
        "```",
        "",
    ]
    return "\n".join(lines)


def format_initial_plan(plan: Any) -> str:
    """Format InitialPlan as Markdown table."""
    if not plan:
        return "## Initial Plan\n\n*No initial plan*\n\n"
    
    lines = [
        "## Initial Plan",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    
    # Add each field as a row
    fields = [
        ("Approach", getattr(plan, 'approach', 'N/A')),
        ("Clarified Intent", getattr(plan, 'clarified_intent', 'N/A') or 'N/A'),
        ("Expected Output Format", getattr(plan, 'expected_output_format', 'N/A') or 'N/A'),
        ("Requires Audio Output", str(getattr(plan, 'requires_audio_output', False))),
        ("Notes", getattr(plan, 'notes', 'N/A') or 'N/A'),
    ]
    
    for field, value in fields:
        # Escape pipe characters in value
        value_str = str(value).replace('|', '\\|')
        lines.append(f"| {field} | {value_str} |")
    
    # Add focus points
    focus_points = getattr(plan, 'focus_points', [])
    if focus_points:
        lines.append(f"| Focus Points | {', '.join(focus_points)} |")
    
    # Add possible tool types
    tool_types = getattr(plan, 'possible_tool_types', [])
    if tool_types:
        lines.append(f"| Possible Tool Types | {', '.join(tool_types)} |")
    
    lines.append("")
    
    # Add detailed plan if present
    detailed_plan = getattr(plan, 'detailed_plan', [])
    if detailed_plan:
        lines.append("### Detailed Execution Plan")
        lines.append("")
        lines.append("| Step | Description | Tool Type | Expected Output |")
        lines.append("|------|-------------|-----------|-----------------|")
        
        for step in detailed_plan:
            step_num = getattr(step, 'step_number', 0)
            description = getattr(step, 'description', '')
            tool_type = getattr(step, 'tool_type', '') or '-'
            expected = getattr(step, 'expected_output', '') or '-'
            
            # Escape pipe characters and truncate if needed
            description = str(description).replace('|', '\\|')[:80]
            tool_type = str(tool_type).replace('|', '\\|')[:20]
            expected = str(expected).replace('|', '\\|')[:40]
            
            lines.append(f"| {step_num} | {description} | {tool_type} | {expected} |")
        
        lines.append("")
    
    return "\n".join(lines)


def format_evidence_log(evidence_log: list[Any]) -> str:
    """Format evidence log as Markdown."""
    if not evidence_log:
        return "## Evidence Log\n\n*No evidence collected*\n\n"
    
    lines = ["## Evidence Log", ""]
    
    for i, item in enumerate(evidence_log, 1):
        source = getattr(item, 'source', 'Unknown')
        evidence_type = getattr(item, 'evidence_type', 'text')
        content = getattr(item, 'content', '')
        confidence = getattr(item, 'confidence', 0.0)
        timestamp = getattr(item, 'timestamp', None)
        
        lines.append(f"### Evidence {i}: {source}")
        lines.append("")
        lines.append(f"- **Source**: {source}")
        lines.append(f"- **Type**: {evidence_type}")
        lines.append(f"- **Confidence**: {confidence:.2f}")
        if timestamp:
            lines.append(f"- **Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("**Content**:")
        lines.append("```")
        # Truncate very long content
        content_str = str(content)
        if len(content_str) > 2000:
            content_str = content_str[:2000] + "\n... (truncated)"
        lines.append(content_str)
        lines.append("```")
        lines.append("")
    
    return "\n".join(lines)


def format_tool_call_history(tool_history: list[Any]) -> str:
    """Format tool call history as Markdown."""
    if not tool_history:
        return "## Tool Call History\n\n*No tools called*\n\n"
    
    lines = ["## Tool Call History", ""]
    
    for i, record in enumerate(tool_history, 1):
        request = getattr(record, 'request', None)
        result = getattr(record, 'result', None)
        step_number = getattr(record, 'step_number', i - 1)
        
        tool_name = getattr(request, 'tool_name', 'Unknown') if request else 'Unknown'
        args = getattr(request, 'args', {}) if request else {}
        success = getattr(result, 'success', False) if result else False
        
        lines.append(f"### Tool Call {i}: {tool_name}")
        lines.append("")
        lines.append(f"- **Step**: {step_number}")
        lines.append(f"- **Tool**: {tool_name}")
        lines.append(f"- **Success**: {success}")
        lines.append("")
        
        # Format args
        if args:
            lines.append("**Arguments**:")
            lines.append("```json")
            import json
            try:
                lines.append(json.dumps(args, indent=2, default=str))
            except Exception:
                lines.append(str(args))
            lines.append("```")
            lines.append("")
        
        # Format result output
        if result:
            output = getattr(result, 'output', {})
            error_message = getattr(result, 'error_message', None)
            execution_time_ms = getattr(result, 'execution_time_ms', 0.0)
            
            lines.append(f"- **Execution Time**: {execution_time_ms:.2f}ms")
            
            if error_message:
                lines.append(f"- **Error**: {error_message}")
            
            if output:
                lines.append("")
                lines.append("**Output**:")
                lines.append("```json")
                import json
                try:
                    # Filter out very large content
                    display_output = {}
                    for key, value in output.items():
                        if key == 'content' and isinstance(value, list):
                            display_output[key] = f"[{len(value)} items]"
                        elif isinstance(value, str) and len(value) > 500:
                            display_output[key] = value[:500] + "... (truncated)"
                        else:
                            display_output[key] = value
                    lines.append(json.dumps(display_output, indent=2, default=str))
                except Exception:
                    lines.append(str(output))
                lines.append("```")
        
        lines.append("")
    
    return "\n".join(lines)


def format_planner_trace(planner_trace: list[Any]) -> str:
    """Format planner trace as Markdown."""
    if not planner_trace:
        return "## Planner Trace\n\n*No planner decisions*\n\n"
    
    lines = ["## Planner Trace", ""]
    
    for i, decision in enumerate(planner_trace, 1):
        action = getattr(decision, 'action', 'Unknown')
        rationale = getattr(decision, 'rationale', '')
        tool_calls = getattr(decision, 'selected_tool_calls', None) or []
        draft_answer = getattr(decision, 'draft_answer', None)
        confidence = getattr(decision, 'confidence', 0.0)

        lines.append(f"### Decision {i}: {action}")
        lines.append("")
        lines.append(f"- **Action**: {action}")
        lines.append(f"- **Confidence**: {confidence:.2f}")

        if tool_calls:
            tool_names = ", ".join(getattr(tc, "tool_name", "?") for tc in tool_calls)
            lines.append(f"- **Tool(s)** ({len(tool_calls)}): {tool_names}")
            for j, tc in enumerate(tool_calls, 1):
                tc_args = getattr(tc, "args", {}) or {}
                lines.append(f"  - Call {j}: `{getattr(tc, 'tool_name', '?')}({tc_args})`")
        
        lines.append("")
        lines.append("**Rationale**:")
        lines.append(f"> {rationale}")
        
        if draft_answer:
            lines.append("")
            lines.append("**Draft Answer**:")
            lines.append("```")
            # Truncate very long answers
            answer_str = str(draft_answer)
            if len(answer_str) > 1000:
                answer_str = answer_str[:1000] + "\n... (truncated)"
            lines.append(answer_str)
            lines.append("```")
        
        lines.append("")
    
    return "\n".join(lines)


def format_final_answer(final_answer: Any) -> str:
    """Format FinalAnswer as Markdown."""
    if not final_answer:
        return "## Final Answer\n\n*No final answer*\n\n"
    
    lines = ["## Final Answer", ""]
    
    answer = getattr(final_answer, 'answer', '')
    confidence = getattr(final_answer, 'confidence', 0.0)
    evidence_summary = getattr(final_answer, 'evidence_summary', '')
    reasoning_trace = getattr(final_answer, 'reasoning_trace', '')
    rationale = getattr(final_answer, 'rationale', None)
    output_audio = getattr(final_answer, 'output_audio', None)
    timestamp = getattr(final_answer, 'timestamp', None)
    
    if timestamp:
        lines.append(f"- **Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    lines.append(f"- **Confidence**: {confidence:.2f}")
    lines.append("")
    lines.append("**Answer**:")
    lines.append("```")
    lines.append(answer)
    lines.append("```")
    
    if rationale:
        lines.append("")
        lines.append("**Rationale**:")
        lines.append("```")
        rationale_str = str(rationale)
        if len(rationale_str) > 2000:
            rationale_str = rationale_str[:2000] + "\n... (truncated)"
        lines.append(rationale_str)
        lines.append("```")
    
    if output_audio:
        lines.append("")
        lines.append("**Output Audio**:")
        audio_id = getattr(output_audio, 'audio_id', 'Unknown')
        path = getattr(output_audio, 'path', 'Unknown')
        description = getattr(output_audio, 'description', '')
        lines.append(f"- **ID**: {audio_id}")
        lines.append(f"- **Path**: {path}")
        lines.append(f"- **Description**: {description}")
    
    if evidence_summary:
        lines.append("")
        lines.append("**Evidence Summary**:")
        lines.append("```")
        lines.append(evidence_summary)
        lines.append("```")
    
    if reasoning_trace:
        lines.append("")
        lines.append("**Reasoning Trace**:")
        lines.append("```")
        lines.append(reasoning_trace)
        lines.append("```")
    
    lines.append("")
    return "\n".join(lines)


def format_audio_list(audio_list: list[Any]) -> str:
    """Format audio list as Markdown table."""
    if not audio_list:
        return "## Audio Files Generated\n\n*No audio files*\n\n"
    
    lines = [
        "## Audio Files Generated",
        "",
        "| ID | Source | Path | Description |",
        "|------|--------|------|-------------|",
    ]
    
    for audio in audio_list:
        audio_id = getattr(audio, 'audio_id', 'Unknown')
        source = getattr(audio, 'source', 'Unknown')
        path = getattr(audio, 'path', 'Unknown')
        description = getattr(audio, 'description', '')
        
        # Escape pipe characters
        path = path.replace('|', '\\|')
        description = description.replace('|', '\\|')
        
        lines.append(f"| {audio_id} | {source} | {path} | {description} |")
    
    lines.append("")
    return "\n".join(lines)


def format_evidence_summary(evidence_summary: str | None) -> str:
    """Format evidence summary as Markdown."""
    if not evidence_summary:
        return "## Evidence Summary\n\n*No evidence summary generated*\n\n"
    
    lines = [
        "## Evidence Summary",
        "",
        "```",
        evidence_summary,
        "```",
        "",
    ]
    return "\n".join(lines)



def format_frontend_final_answer(final_answer: Any) -> str:
    """Format the frontend-generated final answer."""
    if not final_answer:
        return "## Frontend Final Answer\n\n*No frontend final answer generated*\n\n"

    lines = ["## Frontend Final Answer", ""]

    answer = getattr(final_answer, 'answer', '')
    confidence = getattr(final_answer, 'confidence', 0.0)
    rationale = getattr(final_answer, 'rationale', None)

    lines.append(f"- **Confidence**: {confidence:.2f}")
    lines.append("")
    lines.append("**Generated Answer**:")
    lines.append("```")
    answer_str = str(answer)
    if len(answer_str) > 2000:
        answer_str = answer_str[:2000] + "\n... (truncated)"
    lines.append(answer_str)
    lines.append("```")

    if rationale:
        lines.append("")
        lines.append("**Rationale**:")
        lines.append("```")
        rationale_str = str(rationale)
        if len(rationale_str) > 2000:
            rationale_str = rationale_str[:2000] + "\n... (truncated)"
        lines.append(rationale_str)
        lines.append("```")

    lines.append("")
    return "\n".join(lines)


def format_format_check_result(format_check_result: Any) -> str:
    """Format format check result as Markdown."""
    if not format_check_result:
        return "## Format Check Result\n\n*No format check performed*\n\n"
    
    lines = ["## Format Check Result", ""]
    
    passed = getattr(format_check_result, 'passed', False)
    critique = getattr(format_check_result, 'critique', None)
    confidence = getattr(format_check_result, 'confidence', 0.0)
    
    status = "✅ Passed" if passed else "❌ Failed"
    lines.append(f"- **Status**: {status}")
    lines.append(f"- **Confidence**: {confidence:.2f}")
    
    if critique:
        lines.append("")
        lines.append("**Format Critique**:")
        lines.append("```")
        lines.append(critique)
        lines.append("```")
    
    lines.append("")
    return "\n".join(lines)


def format_error(error_message: str | None) -> str:
    """Format error section as Markdown."""
    lines = ["## Errors", ""]
    
    if error_message:
        lines.append("```")
        lines.append(error_message)
        lines.append("```")
    else:
        lines.append("*No errors*")
    
    lines.append("")
    return "\n".join(lines)


def format_frontend_followup_history(planner_trace: list, evidence_log: list) -> str:
    """Format frontend follow-up calls by correlating planner decisions with evidence."""
    # Find all CALL_FRONTEND decisions
    followup_decisions = [
        (i, d) for i, d in enumerate(planner_trace)
        if getattr(d, 'action', None) == "call_frontend"
    ]
    
    if not followup_decisions:
        return "## Frontend Follow-Up History\n\n*No frontend follow-up calls made*\n\n"
    
    lines = ["## Frontend Follow-Up History", ""]
    
    for idx, (decision_idx, decision) in enumerate(followup_decisions, 1):
        # Find matching evidence items
        evidence_items = [
            e for e in evidence_log
            if getattr(e, 'evidence_type', None) == "frontend_followup"
        ]
        # Match by metadata audio_ids if available
        matched_evidence = None
        decision_audio_ids = getattr(decision, 'selected_audio_ids', []) or []
        for e in evidence_items:
            meta_audio_ids = (e.metadata or {}).get("selected_audio_ids", [])
            if meta_audio_ids == decision_audio_ids:
                matched_evidence = e
                break
        if not matched_evidence and idx <= len(evidence_items):
            matched_evidence = evidence_items[idx - 1]
        
        lines.append(f"### Follow-Up {idx}")
        lines.append(f"- **Audio IDs**: {', '.join(decision_audio_ids) if decision_audio_ids else 'N/A'}")
        lines.append(f"- **Goal**: {getattr(decision, 'frontend_followup_goal', None) or 'N/A'}")
        prompt = getattr(decision, 'frontend_followup_prompt', None) or "N/A"
        lines.append(f"- **Prompt**: `{prompt[:200]}{'...' if len(prompt) > 200 else ''}`")
        if matched_evidence:
            caption = getattr(matched_evidence, 'content', '')
            lines.append(f"- **Output**:")
            lines.append("```")
            lines.append(caption[:2000] + ("... (truncated)" if len(caption) > 2000 else ""))
            lines.append("```")
        lines.append("")
    
    return "\n".join(lines)


def sanitize_filename(question: str, max_length: int = 30) -> str:
    """
    Sanitize question for use in filename.
    
    - Lowercase
    - Replace spaces with underscores
    - Remove special characters
    - Limit length
    """
    import re
    
    # Lowercase and replace spaces
    sanitized = question.lower().replace(' ', '_')
    
    # Remove non-alphanumeric characters (except underscores)
    sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
    
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized or "run"
