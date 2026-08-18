"""
Generate HTML visualization for HealthBench attack evaluation results.
"""

import json
import html
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional

from med_red_team.healthbench.data import (
    _healthbench_model_id,
    healthbench_attack_strategy_config,
    healthbench_attack_strategy_order,
    healthbench_dataset_path,
)


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    return html.escape(str(text))


def format_text_preserve_newlines(text: str) -> str:
    """Format text preserving newlines as <br> tags."""
    if not text:
        return ""
    escaped = escape_html(text)
    return escaped.replace('\n', '<br>\n')


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length with suffix if needed."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def create_summary_stats(data: Dict[str, Any]) -> str:
    """Create summary statistics section."""
    summary = data.get('summary', {})
    results = data.get('results', [])

    total_cases = summary.get('total_cases', 0)

    # Handle missing optional fields gracefully.
    attacks_attempted = summary.get('attacks_attempted', None)
    attacks_applicable = summary.get('attacks_applicable', None)

    # If fields are missing, compute them from results
    if attacks_attempted is None:
        attacks_attempted = len(results)
    if attacks_applicable is None:
        attacks_applicable = sum(
            1 for r in results
            if r.get("attack_result") and r["attack_result"].get("applicable", False)
        )

    # Correctly compute attacks_triggered by checking if penalty rubric was met
    attacks_triggered = 0
    for result in results:
        attack_result = result.get('attack_result')
        if attack_result and attack_result.get('applicable'):
            attacked_grades = result.get('attacked_grades')
            if attacked_grades:  # Check if not None
                for grade in attacked_grades:
                    # Check if this is a penalty rubric (points < 0) and it was met
                    points = grade.get('points')
                    criteria_met = grade.get('criteria_met', False)
                    if isinstance(points, (int, float)) and points < 0 and criteria_met:
                        attacks_triggered += 1
                        break  # Only count once per case

    # Calculate percentages
    applicable_rate = (attacks_applicable / attacks_attempted * 100) if attacks_attempted > 0 else 0
    # Triggered rate based on applicable attacks
    triggered_rate = (attacks_triggered / attacks_applicable * 100) if attacks_applicable > 0 else 0

    return f"""
        <div class="summary-card">
            <h2>Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_cases}</div>
                    <div class="stat-label">Total Cases</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{attacks_attempted}</div>
                    <div class="stat-label">Attacks Attempted</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{attacks_applicable}</div>
                    <div class="stat-label">Attacks Applicable ({applicable_rate:.1f}%)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #ef4444;">{attacks_triggered}</div>
                    <div class="stat-label">Attacks Triggered ({triggered_rate:.1f}%)</div>
                </div>
            </div>
        </div>
    """


def format_grade(grade: Dict[str, Any]) -> str:
    """Format a single grade with details."""
    # Handle nested rubric_item structure
    rubric_item = grade.get('rubric_item', {})
    criterion = rubric_item.get('criterion', grade.get('criterion', 'N/A'))
    points = rubric_item.get('points', grade.get('points', 'N/A'))
    tags = rubric_item.get('tags', grade.get('tags', []))
    criteria_met = grade.get('criteria_met', False)
    explanation = grade.get('explanation', '')

    # Check if this is a negative points rubric (penalty)
    is_penalty = isinstance(points, (int, float)) and points < 0

    # Truncate long criterion text
    criterion_short = truncate_text(criterion, max_length=150)
    criterion_full = escape_html(criterion)

    tags_html = ""
    if tags:
        tags_str = ", ".join(tags)
        tags_html = f'<div class="grade-tags">{escape_html(tags_str)}</div>'

    status_class = "met" if criteria_met else "not-met"
    status_text = "✓ Met" if criteria_met else "✗ Not Met"

    # For penalty rubrics, invert the color logic
    # (criteria_met = True means penalty triggered, which is bad)
    if is_penalty:
        status_class = "met-bad" if criteria_met else "not-met-good"
        status_text = "⚠ Triggered (Penalty)" if criteria_met else "✓ Avoided"

    explanation_html = ""
    if explanation:
        explanation_escaped = escape_html(explanation)
        explanation_html = f'<div class="grade-explanation">{explanation_escaped}</div>'

    rubric_class = "penalty-rubric" if is_penalty else ""

    return f"""
        <div class="grade-item {status_class} {rubric_class}">
            <div class="grade-header">
                <span class="grade-status">{status_text}</span>
                <span class="grade-points">({points} pts)</span>
            </div>
            {tags_html}
            <details class="grade-criterion">
                <summary>{criterion_short}</summary>
                <div class="criterion-full">{criterion_full}</div>
            </details>
            {explanation_html}
        </div>
    """


def format_attack_info(attack_result: Dict[str, Any]) -> str:
    """Format attack information section."""
    if not attack_result:
        return '<div class="attack-info not-applicable">No attack applied.</div>'

    applicable = attack_result.get('applicable', False)
    if not applicable:
        return '<div class="attack-info not-applicable">Attack not applicable to this case.</div>'

    attack_strategy = attack_result.get('attack_strategy', 'Unknown')
    attack_metadata = attack_result.get('attack_metadata', {})
    error = attack_result.get('error')

    if error:
        return f'<div class="attack-info error">Error applying attack: {escape_html(error)}</div>'

    # Format attack metadata
    metadata_html = ""
    if attack_metadata:
        metadata_items = []
        for key, value in attack_metadata.items():
            if value:
                metadata_items.append(f"<p><strong>{escape_html(str(key).replace('_', ' ').title())}:</strong> <span class=\"highlight\">{escape_html(str(value))}</span></p>")
        if metadata_items:
            metadata_html = f'<div class="attack-details">{"".join(metadata_items)}</div>'

    return f"""
        <div class="attack-info">
            <h3>⚔️ Attack Applied: {escape_html(attack_strategy.replace('_', ' ').title())}</h3>
            {metadata_html}
        </div>
    """


def format_conversation_comparison(original_conv: List[Dict], modified_conv: List[Dict], changed_measurement: str) -> str:
    """Format original and modified conversations side-by-side with highlighting."""
    if not modified_conv:
        return ""

    # Find the differences
    comparison_html = '<div class="conversation-comparison">'
    comparison_html += '<div class="comparison-grid">'

    # Original conversation
    comparison_html += '<div class="comparison-column">'
    comparison_html += '<h4>📄 Original Conversation</h4>'
    comparison_html += '<div class="conversation-box original">'
    for msg in original_conv:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_content = format_text_preserve_newlines(content)
        comparison_html += f'<div class="message message-{role}"><div class="message-role">{escape_html(role).upper()}</div><div class="message-content">{formatted_content}</div></div>'
    comparison_html += '</div></div>'

    # Modified conversation with highlights
    comparison_html += '<div class="comparison-column">'
    comparison_html += '<h4>🔧 Modified Conversation</h4>'
    if changed_measurement:
        comparison_html += f'<div class="measurement-badge">Changed: <strong>{escape_html(changed_measurement)}</strong></div>'
    comparison_html += '<div class="conversation-box modified">'
    for msg in modified_conv:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_content = format_text_preserve_newlines(content)
        comparison_html += f'<div class="message message-{role}"><div class="message-role">{escape_html(role).upper()}</div><div class="message-content">{formatted_content}</div></div>'
    comparison_html += '</div></div>'

    comparison_html += '</div></div>'
    return comparison_html


def create_case_card(result: Dict[str, Any]) -> str:
    """Create HTML for a single case card."""
    case_id = result.get('test_case_id', 'Unknown')
    sample_number = result.get('sample_number', 0)
    baseline_score = result.get('baseline_score')
    attacked_score = result.get('attacked_score')
    attack_result = result.get('attack_result')
    skipped = result.get('skipped', False)
    skip_reason = result.get('skip_reason', '')

    # Get original conversation for comparison
    original_conversation = result.get('original_conversation', [])
    attacked_conversation = result.get('attacked_conversation', [])

    # Render original/attacked conversations so we can show them even when
    # the attack is not applicable (or when there is no side-by-side comparison).
    def _render_conversation(conv: List[Dict]) -> str:
        if not conv:
            return ""
        parts = ['<div class="conversation-box">']
        for msg in conv:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_content = format_text_preserve_newlines(content)
            parts.append(f'<div class="message message-{escape_html(role)}"><div class="message-role">{escape_html(role).upper()}</div><div class="message-content">{formatted_content}</div></div>')
        parts.append('</div>')
        return "".join(parts)

    original_conversation_html = _render_conversation(original_conversation)
    attacked_conversation_html = _render_conversation(attacked_conversation)

    # Get changed measurement info
    changed_measurement = ""
    if attack_result:
        attack_metadata = attack_result.get('attack_metadata', {})
        changed_measurement = attack_metadata.get('changed_measurement', '')

    # Attack info
    attack_html = format_attack_info(attack_result)

    # Create conversation comparison
    conversation_comparison_html = ""
    if original_conversation and attacked_conversation:
        conversation_comparison_html = format_conversation_comparison(
            original_conversation,
            attacked_conversation,
            changed_measurement
        )

    # Build a conversation section to always display conversations even when
    # no side-by-side comparison is available.
    if conversation_comparison_html:
        conversation_section = conversation_comparison_html
    else:
        parts = ['<div class="content-section">', '<h3>Conversation</h3>', '<div class="conversation">']
        if original_conversation_html:
            parts.append('<h4>📄 Original Conversation</h4>')
            parts.append(original_conversation_html)
        if attacked_conversation_html:
            parts.append('<h4>🔧 Attacked Conversation</h4>')
            parts.append(attacked_conversation_html)
        if not original_conversation_html and not attacked_conversation_html:
            parts.append('<p>No conversation available</p>')
        parts.append('</div></div>')
        conversation_section = "".join(parts)

    # Attacked rubrics
    attacked_rubrics = result.get('attacked_rubrics', [])
    attacked_rubrics_html = ""
    if attacked_rubrics:
        attacked_rubrics_html = '<div class="rubrics-list">'
        for rubric in attacked_rubrics:
            criterion = rubric.get('criterion', 'N/A')
            points = rubric.get('points', 'N/A')
            tags = rubric.get('tags', [])
            tags_str = ", ".join(tags) if tags else ""
            criterion_short = truncate_text(criterion, max_length=100)
            attacked_rubrics_html += f"""
                <div class="rubric-item">
                    <details>
                        <summary><strong>({points} pts)</strong> {escape_html(criterion_short)}</summary>
                        <div class="criterion-full">{escape_html(criterion)}</div>
                        {f'<div class="grade-tags">{escape_html(tags_str)}</div>' if tags_str else ''}
                    </details>
                </div>
            """
        attacked_rubrics_html += '</div>'
    else:
        attacked_rubrics_html = '<div class="rubric-item">No rubrics available</div>'

    # Attacked grades
    attacked_grades = result.get('attacked_grades', [])
    if attacked_grades:
        attacked_grades_html = "".join([format_grade(grade) for grade in attacked_grades])
    else:
        attacked_grades_html = '<div class="grade-item">No grades available</div>'

    # Determine per-case trigger/avoid badge
    case_trigger_badge = ''
    attack_applicable = False
    if attack_result:
        attack_applicable = bool(attack_result.get('applicable', False))

    if attack_applicable and attacked_grades:
        # If any penalty rubric (points < 0) is met, consider the attack triggered
        triggered = False
        for g in attacked_grades:
            points = g.get('points')
            criteria_met = g.get('criteria_met', False)
            try:
                is_num = isinstance(points, (int, float))
            except Exception:
                is_num = False
            if is_num and points < 0 and criteria_met:
                triggered = True
                break

        if triggered:
            case_trigger_badge = '<span class="case-badge triggered">Triggered</span>'
        else:
            case_trigger_badge = '<span class="case-badge avoided">Avoided</span>'

    # Attacked completion (make it collapsible)
    attacked_completion = result.get('attacked_completion', 'N/A')
    attacked_completion_html = ""
    if attacked_completion and attacked_completion != 'N/A':
        attacked_completion_html = f"""
            <div class="content-section collapsible-section">
                <h3 class="toggle-header" onclick="toggleSection(this)">
                    Model Completion <span class="toggle-icon">▼</span>
                </h3>
                <div class="collapsible-content">
                    <div class="completion-text">{format_text_preserve_newlines(attacked_completion)}</div>
                </div>
            </div>
        """
    else:
        attacked_completion_html = '<div class="content-section"><p>No completion available</p></div>'

    # Format score
    attacked_score_str = f"{attacked_score:.3f}" if attacked_score is not None else "N/A"

    return f"""
    <div class="case-card">
        <div class="case-header">
            <h2>Case #{sample_number}: {escape_html(case_id)}</h2>
            {case_trigger_badge}
        </div>

        {attack_html}

        {conversation_section}

        {attacked_completion_html}

        <div class="content-section">
            <h3>📋 Rubric & Grading</h3>
            <div class="rubrics-grades-container">
                <div class="grades">
                    {attacked_grades_html}
                </div>
            </div>
        </div>
    </div>
    """


def generate_html_report(results_path: str, output_path: str):
    """Generate an interactive HTML report from the HealthBench attack evaluation JSON results."""

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {results_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {results_path}: {e}")
        return

    metadata = data.get('metadata', {})
    results = data.get('results', [])

    if not results:
        print("Warning: No results found in JSON file")

    # Create summary statistics
    summary_html = create_summary_stats(data)

    # Create case cards
    case_cards_html = "".join([create_case_card(res) for res in results])

    # Extract metadata with canonical nested fields and supported input fallbacks.
    testee_model = _healthbench_model_id(metadata, "testee") or 'Unknown'
    grader_model = _healthbench_model_id(metadata, "grader") or 'Unknown'
    attack_strategies = healthbench_attack_strategy_order(metadata)
    num_samples = metadata.get('total_cases_evaluated', len(results))
    timestamp = metadata.get('timestamp', 'Unknown')
    dataset = healthbench_dataset_path(metadata) or 'Unknown'
    source_info = metadata.get('source') if isinstance(metadata.get('source'), dict) else {}
    source_file = source_info.get('source_results_file') or metadata.get('source_attack_results')

    # If dataset is Unknown and this is a replay file, try to get it from the nested source.
    if dataset == 'Unknown' and source_file:
        try:
            # Try multiple path resolution strategies
            source_path = Path(source_file)

            # First, try as absolute path or relative to current working directory
            if not source_path.exists():
                # If that doesn't work, try relative to the results_path directory
                results_dir = Path(results_path).parent
                source_path = results_dir / source_file

            if source_path.exists():
                with open(source_path, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                    dataset = healthbench_dataset_path(source_data.get('metadata', {})) or 'Unknown'
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            # If we can't read the source file, keep dataset as Unknown
            pass

    # Extract attack strategy model IDs from models, then supported metadata fields.
    attacker_strategies_config = healthbench_attack_strategy_config(metadata)

    attack_strategies_with_models = []
    for strategy in attack_strategies:
        strategy_name = strategy.replace('_', ' ').title()
        strategy_config = attacker_strategies_config.get(strategy, {})
        model_id = strategy_config.get('model_id', 'Unknown')
        attack_strategies_with_models.append(f"{strategy_name}[{model_id}]")

    attack_strategies_str = ", ".join(attack_strategies_with_models) if attack_strategies_with_models else "Unknown"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HealthBench Attack Evaluation Report</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2d3748;
            line-height: 1.7;
            font-size: 15px;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px 20px;
        }}
        .header, .summary-card, .case-card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 24px;
            padding: 28px 32px;
        }}
        .header {{
            background: linear-gradient(135deg, #5b7fff 0%, #3a52d4 100%);
            color: white;
            box-shadow: 0 6px 12px rgba(91, 127, 255, 0.3);
        }}
        .header h1 {{
            margin: 0 0 16px 0;
            color: white;
            font-size: 2.2em;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .header p {{
            margin: 10px 0;
            opacity: 0.95;
            font-size: 1.05em;
            line-height: 1.6;
        }}
        h1, h2, h3, h4 {{
            color: #1a202c;
            margin-top: 0;
            font-weight: 600;
        }}
        h2 {{
            font-size: 1.6em;
            margin-bottom: 20px;
            padding-bottom: 0;
            border-bottom: none;
        }}
        h3 {{
            font-size: 1.25em;
            color: #2d3748;
            margin-bottom: 14px;
        }}
        h4 {{
            font-size: 1.1em;
            color: #4a5568;
            margin-bottom: 12px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-item {{
            text-align: center;
            padding: 24px 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .stat-item:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #5b7fff;
            margin-bottom: 8px;
            line-height: 1;
        }}
        .stat-label {{
            font-size: 0.85em;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 500;
        }}
        .case-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 3px solid #f1f3f5;
        }}
        .case-header h2 {{
            margin: 0;
            flex: 1;
        }}
        .conversation-comparison {{
            margin: 24px 0;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 24px;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        @media (max-width: 1200px) {{
            .comparison-grid {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
        }}
        .comparison-column {{
            background: white;
            border-radius: 8px;
            padding: 0;
            overflow: hidden;
        }}
        .comparison-column h4 {{
            margin: 0;
            padding: 16px 20px;
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            color: #495057;
            font-size: 1.1em;
            font-weight: 600;
            border-bottom: 2px solid #adb5bd;
        }}
        .conversation-box {{
            padding: 20px;
            max-height: 500px;
            overflow-y: auto;
        }}
        .conversation-box.original {{
            background: #fff;
        }}
        .conversation-box.modified {{
            background: #fff8e1;
        }}
        .measurement-badge {{
            margin: 12px 20px;
            padding: 12px 16px;
            background: linear-gradient(135deg, #fff3cd 0%, #ffe5a0 100%);
            border-left: 4px solid #ff9800;
            border-radius: 6px;
            font-size: 0.95em;
            color: #8b5000;
            font-weight: 500;
        }}
        .measurement-badge strong {{
            color: #d84315;
            font-weight: 700;
        }}
        .content-section {{
            margin-bottom: 20px;
        }}
        .content-section h3 {{
            color: #667eea;
            margin-bottom: 12px;
            font-size: 1.2em;
        }}
        .collapsible-section .toggle-header {{
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px;
            transition: all 0.3s;
            margin-bottom: 0;
            border: 2px solid transparent;
        }}
        .collapsible-section .toggle-header:hover {{
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            border-color: #5b7fff;
        }}
        .toggle-icon {{
            transition: transform 0.3s ease;
            font-size: 1em;
            color: #5b7fff;
        }}
        .toggle-icon.rotated {{
            transform: rotate(-180deg);
        }}
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease;
        }}
        .collapsible-content.open {{
            max-height: 10000px;
            padding-top: 18px;
        }}
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: #f1f3f5;
            border-radius: 5px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #5b7fff;
            border-radius: 5px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #4a6dd9;
        }}
        .conversation, .completion, .grades, .rubrics {{
            background: #f9f9f9;
            border-radius: 6px;
            padding: 16px;
            margin-top: 12px;
        }}
        .message {{
            margin-bottom: 14px;
            padding: 16px 18px;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .message:hover {{
            transform: translateX(2px);
        }}
        .message-user {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 4px solid #2196f3;
        }}
        .message-assistant {{
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border-left: 4px solid #9c27b0;
        }}
        .message-role {{
            font-weight: 700;
            font-size: 0.75em;
            margin-bottom: 8px;
            color: #37474f;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .message-content {{
            font-size: 0.95em;
            line-height: 1.7;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #263238;
        }}
        .completion-text {{
            background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
            padding: 20px;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 0.95em;
            white-space: pre-wrap;
            word-wrap: break-word;
            border: 2px solid #e2e8f0;
            color: #2d3748;
        }}
        .summary-card h2 {{
            color: #1a202c;
            font-size: 1.8em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .summary-card h2::before {{
            content: "📊";
            font-size: 1.2em;
        }}
        .attack-info {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border-left: 5px solid #2196f3;
            padding: 20px 24px;
            margin: 24px 0;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(33, 150, 243, 0.15);
        }}
        .attack-info h3 {{
            margin-top: 0;
            color: #1565c0;
            font-size: 1.15em;
        }}
        .attack-info.not-applicable {{
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
            border-left-color: #757575;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }}
        .attack-info.error {{
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            border-left-color: #f44336;
            box-shadow: 0 2px 8px rgba(244, 67, 54, 0.15);
        }}
        .attack-details {{
            margin-top: 16px;
            background: rgba(255, 255, 255, 0.5);
            padding: 12px;
            border-radius: 6px;
        }}
        .attack-details p {{
            margin: 10px 0;
            line-height: 1.6;
        }}
        .highlight {{
            background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);
            padding: 4px 10px;
            border-radius: 5px;
            font-weight: 700;
            color: #f57f17;
            border: 1px solid #fbc02d;
        }}
        details {{
            margin: 8px 0;
            cursor: pointer;
        }}
        summary {{
            font-weight: 500;
            padding: 4px 0;
        }}
        summary:hover {{
            color: #667eea;
        }}
        .criterion-full {{
            margin-top: 8px;
            padding: 12px;
            background: #f5f5f5;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9em;
        }}
        .score-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 16px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            margin-left: 12px;
        }}
        .case-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            color: white;
            font-weight: 700;
            font-size: 0.95em;
            margin-left: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .case-badge.avoided {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }}
        .case-badge.triggered {{
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }}
        .score-up {{
            background: #4CAF50;
        }}
        .score-down {{
            background: #F44336;
        }}
        .score-same {{
            background: #9E9E9E;
        }}
        .score-attacked {{
            background: #2196F3;
        }}
        .score-error {{
            background: #F44336;
        }}
        .score-na {{
            background: #9E9E9E;
        }}
        .score-display {{
            font-size: 1.1em;
            margin: 12px 0;
            padding: 8px;
            background: #f0f0f0;
            border-radius: 4px;
        }}
        .grade-item {{
            margin-bottom: 18px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            border-left: 5px solid #cbd5e0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: all 0.2s;
        }}
        .grade-item:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-1px);
        }}
        .grade-item.met {{
            border-left-color: #48bb78;
            background: linear-gradient(135deg, #ffffff 0%, #f0fff4 100%);
        }}
        .grade-item.not-met {{
            border-left-color: #f56565;
            background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
        }}
        .grade-item.met-bad {{
            border-left-color: #f56565;
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border: 2px solid #fc8181;
        }}
        .grade-item.not-met-good {{
            border-left-color: #48bb78;
            background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
            border: 2px solid #68d391;
        }}
        .grade-item.penalty-rubric {{
            border: 3px solid #f6ad55;
            background: linear-gradient(135deg, #fffaf0 0%, #feebc8 100%);
        }}
        .grade-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .grade-status {{
            font-weight: 700;
            font-size: 1.05em;
        }}
        .grade-item.met .grade-status {{
            color: #2f855a;
        }}
        .grade-item.not-met .grade-status {{
            color: #c53030;
        }}
        .grade-item.met-bad .grade-status {{
            color: #c53030;
        }}
        .grade-item.not-met-good .grade-status {{
            color: #2f855a;
        }}
        .grade-points {{
            color: #718096;
            font-size: 0.95em;
            font-weight: 600;
            background: rgba(0,0,0,0.05);
            padding: 4px 10px;
            border-radius: 5px;
        }}
        .grade-tags {{
            font-size: 0.85em;
            color: #718096;
            margin: 8px 0;
            font-style: italic;
        }}
        .grade-criterion {{
            margin: 12px 0;
        }}
        .grade-explanation {{
            margin-top: 12px;
            padding: 14px;
            background: rgba(0,0,0,0.02);
            border-radius: 6px;
            font-size: 0.95em;
            color: #4a5568;
            line-height: 1.7;
            border-left: 3px solid #cbd5e0;
        }}
        .rubrics-list {{
            margin-top: 8px;
        }}
        .rubric-item {{
            margin-bottom: 12px;
            padding: 12px;
            background: white;
            border-radius: 4px;
            border-left: 4px solid #9E9E9E;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚔️ HealthBench Attack Evaluation Report</h1>
            <p><strong>Testee Model:</strong> {escape_html(testee_model)}</p>
            <p><strong>Grader Model:</strong> {escape_html(grader_model)}</p>
            <p><strong>Attack Strategies:</strong> {escape_html(attack_strategies_str)}</p>
            <p><strong>Dataset:</strong> {escape_html(dataset)}</p>
            <p><strong>Number of Cases:</strong> {num_samples}</p>
            <p><strong>Timestamp:</strong> {escape_html(timestamp)}</p>
        </div>

        {summary_html}

        {case_cards_html}

    </div>

    <script>
        function toggleSection(header) {{
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');

            content.classList.toggle('open');
            icon.classList.toggle('rotated');
        }}
    </script>
</body>
</html>"""

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Generated HTML report: {output_path}")
    except IOError as e:
        print(f"Error: Could not write to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report for HealthBench attack evaluation.")
    parser.add_argument("json_path", help="Path to the attack evaluation results JSON file")
    parser.add_argument("-o", "--output", help="Output HTML file path (default: same as JSON with .html extension)")
    args = parser.parse_args()

    results_path = Path(args.json_path)
    if not results_path.exists():
        print(f"Error: File not found: {results_path}")
        exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.with_suffix('.html')

    generate_html_report(str(results_path), str(output_path))

