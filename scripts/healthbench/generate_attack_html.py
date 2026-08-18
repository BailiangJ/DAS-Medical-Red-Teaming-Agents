#!/usr/bin/env python3
"""
Generate HTML visualization for attack comparison results (unified for distraction/cognitive bias).

This script creates an interactive HTML viewer for attack results,
with side-by-side comparison of baseline vs attacked performance.

Supports both:
- Cognitive bias attacks (from run_cognitive_bias_v1.py)
- Distraction attacks (from healthbench_distraction_v2.py)

The attack type is auto-detected from the JSON structure.

Usage:
    python -m scripts.healthbench.generate_attack_html <results.json> [-o output.html] [-t attack_type]
"""

import json
import html
import argparse
from pathlib import Path
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


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_attack_type(data: Dict[str, Any]) -> str:
    """
    Auto-detect attack type from JSON data.

    Returns 'cognitive_bias' or 'distraction'.
    """
    # First check metadata for explicit version/attack_type
    metadata = data.get('metadata', {})
    version = metadata.get('version', '').lower()
    attack_type_meta = metadata.get('attack_type', '').lower()
    attack_strategies = healthbench_attack_strategy_order(metadata)

    # Check attack_strategies list
    if attack_strategies:
        if 'distraction' in attack_strategies:
            return 'distraction'
        if 'cognitive_bias' in attack_strategies or 'cognitive' in str(attack_strategies).lower():
            return 'cognitive_bias'

    if 'cognitive' in version or 'bias' in version or 'cognitive' in attack_type_meta:
        return 'cognitive_bias'
    if 'distraction' in version or 'distraction' in attack_type_meta:
        return 'distraction'

    # Check attack_result -> attack_strategy in first result
    results = data.get('results', [])
    if results:
        attack_result = results[0].get('attack_result', {})
        attack_strategy = attack_result.get('attack_strategy', '').lower()
        if attack_strategy:
            if 'distraction' in attack_strategy:
                return 'distraction'
            if 'cognitive' in attack_strategy or 'bias' in attack_strategy:
                return 'cognitive_bias'

        # Fallback: check attack_info/attack_metadata structure
        attack_info = results[0].get('attack_info')
        if not attack_info:
            attack_info = attack_result.get('attack_metadata')

        if attack_info:
            # Cognitive bias has 'attack_strategies' or 'vulnerability_analysis'
            if 'attack_strategies' in attack_info or 'vulnerability_analysis' in attack_info:
                return 'cognitive_bias'
            # Distraction has 'trigger_keywords' or 'root_cause_analysis'
            if 'trigger_keywords' in attack_info or 'root_cause_analysis' in attack_info:
                return 'distraction'

    # Default to cognitive_bias if unknown
    return 'cognitive_bias'


def is_legacy_all_rubric_record(result: Dict[str, Any]) -> bool:
    """
    Detect if a result is affected by the non-comparable all-rubric signature signature.

    Matching records have applicable=True, selected_rubric_index=-1, an
    unchanged conversation, and grades for the full rubric set.

    Signature:
    1. attack_result.applicable = True
    2. attack_result.attack_metadata.selected_rubric_index = -1
    3. attacked_rubrics contains ALL rubrics (same count as original_rubrics)
    4. skipped = False (record is not marked skipped)
    5. attacked_conversation exists (but may be identical to original)

    Args:
        result: A single result dict from the JSON output

    Returns:
        True if this result is affected by the non-comparable all-rubric signature
    """
    # Check if attack_result exists and is applicable
    attack_result = result.get('attack_result')
    if not attack_result or not attack_result.get('applicable', False):
        return False

    # Check whether the record is marked skipped.
    if result.get('skipped', False):
        return False

    # Check selected_rubric_index = -1
    attack_metadata = attack_result.get('attack_metadata', {})
    selected_idx = attack_metadata.get('selected_rubric_index', -1)
    if selected_idx != -1:
        return False

    # Check if attack strategy is distraction or cognitive_bias
    attack_strategy = attack_result.get('attack_strategy', '').lower()
    if attack_strategy not in ['distraction', 'cognitive_bias']:
        return False

    # Check if attacked_rubrics has same count as original_rubrics (ALL rubrics fallback)
    original_rubrics = result.get('original_rubrics', [])
    attacked_rubrics = result.get('attacked_rubrics', [])

    if not attacked_rubrics:
        return False

    # Key signature: attacked_rubrics has ALL rubrics (not just 1 selected rubric)
    # For distraction/cognitive_bias, should only have 1 rubric if attack applied correctly
    if len(attacked_rubrics) == len(original_rubrics) and len(attacked_rubrics) > 1:
        return True

    return False


def is_conversation_unchanged(original_conv: List[Dict[str, str]], modified_conv: List[Dict[str, str]]) -> bool:
    """
    Check if two conversations are identical (helper for non-comparable all-rubric signature detection).

    Args:
        original_conv: Original conversation
        modified_conv: Modified conversation

    Returns:
        True if conversations are identical
    """
    if not original_conv or not modified_conv:
        return False

    if len(original_conv) != len(modified_conv):
        return False

    for orig, mod in zip(original_conv, modified_conv):
        if orig.get('role') != mod.get('role'):
            return False
        if orig.get('content', '').strip() != mod.get('content', '').strip():
            return False

    return True


def get_attack_config(attack_type: str) -> Dict[str, Any]:
    """Get configuration for the attack type."""
    if attack_type == 'distraction':
        return {
            'name': 'Distraction Attack',
            'title_suffix': 'Distraction Attacks',
            'primary_color': '#f59e0b',
            'secondary_color': '#d97706',
            'gradient': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
        }
    else:  # cognitive_bias
        return {
            'name': 'Cognitive Bias Attack',
            'title_suffix': 'Cognitive Bias Attacks',
            'primary_color': '#667eea',
            'secondary_color': '#764ba2',
            'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        }


def create_summary_stats(results: List[Dict[str, Any]], attack_type: str) -> str:
    """Create summary statistics section."""
    total_cases = len(results)
    config = get_attack_config(attack_type)

    # ==============================================================================
    # Filter out non-comparable all-rubric signature cases
    # ==============================================================================
    noncomparable_records = [r for r in results if is_legacy_all_rubric_record(r)]
    clean_results = [r for r in results if not is_legacy_all_rubric_record(r)]
    num_noncomparable = len(noncomparable_records)

    # Count cases where attack was actually applied (using clean results)
    cases_with_attacks = sum(1 for r in clean_results if r.get('attacked_conversation') is not None)

    # Count successful attacks (targeted rubric was degraded)
    successful_attacks = 0
    for r in clean_results:
        # Resolve attack information from supported locations.
        attack_info = r.get('attack_info')
        if not attack_info:
            attack_result = r.get('attack_result') or {}
            attack_info = attack_result.get('attack_metadata') if attack_result else None

        if attack_info and r.get('attacked_grades'):
            # Resolve supported field names.
            original_rubric_index = attack_info.get('original_rubric_index')
            # print(f"Original rubric index from attack_info: {original_rubric_index}")

            # Fallback: If still not found, infer from attacked_rubrics vs original_rubrics
            if original_rubric_index is None or original_rubric_index < 0:
                attacked_rubrics = r.get('attacked_rubrics', [])
                original_rubrics = r.get('original_rubrics', [])
                if attacked_rubrics and original_rubrics:
                    # Compare the attacked rubric with original rubrics to find the index
                    attacked_criterion = attacked_rubrics[0].get('criterion', '')
                    for idx, orig_rubric in enumerate(original_rubrics):
                        if orig_rubric.get('criterion', '') == attacked_criterion:
                            original_rubric_index = idx
                            print(f"Detected original_rubric_index as {idx} based on rubric matching.")
                            break

            if original_rubric_index is not None and original_rubric_index >= 0:
                baseline_grades = r.get('baseline_grades', [])
                attacked_grades = r.get('attacked_grades', [])

                if original_rubric_index < len(baseline_grades) and len(attacked_grades) > 0:
                    baseline_grade = baseline_grades[original_rubric_index]
                    attacked_grade = attacked_grades[0]  # Selected-rubric records contain one grade.

                    baseline_met = baseline_grade.get('criteria_met', False)
                    attacked_met = attacked_grade.get('criteria_met', False)
                    points = baseline_grade.get('points', 0)

                    # Check if attack succeeded (rubric was degraded)
                    if points > 0 and baseline_met and not attacked_met:
                        successful_attacks += 1
                    elif points < 0 and not baseline_met and attacked_met:
                        successful_attacks += 1

    # Calculate Attack Success Rate
    asr = (successful_attacks / cases_with_attacks * 100) if cases_with_attacks > 0 else 0

    # Add non-comparable all-rubric signature warning if cases were excluded
    comparability_warning_html = ""
    if num_noncomparable > 0:
        comparability_warning_html = f"""
        <div class="warning-banner" style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 16px; margin-top: 16px; border-radius: 8px;">
            <h3 style="margin: 0 0 8px 0; color: #92400e; font-size: 16px;">⚠️ Non-comparable Legacy Records</h3>
            <p style="margin: 0; color: #78350f; font-size: 14px;">
                <strong>{num_noncomparable}</strong> cases were excluded from statistics due to the non-comparable all-rubric signature
                (applicable=True, selected_rubric_index=-1, evaluated with ALL rubrics).
                These records are not comparable under the selected-rubric evaluation contract.
            </p>
            <p style="margin: 8px 0 0 0; color: #78350f; font-size: 13px;">
                Statistics above are computed from <strong>{len(clean_results)}</strong> comparable records only.
            </p>
        </div>
        """

    return f"""
        <div class="summary-card">
            <h2>Summary Statistics ({config['title_suffix']})</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_cases}</div>
                    <div class="stat-label">Total Cases</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{cases_with_attacks}</div>
                    <div class="stat-label">Attacks Applied</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{successful_attacks}</div>
                    <div class="stat-label">Successful Attacks</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{asr:.1f}%</div>
                    <div class="stat-label">Attack Success Rate</div>
                </div>
            </div>
            {comparability_warning_html}
        </div>
    """


def format_conversation(conversation: List[Dict[str, str]]) -> str:
    """Format conversation messages."""
    html_parts = []
    for msg in conversation:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_content = format_text_preserve_newlines(content)
        html_parts.append(
            f'<div class="message message-{role}">'
            f'<strong>{escape_html(role).title()}:</strong>'
            f'<div class="message-content">{formatted_content}</div>'
            f'</div>'
        )
    return ''.join(html_parts)


def format_attack_info_cognitive_bias(attack_info: Dict[str, Any]) -> str:
    """Format cognitive bias attack information."""
    if not attack_info:
        return '<div class="attack-info">No attack applied.</div>'

    # Resolve supported field names.
    # Note: Use 'or' to handle None and empty strings
    vulnerability_analysis = attack_info.get('vulnerability_analysis') or attack_info.get('rationale') or 'N/A'
    rubric_type = attack_info.get('rubric_type') or 'N/A'
    attack_goal = attack_info.get('attack_goal') or 'N/A'
    bias_type = attack_info.get('bias_type') or 'N/A'
    attackability_reason = attack_info.get('attackability_reason') or 'N/A'
    modification_reason = attack_info.get('modification_reason') or 'N/A'
    rubric_validity_reason = attack_info.get('rubric_validity_reason') or 'N/A'

    # Rubric index
    selected_rubric_index = attack_info.get('selected_rubric_index', 'N/A')

    # Try to get rubric text from multiple sources
    selected_rubric_text = attack_info.get('selected_rubric_text') or attack_info.get('selected_rubric_summary') or ''
    if not selected_rubric_text:
        # Try to extract from attackable_rubric_items if available
        attackable_items = attack_info.get('attackable_rubric_items', [])
        if attackable_items and isinstance(selected_rubric_index, int) and selected_rubric_index < len(attackable_items):
            selected_rubric_text = attackable_items[selected_rubric_index].get('criterion', 'N/A')
        else:
            selected_rubric_text = 'N/A'

    # Attack strategies (list) - may or may not exist
    attack_strategies = attack_info.get('attack_strategies', [])
    strategies_html = ""
    if attack_strategies:
        strategies_parts = []
        for strat in attack_strategies:
            bias = strat.get('cognitive_bias', 'N/A')
            framing = strat.get('framing_effect_applied', False)
            reason = strat.get('reason', 'N/A')
            framing_badge = '<span class="framing-badge">+Framing</span>' if framing else ''
            strategies_parts.append(
                f'<div class="strategy-item">'
                f'<span class="strategy-badge">{escape_html(bias)}</span> {framing_badge}'
                f'<div class="strategy-reason">{escape_html(reason)}</div>'
                f'</div>'
            )
        strategies_html = '<div class="strategies-list">' + ''.join(strategies_parts) + '</div>'

    # Build details sections dynamically based on what's available
    details_html = ""

    if vulnerability_analysis != 'N/A':
        details_html += f"""
                <details open>
                    <summary><strong>Rationale / Vulnerability Analysis</strong></summary>
                    <div class="info-block">{escape_html(vulnerability_analysis)}</div>
                </details>"""

    if modification_reason != 'N/A':
        details_html += f"""
                <details>
                    <summary><strong>Modification Reason</strong></summary>
                    <div class="info-block">{escape_html(modification_reason)}</div>
                </details>"""

    if attackability_reason != 'N/A':
        details_html += f"""
                <details>
                    <summary><strong>Attackability Reason</strong></summary>
                    <div class="info-block">{escape_html(attackability_reason)}</div>
                </details>"""

    if strategies_html:
        details_html += f"""
                <details>
                    <summary><strong>Attack Strategies</strong></summary>
                    {strategies_html}
                </details>"""

    if selected_rubric_text != 'N/A':
        details_html += f"""
                <details>
                    <summary><strong>Targeted Rubric</strong></summary>
                    <div class="rubric-full">{escape_html(selected_rubric_text)}</div>
                </details>"""

    if rubric_validity_reason != 'N/A':
        details_html += f"""
                <details>
                    <summary><strong>Rubric Validity Reason</strong></summary>
                    <div class="info-block">{escape_html(rubric_validity_reason)}</div>
                </details>"""

    rubric_info = ""
    if selected_rubric_index != 'N/A':
        rubric_info = f"<p><strong>Targeted Rubric Index:</strong> #{selected_rubric_index}</p>"
    if rubric_type != 'N/A' and attack_goal != 'N/A':
        rubric_info += f"<p><strong>Rubric Type:</strong> {escape_html(rubric_type)} → <strong>Attack Goal:</strong> {escape_html(attack_goal)}</p>"

    return f"""
        <div class="attack-info">
            <h3>Cognitive Bias Attack</h3>
            <div class="attack-details">
                {rubric_info}
                {details_html}
            </div>
        </div>
    """


def format_attack_info_distraction(attack_info: Dict[str, Any]) -> str:
    """Format distraction attack information."""
    if not attack_info:
        return '<div class="attack-info">No attack applied.</div>'

    # Extract fields from supported result formats.
    # Note: Use 'or' to handle None and empty strings
    root_cause_analysis = attack_info.get('root_cause_analysis') or 'N/A'
    rubric_validity_reason = attack_info.get('rubric_validity_reason') or 'N/A'

    # Rubric index
    selected_rubric_index = attack_info.get('selected_rubric_index', 'N/A')

    # Try to get rubric text from attackable_rubric_items if available
    selected_rubric_text = attack_info.get('selected_rubric_text') or ''
    if not selected_rubric_text:
        attackable_items = attack_info.get('attackable_rubric_items', [])
        if attackable_items and isinstance(selected_rubric_index, int) and selected_rubric_index < len(attackable_items):
            selected_rubric_text = attackable_items[selected_rubric_index].get('criterion', 'N/A')
        else:
            selected_rubric_text = 'N/A'

    # Old format fields (may not exist in new format)
    rubric_type = attack_info.get('rubric_type') or 'N/A'
    attack_goal = attack_info.get('attack_goal') or 'N/A'

    # Trigger keywords (list)
    trigger_keywords = attack_info.get('trigger_keywords', [])
    keywords_html = ""
    if trigger_keywords:
        keyword_badges = ''.join([
            f'<span class="keyword-badge">{escape_html(kw)}</span>'
            for kw in trigger_keywords
        ])
        keywords_html = f'<div class="keywords-list">{keyword_badges}</div>'

    # Build details sections dynamically based on what's available
    details_html = ""

    if root_cause_analysis != 'N/A':
        details_html += f"""
                <details open>
                    <summary><strong>Root Cause Analysis</strong></summary>
                    <div class="info-block">{escape_html(root_cause_analysis)}</div>
                </details>"""

    if keywords_html:
        details_html += f"""
                <details>
                    <summary><strong>Trigger Keywords</strong></summary>
                    {keywords_html}
                </details>"""

    if selected_rubric_text != 'N/A':
        details_html += f"""
                <details>
                    <summary><strong>Targeted Rubric</strong></summary>
                    <div class="rubric-full">{escape_html(selected_rubric_text)}</div>
                </details>"""

    if rubric_validity_reason != 'N/A':
        details_html += f"""
                <details>
                    <summary><strong>Rubric Validity Reason</strong></summary>
                    <div class="info-block">{escape_html(rubric_validity_reason)}</div>
                </details>"""

    # Build rubric info header
    rubric_info = ""
    if selected_rubric_index != 'N/A':
        rubric_info = f"<p><strong>Targeted Rubric Index:</strong> #{selected_rubric_index}</p>"
    if rubric_type != 'N/A' and attack_goal != 'N/A':
        rubric_info += f"<p><strong>Rubric Type:</strong> {escape_html(rubric_type)} → <strong>Attack Goal:</strong> {escape_html(attack_goal)}</p>"

    return f"""
        <div class="attack-info">
            <h3>Distraction Attack</h3>
            <div class="attack-details">
                {rubric_info}
                {details_html}
            </div>
        </div>
    """


def format_attack_info(attack_info: Dict[str, Any], attack_type: str) -> str:
    """Format attack information based on attack type."""
    if attack_type == 'distraction':
        return format_attack_info_distraction(attack_info)
    else:  # cognitive_bias
        return format_attack_info_cognitive_bias(attack_info)


def create_baseline_only_row(baseline_grade: Dict[str, Any], index: int) -> str:
    """Create a row showing only baseline grade (rubric was not the selected target)."""
    criterion = baseline_grade.get('criterion', 'N/A')
    points = baseline_grade.get('points', 'N/A')

    # Determine rubric type
    is_positive = isinstance(points, (int, float)) and points > 0
    is_negative = isinstance(points, (int, float)) and points < 0

    # Format points
    if isinstance(points, (int, float)):
        points_display = f"{points:+.1f}" if points != 0 else "0.0"
    else:
        points_display = str(points)

    points_class = "points-positive" if is_positive else ("points-negative" if is_negative else "")

    # Truncate criterion
    criterion_short = truncate_text(criterion, max_length=100)
    criterion_full = escape_html(criterion)

    # Baseline grade info
    baseline_met = baseline_grade.get('criteria_met', False)
    baseline_explanation = baseline_grade.get('explanation', '')

    if is_positive:
        baseline_status_class = "status-good" if baseline_met else "status-bad"
        not_targeted_reason = "Not selected as target" if baseline_met else "Not attackable (not met)"
    elif is_negative:
        baseline_status_class = "status-bad" if baseline_met else "status-good"
        not_targeted_reason = "Not selected as target" if not baseline_met else "Not attackable (met)"
    else:
        baseline_status_class = "status-good" if baseline_met else "status-bad"
        not_targeted_reason = "Not selected as target"

    baseline_status = "✓ Met" if baseline_met else "✗ Not Met"

    return f"""
        <div class="rubric-comparison-row non-attackable-row">
            <div class="rubric-info-col">
                <div class="rubric-number">#{index}</div>
                <div class="rubric-criterion-compare">
                    <details>
                        <summary>{criterion_short}</summary>
                        <div class="criterion-full">{criterion_full}</div>
                    </details>
                </div>
                <span class="grade-points-badge {points_class}">Points: {points_display}</span>
            </div>
            <div class="rubric-grade-col">
                <div class="grade-status-compare {baseline_status_class}">{baseline_status}</div>
                <details class="grade-explanation-details">
                    <summary>Grader Reasoning</summary>
                    <div class="grade-explanation-full">{escape_html(baseline_explanation)}</div>
                </details>
            </div>
            <div class="rubric-grade-col not-attackable-col">
                <div class="not-attackable-badge">{not_targeted_reason}</div>
                <div class="not-attackable-note">This rubric was not graded in the attack round</div>
            </div>
        </div>
    """


def create_rubric_comparison_row(baseline_grade: Dict[str, Any], attacked_grade: Dict[str, Any], index: int, is_targeted: bool = False) -> str:
    """Create a horizontal comparison row for a single rubric."""
    criterion = baseline_grade.get('criterion', 'N/A')
    points = baseline_grade.get('points', 'N/A')

    # Determine rubric type
    is_positive = isinstance(points, (int, float)) and points > 0
    is_negative = isinstance(points, (int, float)) and points < 0

    # Format points
    if isinstance(points, (int, float)):
        points_display = f"{points:+.1f}" if points != 0 else "0.0"
    else:
        points_display = str(points)

    points_class = "points-positive" if is_positive else ("points-negative" if is_negative else "")

    # Truncate criterion
    criterion_short = truncate_text(criterion, max_length=100)
    criterion_full = escape_html(criterion)

    # Baseline grade info
    baseline_met = baseline_grade.get('criteria_met', False)
    baseline_explanation = baseline_grade.get('explanation', '')

    if is_positive:
        baseline_status_class = "status-good" if baseline_met else "status-bad"
    elif is_negative:
        baseline_status_class = "status-bad" if baseline_met else "status-good"
    else:
        baseline_status_class = "status-good" if baseline_met else "status-bad"

    baseline_status = "✓ Met" if baseline_met else "✗ Not Met"

    # Attacked grade info
    attacked_met = attacked_grade.get('criteria_met', False)
    attacked_explanation = attacked_grade.get('explanation', '')

    if is_positive:
        attacked_status_class = "status-good" if attacked_met else "status-bad"
    elif is_negative:
        attacked_status_class = "status-bad" if attacked_met else "status-good"
    else:
        attacked_status_class = "status-good" if attacked_met else "status-bad"

    attacked_status = "✓ Met" if attacked_met else "✗ Not Met"

    # Check if degraded
    if is_positive:
        is_degraded = baseline_met and not attacked_met
    elif is_negative:
        is_degraded = not baseline_met and attacked_met
    else:
        is_degraded = False

    degraded_class = "degraded-row" if is_degraded else ""
    targeted_badge = '<span class="targeted-badge">🎯 TARGETED</span>' if is_targeted else ""

    return f"""
        <div class="rubric-comparison-row {degraded_class}">
            <div class="rubric-info-col">
                <div class="rubric-number">#{index} {targeted_badge}</div>
                <div class="rubric-criterion-compare">
                    <details>
                        <summary>{criterion_short}</summary>
                        <div class="criterion-full">{criterion_full}</div>
                    </details>
                </div>
                <span class="grade-points-badge {points_class}">Points: {points_display}</span>
            </div>
            <div class="rubric-grade-col">
                <div class="grade-status-compare {baseline_status_class}">{baseline_status}</div>
                <details class="grade-explanation-details">
                    <summary>Grader Reasoning</summary>
                    <div class="grade-explanation-full">{escape_html(baseline_explanation)}</div>
                </details>
            </div>
            <div class="rubric-grade-col">
                <div class="grade-status-compare {attacked_status_class}">{attacked_status}</div>
                <details class="grade-explanation-details">
                    <summary>Grader Reasoning</summary>
                    <div class="grade-explanation-full">{escape_html(attacked_explanation)}</div>
                </details>
            </div>
        </div>
    """


def create_case_card(result: Dict[str, Any], attack_type: str) -> str:
    """Create HTML for a single case card with side-by-side comparison."""
    case_id = result.get('test_case_id') or result.get('case_id', 'Unknown')
    config = get_attack_config(attack_type)

    # Resolve attack information from supported locations.
    attack_info = result.get('attack_info')
    if not attack_info:
        # New format: attack_result -> attack_metadata
        attack_result = result.get('attack_result') or {}
        attack_info = attack_result.get('attack_metadata') if attack_result else None
    targeted_attack_badge = ""
    if attack_info:
        # Resolve supported field names.
        original_rubric_index = attack_info.get('original_rubric_index')

        # Fallback: If still not found, infer from attacked_rubrics vs original_rubrics
        if original_rubric_index is None or original_rubric_index < 0:
            attacked_rubrics = result.get('attacked_rubrics', [])
            original_rubrics = result.get('original_rubrics', [])
            if attacked_rubrics and original_rubrics:
                # Compare the attacked rubric with original rubrics to find the index
                attacked_criterion = attacked_rubrics[0].get('criterion', '')
                for idx, orig_rubric in enumerate(original_rubrics):
                    if orig_rubric.get('criterion', '') == attacked_criterion:
                        original_rubric_index = idx
                        print(f"Detected original_rubric_index as {idx} based on rubric matching.")
                        break

        if original_rubric_index is not None and original_rubric_index >= 0:
            # Find the targeted rubric in baseline and attacked grades
            baseline_grades = result.get('baseline_grades', [])
            attacked_grades = result.get('attacked_grades', [])

            if original_rubric_index < len(baseline_grades) and len(attacked_grades) > 0:
                baseline_grade = baseline_grades[original_rubric_index]
                attacked_grade = attacked_grades[0]  # Selected-rubric records contain one grade.

                baseline_met = baseline_grade.get('criteria_met', False)
                attacked_met = attacked_grade.get('criteria_met', False)
                points = baseline_grade.get('points', 0)

                # Check if degraded
                is_degraded = False
                if points > 0:  # Positive rubric
                    is_degraded = baseline_met and not attacked_met
                elif points < 0:  # Negative rubric
                    is_degraded = not baseline_met and attacked_met

                if is_degraded:
                    targeted_attack_badge = '<span class="attack-badge attack-success">✓ Attack Succeeded</span>'
                else:
                    targeted_attack_badge = '<span class="attack-badge attack-failed">✗ Attack Failed</span>'
        else:
            # No attack was applied
            targeted_attack_badge = '<span class="attack-badge attack-unknown">No Attack</span>'
    else:
        # No attack was applied
        targeted_attack_badge = '<span class="attack-badge attack-unknown">No Attack</span>'

    # Attack info
    attack_html = format_attack_info(attack_info, attack_type)

    # Original conversation
    original_conversation = result.get('original_conversation', [])
    original_conversation_html = format_conversation(original_conversation)

    # Attacked conversation (can be list of messages or string)
    attacked_conversation_data = result.get('attacked_conversation', None)
    attacked_conversation_html = ""
    if attacked_conversation_data:
        # Check if it's a list (structured messages) or string
        if isinstance(attacked_conversation_data, list):
            # Format as structured conversation
            attacked_conversation_html = format_conversation(attacked_conversation_data)
        elif isinstance(attacked_conversation_data, str):
            # Display as text
            attacked_conversation_html = f'<div class="conversation-text">{format_text_preserve_newlines(attacked_conversation_data)}</div>'
        else:
            attacked_conversation_html = '<div class="message">N/A</div>'
    else:
        attacked_conversation_html = '<div class="message">N/A</div>'

    # Create rubric comparison - show ALL baseline rubrics but only the targeted one was graded in attack
    baseline_grades = result.get('baseline_grades') or []
    attacked_grades = result.get('attacked_grades') or []

    rubric_comparison_html = ""
    if baseline_grades:
        # Resolve supported field names.
        if attack_info:
            original_rubric_index = attack_info.get('original_rubric_index')

            # Fallback: If still not found, infer from attacked_rubrics vs original_rubrics
            if original_rubric_index is None or original_rubric_index < 0:
                attacked_rubrics = result.get('attacked_rubrics', [])
                original_rubrics = result.get('original_rubrics', [])
                if attacked_rubrics and original_rubrics:
                    # Compare the attacked rubric with original rubrics to find the index
                    attacked_criterion = attacked_rubrics[0].get('criterion', '')
                    for idx, orig_rubric in enumerate(original_rubrics):
                        if orig_rubric.get('criterion', '') == attacked_criterion:
                            original_rubric_index = idx
                            print(f"Detected original_rubric_index as {idx} based on rubric matching.")
                            break
        else:
            original_rubric_index = -1

        for i, baseline_grade in enumerate(baseline_grades):
            # Check if this is the targeted rubric
            if i == original_rubric_index and len(attacked_grades) > 0:
                # This is the targeted rubric - show comparison
                attacked_grade = attacked_grades[0]  # Selected-rubric records contain one grade.
                rubric_comparison_html += create_rubric_comparison_row(baseline_grade, attacked_grade, i, is_targeted=True)
            else:
                # Not the targeted rubric - show baseline only
                rubric_comparison_html += create_baseline_only_row(baseline_grade, i)
    else:
        rubric_comparison_html = '<div class="no-grades">No grades available</div>'

    # Baseline completion
    baseline_completion = result.get('baseline_completion', 'N/A')
    baseline_completion_html = format_text_preserve_newlines(baseline_completion)

    # Attacked completion
    attacked_completion = result.get('attacked_completion', 'N/A')
    attacked_completion_html = format_text_preserve_newlines(attacked_completion) if attacked_completion != 'N/A' else 'N/A'

    return f"""
    <div class="case-card">
        <h2>Case: {escape_html(case_id)} {targeted_attack_badge}</h2>

        {attack_html}

        <div class="grid">
            <div class="column">
                <h3>Baseline</h3>
                <div class="conversation">
                    <h4>Original Conversation</h4>
                    {original_conversation_html}
                </div>
                <div class="completion">
                    <h4>Model Completion</h4>
                    <div class="completion-content">{baseline_completion_html}</div>
                </div>
            </div>
            <div class="column">
                <h3>After {config['name']}</h3>
                <div class="conversation conversation-attacked">
                    <h4>Modified Conversation</h4>
                    {attacked_conversation_html}
                </div>
                <div class="completion">
                    <h4>Model Completion</h4>
                    <div class="completion-content">{attacked_completion_html}</div>
                </div>
            </div>
        </div>

        <div class="rubric-comparison-section">
            <h3>Rubric Grades Comparison</h3>
            <div class="rubric-comparison-note">
                <strong>Note:</strong> Only the targeted rubric (marked with 🎯) was graded in the attack round.
            </div>
            <div class="rubric-comparison-header">
                <div class="rubric-info-col">Rubric</div>
                <div class="rubric-grade-col">Baseline</div>
                <div class="rubric-grade-col">After Attack</div>
            </div>
            {rubric_comparison_html}
        </div>
    </div>
    """


def generate_css(config: Dict[str, Any]) -> str:
    """Generate CSS with dynamic colors based on attack type."""
    primary = config['primary_color']
    gradient = config['gradient']

    return f"""
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1600px;
            margin: 20px auto;
            padding: 20px;
        }}
        .header, .summary-card, .case-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 24px;
        }}
        .header {{
            background: {gradient};
            color: white;
        }}
        .header h1 {{
            margin: 0 0 12px 0;
            color: white;
        }}
        .header p {{
            margin: 8px 0;
            opacity: 0.95;
        }}
        h1, h2, h3, h4 {{
            color: #4A4A4A;
            margin-top: 0;
        }}
        h2 {{
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .stat-item {{
            text-align: center;
            padding: 16px;
            background: #f9f9f9;
            border-radius: 6px;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: {primary};
            margin-bottom: 4px;
        }}
        .stat-label {{
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-top: 20px;
        }}
        @media (max-width: 1200px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .column {{
            min-width: 0;
        }}
        .column h3 {{
            color: {primary};
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        .conversation, .completion {{
            background: #f9f9f9;
            border-radius: 6px;
            padding: 16px;
            margin-top: 12px;
        }}
        .conversation-attacked {{
            background: #fffbeb;
            border-left: 4px solid {primary};
        }}
        .conversation-text {{
            white-space: pre-wrap;
            word-wrap: break-word;
            padding: 12px;
            background: white;
            border-radius: 4px;
        }}
        .message {{
            margin-bottom: 12px;
            padding: 12px;
            border-radius: 4px;
            background: white;
        }}
        .message-user {{
            border-left: 4px solid #2196F3;
        }}
        .message-assistant {{
            border-left: 4px solid #4CAF50;
        }}
        .message-content {{
            margin-top: 8px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .completion-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.8;
        }}
        .attack-info {{
            background: #FFF9C4;
            border-left: 4px solid #FBC02D;
            padding: 16px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .attack-details {{
            margin-top: 12px;
        }}
        .attack-details p {{
            margin: 8px 0;
        }}
        .strategy-badge {{
            display: inline-block;
            background: #dbeafe;
            color: #1e40af;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        .framing-badge {{
            display: inline-block;
            background: #fce7f3;
            color: #9f1239;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: 600;
            font-size: 0.8em;
            margin-left: 6px;
        }}
        .strategies-list {{
            margin-top: 10px;
        }}
        .strategy-item {{
            margin: 8px 0;
            padding: 10px;
            background: #f0f9ff;
            border-radius: 4px;
            border-left: 3px solid #3b82f6;
        }}
        .strategy-reason {{
            margin-top: 6px;
            font-size: 0.9em;
            color: #555;
        }}
        .keyword-badge {{
            display: inline-block;
            background: #fef3c7;
            color: #92400e;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.9em;
            margin-right: 6px;
            margin-bottom: 6px;
        }}
        .keywords-list {{
            margin-top: 10px;
        }}
        .info-block {{
            margin-top: 8px;
            padding: 12px;
            background: #f5f5f5;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9em;
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
            color: {primary};
        }}
        .rubric-full, .criterion-full {{
            margin-top: 8px;
            padding: 12px;
            background: #f5f5f5;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.9em;
        }}
        .attack-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 16px;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
            margin-left: 12px;
        }}
        .attack-success {{
            background: #10b981;
        }}
        .attack-failed {{
            background: {primary};
        }}
        .attack-unknown {{
            background: #6b7280;
        }}
        .targeted-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            background: #dbeafe;
            color: #1e40af;
            font-weight: 700;
            font-size: 0.8em;
            margin-left: 8px;
        }}
        .grade-points-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 1.1em;
            font-weight: 700;
            background: #f0f0f0;
            color: #666;
        }}
        .grade-points-badge.points-positive {{
            background: #d1fae5;
            color: #065f46;
        }}
        .grade-points-badge.points-negative {{
            background: #fee2e2;
            color: #991b1b;
        }}
        h4 {{
            margin-bottom: 12px;
            color: #555;
        }}
        .rubric-comparison-section {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }}
        .rubric-comparison-section h3 {{
            color: {primary};
            margin-bottom: 15px;
        }}
        .rubric-comparison-note {{
            background: #dbeafe;
            border-left: 4px solid {primary};
            padding: 12px;
            margin-bottom: 15px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .rubric-comparison-header {{
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 15px;
            padding: 12px 15px;
            background: {primary};
            color: white;
            font-weight: 700;
            border-radius: 6px;
            margin-bottom: 10px;
        }}
        .rubric-comparison-row {{
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 15px;
            padding: 15px;
            background: white;
            border-radius: 6px;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
            transition: all 0.2s;
        }}
        .rubric-comparison-row:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .rubric-comparison-row.degraded-row {{
            border-left: 4px solid #ef4444;
            background: #fef2f2;
        }}
        .rubric-comparison-row.non-attackable-row {{
            opacity: 0.7;
            background: #f9fafb;
        }}
        .not-attackable-col {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }}
        .not-attackable-badge {{
            padding: 6px 12px;
            background: #e5e7eb;
            color: #6b7280;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            text-align: center;
        }}
        .not-attackable-note {{
            font-size: 0.75em;
            color: #9ca3af;
            text-align: center;
            font-style: italic;
        }}
        .rubric-info-col {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .rubric-number {{
            font-weight: 700;
            color: {primary};
            font-size: 0.9em;
        }}
        .rubric-criterion-compare {{
            font-size: 0.9em;
        }}
        .rubric-criterion-compare summary {{
            cursor: pointer;
            font-weight: 500;
        }}
        .rubric-criterion-compare summary:hover {{
            color: {primary};
        }}
        .rubric-grade-col {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 10px;
            background: #f9fafb;
            border-radius: 4px;
        }}
        .grade-status-compare {{
            font-weight: 700;
            font-size: 1em;
            padding: 6px 10px;
            border-radius: 4px;
            text-align: center;
        }}
        .grade-status-compare.status-good {{
            background: #d1fae5;
            color: #065f46;
        }}
        .grade-status-compare.status-bad {{
            background: #fee2e2;
            color: #991b1b;
        }}
        .grade-explanation-details {{
            margin-top: 8px;
        }}
        .grade-explanation-details summary {{
            cursor: pointer;
            font-size: 0.85em;
            font-weight: 500;
            color: {primary};
            padding: 4px 0;
        }}
        .grade-explanation-details summary:hover {{
            color: {config['secondary_color']};
        }}
        .grade-explanation-full {{
            margin-top: 8px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
            font-size: 0.85em;
            color: #666;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .no-grades {{
            text-align: center;
            padding: 20px;
            color: #999;
        }}
        @media (max-width: 900px) {{
            .rubric-comparison-header,
            .rubric-comparison-row {{
                grid-template-columns: 1fr;
                gap: 10px;
            }}
            .rubric-comparison-header {{
                display: none;
            }}
            .rubric-grade-col {{
                position: relative;
            }}
            .rubric-grade-col::before {{
                content: attr(data-label);
                font-weight: 700;
                color: {primary};
                margin-bottom: 5px;
            }}
        }}
    """


def generate_html(results_path: str, output_path: str, attack_type: Optional[str] = None):
    """Generate the complete HTML visualization."""
    try:
        data = load_results(results_path)
    except FileNotFoundError:
        print(f"Error: File not found: {results_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {results_path}: {e}")
        return

    # Auto-detect attack type if not specified
    if attack_type is None:
        attack_type = detect_attack_type(data)
        print(f"Auto-detected attack type: {attack_type}")

    config = get_attack_config(attack_type)
    metadata = data.get('metadata', {})
    results = data.get('results', [])

    if not results:
        print("Warning: No results found in JSON file")

    # ==============================================================================
    # Detect and report records matching the non-comparable all-rubric signature signature.
    # ==============================================================================
    noncomparable_records = []
    for i, result in enumerate(results):
        if is_legacy_all_rubric_record(result):
            # Check if conversation actually unchanged
            original_conv = result.get('original_conversation', [])
            attacked_conv = result.get('attacked_conversation', [])
            conv_unchanged = is_conversation_unchanged(original_conv, attacked_conv)

            noncomparable_records.append({
                'index': i,
                'test_case_id': result.get('test_case_id', 'unknown'),
                'conversation_unchanged': conv_unchanged,
                'num_rubrics_evaluated': len(result.get('attacked_rubrics', [])),
                'selected_rubric_index': result.get('attack_result', {}).get('attack_metadata', {}).get('selected_rubric_index', -1)
            })

    if noncomparable_records:
        print(f"\n{'='*80}")
        print(f"⚠️  NON-COMPARABLE LEGACY RECORDS: {len(noncomparable_records)} cases affected ({len(noncomparable_records)/len(results)*100:.1f}%)")
        print(f"{'='*80}")
        print(f"These cases have:")
        print(f"  • applicable=True but selected_rubric_index=-1")
        print(f"  • Evaluated with ALL rubrics instead of being skipped")
        print(f"  • Records are not comparable under the selected-rubric contract")
        print(f"\n{'─'*80}")
        print(f"AFFECTED TEST CASES:")
        print(f"{'─'*80}\n")

        for record in noncomparable_records:
            unchanged_flag = "🔴 UNCHANGED" if record['conversation_unchanged'] else "🟡 MODIFIED"

            # Get scores
            result = results[record['index']]
            baseline_score = result.get('baseline_score')
            attacked_score = result.get('attacked_score')

            baseline_str = f"{baseline_score:.3f}" if baseline_score is not None else "N/A"
            attacked_str = f"{attacked_score:.3f}" if attacked_score is not None else "N/A"

            score_change = None
            if baseline_score is not None and attacked_score is not None:
                score_change = baseline_score - attacked_score
            score_str = f"Δ={score_change:+.3f}" if score_change is not None else "Δ=N/A"

            attack_strategy = result.get('attack_result', {}).get('attack_strategy', 'unknown')

            print(f"  [{record['index']:3d}] {record['test_case_id']}")
            print(f"        Strategy: {attack_strategy}")
            print(f"        Status: {unchanged_flag}")
            print(f"        Rubrics: {record['num_rubrics_evaluated']} evaluated (selected-rubric contract: 1)")
            print(f"        Scores: baseline={baseline_str}, attacked={attacked_str}, {score_str}")
            print()

        print(f"⚠️  EXCLUDED from success rate calculation. HTML statistics based on {len(results) - len(noncomparable_records)} comparable records.")
        print(f"{'='*80}\n")

    # Create summary statistics
    summary_html = create_summary_stats(results, attack_type)

    # Create case cards
    case_cards_html = "".join([create_case_card(res, attack_type) for res in results])

    # Extract metadata with canonical nested fields and supported input fallbacks.
    testee_model = _healthbench_model_id(metadata, "testee") or 'Unknown'
    grader_model = _healthbench_model_id(metadata, "grader") or 'Unknown'
    dataset = healthbench_dataset_path(metadata) or 'Unknown'
    timestamp = metadata.get('timestamp', 'Unknown')
    total_processed = metadata.get('total_processed', len(results))
    version = metadata.get('version', attack_type)

    # Extract attack strategy model IDs.
    attack_strategies = healthbench_attack_strategy_order(metadata)
    attacker_strategies_config = healthbench_attack_strategy_config(metadata)

    attack_strategies_with_models = []
    for strategy in attack_strategies:
        strategy_name = strategy.replace('_', ' ').title()
        strategy_config = attacker_strategies_config.get(strategy, {})
        model_id = strategy_config.get('model_id', 'Unknown')
        attack_strategies_with_models.append(f"{strategy_name}[{model_id}]")

    attacker_model = ", ".join(attack_strategies_with_models) if attack_strategies_with_models else "Unknown"

    # Generate CSS
    css = generate_css(config)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config['name']} Comparison Report</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{config['name']} Comparison Report</h1>
            <p><strong>Testee Model:</strong> {escape_html(testee_model)}</p>
            <p><strong>Grader Model:</strong> {escape_html(grader_model)}</p>
            <p><strong>Attacker Model:</strong> {escape_html(attacker_model)}</p>
            <p><strong>Dataset:</strong> {escape_html(dataset)}</p>
            <p><strong>Cases Processed:</strong> {total_processed}</p>
            <p><strong>Version:</strong> {escape_html(version)}</p>
            <p><strong>Timestamp:</strong> {escape_html(timestamp)}</p>
        </div>

        {summary_html}

        {case_cards_html}

    </div>
</body>
</html>"""

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ Generated HTML report ({config['name']}): {output_path}")
    except IOError as e:
        print(f"Error: Could not write to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML visualization for attack results (unified for distraction/cognitive bias)"
    )
    parser.add_argument(
        "results",
        help="Path to the results JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output HTML file path (default: results file with .html extension)"
    )
    parser.add_argument(
        "-t", "--type",
        choices=['cognitive_bias', 'distraction'],
        help="Attack type (default: auto-detect from JSON)"
    )

    args = parser.parse_args()

    results_path = Path(args.results)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.with_suffix('.html')

    generate_html(str(results_path), str(output_path), args.type)


if __name__ == "__main__":
    main()
