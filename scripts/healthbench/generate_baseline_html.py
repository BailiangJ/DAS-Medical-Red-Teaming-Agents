"""
Generate HTML visualization for HealthBench baseline evaluation results.

This script creates an interactive HTML viewer that combines baseline results
with the original dataset to show model completions and rubric grading.

Usage:
    python generate_baseline_html.py <baseline_results.json> <dataset.jsonl> [-o output.html]
"""

import json
import html
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict

from med_red_team.healthbench.data import (
    _healthbench_model_id,
    healthbench_filter_single_turn,
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


def format_markdown_text(text: str) -> str:
    """
    Prepare text for markdown rendering via marked.js.
    Just escape HTML entities - the actual markdown rendering will be done client-side.
    """
    if not text:
        return ""
    return escape_html(text)


def extract_first_sentence(text: str, max_chars: int = 150) -> str:
    """Extract the first sentence or truncate to max_chars."""
    if not text:
        return ""

    # Try to find first sentence
    for delimiter in ['. ', '.\n', '? ', '!\n', '! ']:
        idx = text.find(delimiter)
        if idx != -1 and idx < max_chars:
            return text[:idx + 1]

    # If no sentence found or too long, just truncate
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def load_baseline_results(file_path: str) -> Dict[str, Any]:
    """Load baseline results JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_dataset(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load JSONL dataset and create mapping from prompt_id to entry."""
    dataset_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                prompt_id = entry.get('prompt_id')
                if prompt_id:
                    dataset_map[prompt_id] = entry
    return dataset_map


def collect_all_tags(results: List[Dict[str, Any]]) -> tuple[Set[str], Set[str]]:
    """Collect all unique example tags and rubric tags from the results."""
    example_tags = set()
    rubric_tags = set()

    for result in results:
        # Get example tags from original rubrics
        for rubric in result.get('original_rubrics', []):
            rubric_tags.update(rubric.get('tags', []))

    return example_tags, rubric_tags


def group_tags_by_prefix(tags: Set[str]) -> Dict[str, List[str]]:
    """Group tags by their prefix (before the colon)."""
    grouped = defaultdict(list)

    for tag in sorted(tags):
        if ':' in tag:
            prefix = tag.split(':', 1)[0]
            grouped[prefix].append(tag)
        else:
            grouped['other'].append(tag)

    return dict(grouped)


def create_filter_section(example_tags: Set[str], rubric_tags: Set[str]) -> str:
    """Create the filter panel HTML."""
    rubric_groups = group_tags_by_prefix(rubric_tags)

    html_parts = ["""
        <div class="filters-panel">
            <h2>Filters</h2>

            <div class="filter-actions">
                <button onclick="selectAllFilters()">Select All</button>
                <button onclick="deselectAllFilters()">Deselect All</button>
                <button onclick="resetFilters()">Reset Filters</button>
            </div>

            <div class="search-box">
                <label for="keyword-search">Keyword Search:</label>
                <input type="text" id="keyword-search" placeholder="Search in all content..." onkeyup="applyFilters()">
            </div>

            <div class="filter-section">
                <h3>Rubric Tags</h3>
    """]

    # Add rubric tag checkboxes grouped by prefix
    for prefix, tags in sorted(rubric_groups.items()):
        html_parts.append(f'<div class="tag-group"><h4>{escape_html(prefix)}</h4>')
        for tag in tags:
            tag_id = f"rub-{tag.replace(':', '-').replace(' ', '_')}"
            html_parts.append(f'''
                <label class="checkbox-label">
                    <input type="checkbox" class="rubric-tag-filter" data-tag="{escape_html(tag)}" id="{tag_id}" checked onchange="applyFilters()">
                    <span>{escape_html(tag)}</span>
                </label>
            ''')
        html_parts.append('</div>')

    html_parts.append("""
            </div>
        </div>
    """)

    return ''.join(html_parts)


def create_result_card(result: Dict[str, Any], dataset_entry: Dict[str, Any], index: int) -> str:
    """Create HTML for a single result card."""
    test_case_id = result.get('test_case_id', '')
    conversation = result.get('original_conversation', [])
    baseline_completion = result.get('baseline_completion', '')
    baseline_grades = result.get('baseline_grades', [])
    baseline_score = result.get('baseline_score')
    baseline_score = 0.0 if baseline_score is None else baseline_score

    # Get dataset info
    ideal_completion_data = {}
    if dataset_entry:
        ideal_completion_data = dataset_entry.get('ideal_completions_data') or {}
    ideal_completion = ideal_completion_data.get('ideal_completion', '')
    ref_completions = ideal_completion_data.get('ideal_completions_ref_completions', []) or []
    example_tags = dataset_entry.get('example_tags', []) if dataset_entry else []

    # Collect all rubric tags for filtering
    all_rubric_tags = set()
    for grade in baseline_grades:
        all_rubric_tags.update(grade.get('tags', []))
    rubric_tags_str = ','.join(all_rubric_tags)
    example_tags_str = ','.join(example_tags)

    # Build card HTML
    html_parts = [f'''
        <div class="entry-card" data-index="{index}" data-example-tags="{escape_html(example_tags_str)}" data-rubric-tags="{escape_html(rubric_tags_str)}">
            <div class="entry-header">
                <div class="score-badge">
                    <span class="score-label">Baseline Score:</span>
                    <span class="score-value">{baseline_score:.3f}</span>
                </div>
                <div class="tag-badges">
    ''']

    # Add example tag badges
    for tag in example_tags:
        tag_class = tag.split(':')[0] if ':' in tag else 'other'
        html_parts.append(f'<span class="badge badge-{escape_html(tag_class)}">{escape_html(tag)}</span>')

    html_parts.append('''
                </div>
            </div>

            <div class="content-section">
                <h3>Prompt / Conversation</h3>
                <div class="conversation">
    ''')

    # Add conversation messages
    for msg in conversation:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        formatted_content = format_text_preserve_newlines(content)
        html_parts.append(f'''
                    <div class="message message-{escape_html(role)}">
                        <div class="message-role">{escape_html(role).upper()}</div>
                        <div class="message-content">{formatted_content}</div>
                    </div>
        ''')

    html_parts.append('''
                </div>
            </div>

            <div class="content-section collapsible-section">
                <h3 class="toggle-header" onclick="toggleSection(this)">
                    Model Completion <span class="toggle-icon">▼</span>
                </h3>
                <div class="collapsible-content">
    ''')

    # Add model completion
    html_parts.append(f'<div class="completion-text markdown-content" data-markdown="{escape_html(baseline_completion)}"></div>')

    html_parts.append('''
                </div>
            </div>
    ''')

    # Ideal completion section (collapsible)
    if ideal_completion:
        html_parts.append(f'''
            <div class="content-section collapsible-section">
                <h3 class="toggle-header" onclick="toggleSection(this)">
                    Ideal Completion <span class="toggle-icon">▼</span>
                </h3>
                <div class="collapsible-content">
                    <div class="completion-text markdown-content" data-markdown="{escape_html(ideal_completion)}"></div>
                </div>
            </div>
        ''')

    # Reference completions section (collapsible)
    if ref_completions and len(ref_completions) > 0:
        html_parts.append(f'''
            <div class="content-section collapsible-section">
                <h3 class="toggle-header" onclick="toggleSection(this)">
                    Reference Completions ({len(ref_completions)}) <span class="toggle-icon">▼</span>
                </h3>
                <div class="collapsible-content">
        ''')

        for i, ref_completion in enumerate(ref_completions, 1):
            html_parts.append(f'''
                    <div class="reference-completion">
                        <h4>Reference {i}</h4>
                        <div class="completion-text markdown-content" data-markdown="{escape_html(ref_completion)}"></div>
                    </div>
            ''')

        html_parts.append('''
                </div>
            </div>
        ''')

    # Rubrics with grading section
    if baseline_grades:
        # Calculate met/unmet counts
        met_count = sum(1 for g in baseline_grades if g.get('criteria_met', False))
        unmet_count = len(baseline_grades) - met_count

        html_parts.append(f'''
            <div class="content-section collapsible-section">
                <h3 class="toggle-header" onclick="toggleSection(this)">
                    Rubrics & Grading ({len(baseline_grades)} total | <span class="met-count">{met_count} MET</span> | <span class="unmet-count">{unmet_count} UNMET</span>) <span class="toggle-icon">▼</span>
                </h3>
                <div class="collapsible-content">
                    <div class="rubrics-container">
        ''')

        for i, grade in enumerate(baseline_grades):
            criterion = grade.get('criterion', '')
            points = grade.get('points', 0)
            rubric_tags = grade.get('tags', [])
            criteria_met = grade.get('criteria_met', False)
            explanation = grade.get('explanation', '')

            first_sentence = extract_first_sentence(criterion)
            full_criterion = escape_html(criterion)

            # Determine rubric card styling based on criteria_met and points
            # For negative points, invert the badge colors (met=red, unmet=green)
            # For positive points, normal colors (met=green, unmet=red)
            is_negative_points = points < 0

            if is_negative_points:
                # Negative points: meeting them is bad (red), not meeting is good (green)
                if criteria_met:
                    status_class = "met-negative"
                    status_label = "✓ MET"
                else:
                    status_class = "unmet-negative"
                    status_label = "✗ UNMET"
            else:
                # Positive points: meeting them is good (green), not meeting is bad (red)
                if criteria_met:
                    status_class = "met"
                    status_label = "✓ MET"
                else:
                    status_class = "unmet"
                    status_label = "✗ UNMET"

            # Card always has green background
            card_class = f"rubric-card"

            points_class = "rubric-points negative" if points < 0 else "rubric-points"

            html_parts.append(f'''
                    <div class="{card_class}">
                        <div class="rubric-header">
                            <div class="rubric-status rubric-status-{status_class}">{status_label}</div>
                            <div class="{points_class}">Points: {points}</div>
                            <div class="rubric-tags">
            ''')

            for tag in rubric_tags:
                tag_class = tag.split(':')[0] if ':' in tag else 'other'
                html_parts.append(f'<span class="badge badge-small badge-{escape_html(tag_class)}">{escape_html(tag)}</span>')

            html_parts.append(f'''
                            </div>
                        </div>
                        <div class="rubric-criterion">
                            <details class="criterion-details">
                                <summary class="criterion-preview">{escape_html(first_sentence)}</summary>
                                <div class="criterion-full">{full_criterion}</div>
                            </details>
                        </div>
            ''')

            # Add grader explanation if present
            if explanation:
                html_parts.append(f'''
                        <div class="grader-explanation">
                            <strong>Grader Explanation:</strong>
                            <div class="explanation-text">{format_text_preserve_newlines(explanation)}</div>
                        </div>
                ''')

            html_parts.append('''
                    </div>
            ''')

        html_parts.append('''
                    </div>
                </div>
            </div>
        ''')

    html_parts.append('</div>')

    return ''.join(html_parts)


def generate_html(baseline_path: str, dataset_path: str, output_path: str):
    """Generate the complete HTML visualization."""

    # Load baseline results and dataset
    baseline_data = load_baseline_results(baseline_path)
    dataset_map = load_dataset(dataset_path)

    results = baseline_data.get('results', [])
    metadata = baseline_data.get('metadata', {})
    summary = baseline_data.get('summary', {})
    testee_model = _healthbench_model_id(metadata, "testee") or 'N/A'
    grader_model = _healthbench_model_id(metadata, "grader") or 'N/A'
    filter_single_turn = healthbench_filter_single_turn(metadata)
    if filter_single_turn is None:
        filter_single_turn = False

    total_entries = len(results)

    # Calculate positive rubrics met and negative rubrics unmet
    positive_rubrics_met = 0
    negative_rubrics_unmet = 0
    for result in results:
        baseline_grades = result.get('baseline_grades', [])
        for grade in baseline_grades:
            points = grade.get('points', 0)
            criteria_met = grade.get('criteria_met', False)

            # Count positive rubrics that were met
            if points > 0 and criteria_met:
                positive_rubrics_met += 1
            # Count negative rubrics that were unmet (which is good)
            elif points < 0 and not criteria_met:
                negative_rubrics_unmet += 1

    # Add these to summary if not already present
    if 'positive_rubrics_met_baseline' not in summary:
        summary['positive_rubrics_met_baseline'] = positive_rubrics_met
    if 'negative_rubrics_unmet_baseline' not in summary:
        summary['negative_rubrics_unmet_baseline'] = negative_rubrics_unmet

    # Collect all tags
    example_tags, rubric_tags = collect_all_tags(results)

    # Build HTML
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HealthBench Baseline Results Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f7fa;
            color: #333;
            line-height: 1.6;
            overflow-x: hidden;
            max-width: 100vw;
        }}

        .container {{
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            overflow-x: hidden;
            width: 100%;
        }}

        .header {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}

        .metadata {{
            grid-column: 1 / -1;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }}

        .metadata h3 {{
            color: #667eea;
            margin-bottom: 10px;
        }}

        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .metadata-item {{
            display: flex;
            flex-direction: column;
        }}

        .metadata-label {{
            font-size: 0.85em;
            color: #666;
            font-weight: 600;
        }}

        .metadata-value {{
            font-size: 1.1em;
            color: #333;
        }}

        .summary {{
            grid-column: 1 / -1;
            display: flex;
            gap: 20px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}

        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 150px;
        }}

        .summary-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .summary-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }}

        .filters-panel {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: sticky;
            top: 20px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
        }}

        .filters-panel h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}

        .filters-panel h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.1em;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
        }}

        .filters-panel h4 {{
            color: #777;
            margin-top: 12px;
            margin-bottom: 8px;
            font-size: 0.9em;
            font-weight: 600;
        }}

        .filter-actions {{
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}

        .filter-actions button {{
            padding: 8px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85em;
            transition: background 0.2s;
        }}

        .filter-actions button:hover {{
            background: #5568d3;
        }}

        .search-box {{
            margin-bottom: 15px;
        }}

        .search-box label {{
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }}

        .search-box input {{
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 0.95em;
        }}

        .filter-section {{
            margin-bottom: 20px;
        }}

        .tag-group {{
            margin-bottom: 15px;
        }}

        .checkbox-label {{
            display: block;
            padding: 4px 0;
            cursor: pointer;
            font-size: 0.9em;
        }}

        .checkbox-label:hover {{
            background: #f0f0f0;
            padding-left: 5px;
        }}

        .checkbox-label input {{
            margin-right: 8px;
            cursor: pointer;
        }}

        .entries-container {{
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-x: hidden;
            max-width: 100%;
        }}

        .entry-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: box-shadow 0.3s;
            overflow-x: hidden;
            max-width: 100%;
        }}

        .entry-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .entry-card.hidden {{
            display: none;
        }}

        .entry-header {{
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .score-badge {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }}

        .score-label {{
            font-size: 0.9em;
        }}

        .score-value {{
            font-size: 1.3em;
        }}

        .tag-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            color: white;
        }}

        .badge-small {{
            padding: 4px 8px;
            font-size: 0.75em;
        }}

        .badge-theme {{ background: #3b82f6; }}
        .badge-physician_agreed_category {{ background: #10b981; }}
        .badge-level {{ background: #f59e0b; }}
        .badge-cluster {{ background: #8b5cf6; }}
        .badge-axis {{ background: #ec4899; }}
        .badge-other {{ background: #6b7280; }}

        .content-section {{
            margin-bottom: 20px;
        }}

        .content-section h3 {{
            color: #667eea;
            margin-bottom: 12px;
            font-size: 1.2em;
        }}

        .met-count {{
            color: #10b981;
            font-weight: bold;
        }}

        .unmet-count {{
            color: #ef4444;
            font-weight: bold;
        }}

        .collapsible-section .toggle-header {{
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            transition: background 0.2s;
        }}

        .collapsible-section .toggle-header:hover {{
            background: #e9ecef;
        }}

        .toggle-icon {{
            transition: transform 0.3s;
            font-size: 0.8em;
        }}

        .toggle-icon.rotated {{
            transform: rotate(-180deg);
        }}

        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .collapsible-content.open {{
            max-height: 10000px;
            padding-top: 15px;
        }}

        .conversation {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }}

        .message {{
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 6px;
        }}

        .message-user {{
            background: #e0e7ff;
            border-left: 3px solid #4f46e5;
        }}

        .message-assistant {{
            background: #f3e8ff;
            border-left: 3px solid #7c3aed;
        }}

        .message-role {{
            font-weight: bold;
            font-size: 0.85em;
            margin-bottom: 5px;
            color: #555;
        }}

        .message-content {{
            line-height: 1.6;
        }}

        .completion-text {{
            background: #f9fafb;
            padding: 15px;
            border-radius: 6px;
            line-height: 1.6;
            font-size: 0.9em;
            overflow-x: auto;
            max-width: 100%;
        }}

        .completion-text ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}

        .completion-text ol {{
            margin: 10px 0;
            padding-left: 25px;
        }}

        .completion-text li {{
            margin: 5px 0;
        }}

        .completion-text strong {{
            color: #555;
        }}

        .completion-text h1,
        .completion-text h2,
        .completion-text h3,
        .completion-text h4 {{
            color: #667eea;
            margin-top: 15px;
            margin-bottom: 10px;
        }}

        .completion-text h1 {{ font-size: 1.4em; }}
        .completion-text h2 {{ font-size: 1.2em; }}
        .completion-text h3 {{ font-size: 1.1em; }}
        .completion-text h4 {{ font-size: 1.0em; }}

        .table-wrapper {{
            overflow-x: auto;
            max-width: 100%;
            margin: 10px 0;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }}

        .completion-text table {{
            border-collapse: collapse;
            margin: 0;
            width: max-content;
            min-width: 100%;
            font-size: 0.85em;
        }}

        .completion-text table th,
        .completion-text table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}

        .completion-text table th {{
            background: #f0f0f0;
            font-weight: 600;
        }}

        .completion-text hr {{
            border: none;
            border-top: 1px solid #ddd;
            margin: 15px 0;
        }}

        .completion-text p {{
            margin: 10px 0;
        }}

        .completion-text code {{
            background: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}

        .completion-text pre {{
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            max-width: 100%;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .completion-text pre code {{
            background: none;
            padding: 0;
        }}

        .completion-text blockquote {{
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin: 10px 0;
            color: #666;
            font-style: italic;
        }}

        .reference-completion {{
            margin-bottom: 20px;
            padding: 15px;
            background: #fefefe;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            overflow-x: auto;
            max-width: 100%;
        }}

        .reference-completion h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1em;
        }}

        .rubrics-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}

        .rubric-card {{
            background: #f0fdf4;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #10b981;
        }}

        .rubric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .rubric-status {{
            font-weight: bold;
            font-size: 1em;
            padding: 4px 12px;
            border-radius: 5px;
            color: white;
        }}

        .rubric-status-met {{
            background: #10b981;
        }}

        .rubric-status-unmet {{
            background: #ef4444;
        }}

        .rubric-status-met-negative {{
            background: #ef4444;
        }}

        .rubric-status-unmet-negative {{
            background: #10b981;
        }}

        .rubric-points {{
            font-weight: bold;
            color: #10b981;
            font-size: 1.1em;
        }}

        .rubric-points.negative {{
            color: #ef4444;
        }}

        .rubric-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}

        .rubric-criterion {{
            margin-top: 10px;
        }}

        .criterion-details {{
            cursor: pointer;
        }}

        .criterion-details summary {{
            color: #555;
            line-height: 1.5;
            padding: 4px 0;
            font-weight: 500;
            cursor: pointer;
            user-select: none;
        }}

        .criterion-details summary:hover {{
            color: #667eea;
        }}

        .criterion-preview {{
            display: inline;
        }}

        .criterion-full {{
            margin-top: 10px;
            padding: 12px;
            background: #f5f5f5;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.85em;
            line-height: 1.5;
            color: #333;
        }}

        .grader-explanation {{
            margin-top: 12px;
            padding: 12px;
            background: #fffbeb;
            border-left: 3px solid #f59e0b;
            border-radius: 4px;
        }}

        .grader-explanation strong {{
            color: #92400e;
            display: block;
            margin-bottom: 6px;
        }}

        .explanation-text {{
            font-size: 0.9em;
            line-height: 1.5;
            color: #333;
        }}

        .no-results {{
            grid-column: 1 / -1;
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 8px;
            color: #666;
        }}

        .no-results h2 {{
            color: #667eea;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HealthBench Baseline Results Viewer</h1>
            <p>Interactive visualization of baseline evaluation results</p>
        </div>

        <div class="metadata">
            <h3>Evaluation Metadata</h3>
            <div class="metadata-grid">
                <div class="metadata-item">
                    <span class="metadata-label">Testee Model</span>
                    <span class="metadata-value">{escape_html(testee_model)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Grader Model</span>
                    <span class="metadata-value">{escape_html(grader_model)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Timestamp</span>
                    <span class="metadata-value">{escape_html(metadata.get('timestamp', 'N/A'))}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Total Cases</span>
                    <span class="metadata-value">{metadata.get('total_cases_evaluated', 0)}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Filter Single-Turn</span>
                    <span class="metadata-value">{'Yes' if filter_single_turn else 'No'}</span>
                </div>
            </div>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Total Cases</h3>
                <div class="value" id="total-entries">{total_entries}</div>
            </div>
            <div class="summary-card">
                <h3>Visible Cases</h3>
                <div class="value" id="visible-entries">{total_entries}</div>
            </div>
            <div class="summary-card">
                <h3>Baseline Avg Score</h3>
                <div class="value">{summary.get('baseline_avg_score', 0.0):.3f}</div>
            </div>
            <div class="summary-card">
                <h3>Positive Rubrics Met</h3>
                <div class="value">{summary.get('positive_rubrics_met_baseline', 0)}</div>
            </div>
            <div class="summary-card">
                <h3>Negative Rubrics Unmet</h3>
                <div class="value">{summary.get('negative_rubrics_unmet_baseline', 0)}</div>
            </div>
        </div>

        {create_filter_section(example_tags, rubric_tags)}

        <div class="entries-container" id="entries-container">
'''

    # Add all result cards
    for i, result in enumerate(results):
        test_case_id = result.get('test_case_id', '')
        dataset_entry = dataset_map.get(test_case_id)
        html_content += create_result_card(result, dataset_entry, i)

    html_content += '''
        </div>

        <div class="no-results" id="no-results" style="display: none;">
            <h2>No entries match your filters</h2>
            <p>Try adjusting your filter selections or search terms.</p>
        </div>
    </div>

    <script>
        function toggleSection(header) {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');

            content.classList.toggle('open');
            icon.classList.toggle('rotated');
        }

        function applyFilters() {
            const keywordSearch = document.getElementById('keyword-search').value.toLowerCase();

            // Get selected rubric tags
            const selectedRubricTags = new Set();
            document.querySelectorAll('.rubric-tag-filter:checked').forEach(checkbox => {
                selectedRubricTags.add(checkbox.dataset.tag);
            });

            // Filter entries
            const entries = document.querySelectorAll('.entry-card');
            let visibleCount = 0;

            entries.forEach(entry => {
                let shouldShow = true;

                // Check rubric tags - only filter if at least one tag is selected
                if (selectedRubricTags.size > 0) {
                    const entryRubricTags = entry.dataset.rubricTags.split(',').filter(t => t);
                    const hasMatchingRubricTag = entryRubricTags.some(tag => selectedRubricTags.has(tag));
                    if (!hasMatchingRubricTag) {
                        shouldShow = false;
                    }
                }

                // Check keyword search
                if (shouldShow && keywordSearch) {
                    const entryText = entry.textContent.toLowerCase();
                    if (!entryText.includes(keywordSearch)) {
                        shouldShow = false;
                    }
                }

                // Show or hide entry
                if (shouldShow) {
                    entry.classList.remove('hidden');
                    visibleCount++;
                } else {
                    entry.classList.add('hidden');
                }
            });

            // Update visible count
            document.getElementById('visible-entries').textContent = visibleCount;

            // Show/hide no results message
            const noResults = document.getElementById('no-results');
            if (visibleCount === 0) {
                noResults.style.display = 'block';
            } else {
                noResults.style.display = 'none';
            }
        }

        function selectAllFilters() {
            document.querySelectorAll('.rubric-tag-filter').forEach(checkbox => {
                checkbox.checked = true;
            });
            applyFilters();
        }

        function deselectAllFilters() {
            document.querySelectorAll('.rubric-tag-filter').forEach(checkbox => {
                checkbox.checked = false;
            });
            applyFilters();
        }

        function resetFilters() {
            selectAllFilters();
            document.getElementById('keyword-search').value = '';
            applyFilters();
        }

        // Render all markdown content
        function renderMarkdown() {
            document.querySelectorAll('.markdown-content').forEach(element => {
                const markdownText = element.dataset.markdown;
                if (markdownText) {
                    // Use marked.js to convert markdown to HTML
                    element.innerHTML = marked.parse(markdownText);

                    // Wrap all tables in a scrollable container
                    element.querySelectorAll('table').forEach(table => {
                        if (!table.parentElement.classList.contains('table-wrapper')) {
                            const wrapper = document.createElement('div');
                            wrapper.className = 'table-wrapper';
                            table.parentNode.insertBefore(wrapper, table);
                            wrapper.appendChild(table);
                        }
                    });
                }
            });
        }

        // Initialize
        applyFilters();
        renderMarkdown();
    </script>
</body>
</html>
'''

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✓ Generated HTML visualization: {output_path}")
    print(f"  Total cases: {total_entries}")
    print(f"  Baseline avg score: {summary.get('baseline_avg_score', 0.0):.3f}")
    print(f"  Positive rubrics met: {summary.get('positive_rubrics_met_baseline', 0)}")
    print(f"  Negative rubrics unmet: {summary.get('negative_rubrics_unmet_baseline', 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML visualization for HealthBench baseline results")
    parser.add_argument("baseline_results", help="Path to the baseline results JSON file")
    parser.add_argument("dataset", help="Path to the dataset JSONL file")
    parser.add_argument("-o", "--output", help="Output HTML file path (default: baseline_results.html in same directory)")

    args = parser.parse_args()

    baseline_path = Path(args.baseline_results)
    dataset_path = Path(args.dataset)

    if not baseline_path.exists():
        print(f"Error: Baseline results file not found: {baseline_path}")
        exit(1)

    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = baseline_path.with_suffix('.html')

    generate_html(str(baseline_path), str(dataset_path), str(output_path))
