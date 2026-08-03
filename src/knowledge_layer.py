"""
knowledge_layer.py — Stage 7: ParseOS Knowledge Layer Storage (v2.0)
===================================================================
Wraps extracted SOP data into the structured ParseOS Knowledge Layer schema,
enriching it with unique identifiers, telemetry placeholders (for Project 3),
risk assessments, and query metadata. Saves to JSON file in knowledge_layer/.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path

from src.config import KNOWLEDGE_PATH


def save_to_knowledge_layer(
    sop_data: dict,
    manual_name: str,
    category: str = "general",
    query: str = "",
) -> tuple[str, dict]:
    """
    Saves an extracted SOP into the ParseOS Knowledge Layer directory.

    Returns:
        tuple[str, dict]: Saved JSON file path and the complete Knowledge Layer dictionary.
    """
    os.makedirs(KNOWLEDGE_PATH, exist_ok=True)

    # Calculate overall risk level
    steps = sop_data.get("steps", [])
    risk_levels = [step.get("risk_level", "low").lower() for step in steps]
    if "high" in risk_levels:
        overall_risk = "high"
    elif "medium" in risk_levels:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    # Clean manual name for ID slug
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", manual_name).lower()
    slug = re.sub(r"_+", "_", slug).strip("_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    knowledge_id = f"KL_{slug}_{timestamp_str}"

    # Extract keywords from query
    keywords = list(set(re.findall(r"\b\w{3,}\b", query.lower())))

    is_insufficient = bool(sop_data.get("is_insufficient", False))

    kl_entry = {
        "knowledge_id": knowledge_id,
        "source_manual": manual_name,
        "machine_category": category,
        "query_context": query,
        "trigger_keywords": keywords,
        "overall_risk_level": overall_risk if not is_insufficient else "none",
        "is_insufficient_evidence": is_insufficient,
        "telemetry_triggers": [
            {
                "signal": "placeholder_signal",
                "threshold": "placeholder_threshold",
                "status": "unlinked_project3"
            }
        ],
        "created_at": datetime.now().isoformat(),
        "sop": sop_data,
    }

    file_name = f"{knowledge_id}.json"
    file_path = os.path.join(KNOWLEDGE_PATH, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(kl_entry, f, indent=2)

    print(f"  ✓ [Stage 7] Saved Knowledge Layer entry: {file_path}")
    return file_path, kl_entry
