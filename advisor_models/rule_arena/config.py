"""Configuration for rule arena (US tax) domain.

Contains system prompts and templates for US tax calculation with advisor feedback.
"""

from typing import Dict, Any, Tuple
import sys
from pathlib import Path
import re
import numpy as np

STUDENT_SYSTEM_PROMPT = "You are a helpful US taxation consultant. End your response with: '1. The total tax owed is $xxx.' (xxx is a number) if there is tax owed. 2. The total tax overpaid is $xxx.' (xxx is a number) if there is tax overpaid (and should be refunded)."

ADVISOR_INSTRUCTIONS = "You are an advisor whose job is to review the solution to identify flaws and provide specific feedback to improve it if needed. Focus on accuracy and completeness."

PROBLEM_TEMPLATE = """You are given several forms used to report US income tax and the instructions or rules about how to fill the forms. Then you will be given the income and/or payment information about a tax payer According to the given information. You should calculate the income tax owed by this payer.

IRS Forms for the tax payer:
$forms
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:"""


def build_prompt(tax_payer_dict: Dict[str, Any] = None) -> str:
    """Build the initial student prompt."""
    # Import tax forms and example from RuleArena tax
    sys.path.append(str(Path(__file__).parent / "RuleArena" / "tax"))

    try:
        from prompt import (
            basic_forms,
            itemized_forms,
            self_employ_forms,
            edu_forms,
            schedule_8812,
        )
    except ImportError as e:
        raise ImportError(
            f"Failed to import required tax modules: {e}. "
            f"Make sure RuleArena tax files are available."
        )

    # Build forms based on taxpayer characteristics (like auto_test.py)
    forms = [basic_forms]

    # Extract the right data structures
    pydantic_data = tax_payer_dict.get("pydantic", {}) if tax_payer_dict else {}
    dict_data = tax_payer_dict.get("dict", {}) if tax_payer_dict else {}

    # Use pydantic data for taxpayer characteristics
    taxpayer_info = pydantic_data if pydantic_data else tax_payer_dict
    if taxpayer_info:
        if taxpayer_info.get("itemized", False):
            forms.append(itemized_forms)
        if taxpayer_info.get("self_employed", False):
            forms.append(self_employ_forms)
        if taxpayer_info.get("has_student_loans_or_education_expenses", False):
            forms.append(edu_forms)
        if taxpayer_info.get("child_and_dependent", False):
            forms.append(schedule_8812)

    forms_str = "".join(forms)

    # Replace data fields in forms using dict data (which has the 'data' field)
    if dict_data and "data" in dict_data:
        for k, v in dict_data["data"].items():
            forms_str = forms_str.replace(
                "$" + k, "$" + f"{v:,}" if not isinstance(v, str) else v
            )

    # Build the full prompt
    full_prompt = PROBLEM_TEMPLATE.replace("$forms", forms_str)

    # Replace taxpayer personal info if available
    if taxpayer_info:
        full_prompt = full_prompt.replace("$name", taxpayer_info.get("name", "Unknown"))
        full_prompt = full_prompt.replace("$age", str(taxpayer_info.get("age", "")))
        full_prompt = full_prompt.replace(
            "$spouse_age", str(taxpayer_info.get("spouse_age", ""))
        )
        full_prompt = full_prompt.replace(
            "$blind", str(taxpayer_info.get("blind", False))
        )
        full_prompt = full_prompt.replace(
            "$spouse_blind", str(taxpayer_info.get("spouse_blind", False))
        )
        full_prompt = full_prompt.replace(
            "$filing_status", taxpayer_info.get("filing_status", "Unknown")
        )
        full_prompt = full_prompt.replace(
            "$itemized", str(taxpayer_info.get("itemized", False))
        )
        full_prompt = full_prompt.replace(
            "$num_qualifying_children",
            str(taxpayer_info.get("num_qualifying_children", 0)),
        )
        full_prompt = full_prompt.replace(
            "$num_other_dependents", str(taxpayer_info.get("num_other_dependents", 0))
        )

    return full_prompt


def compute_score(response_str: str, ground_truth: str) -> Tuple[float, str]:
    """Compute the score."""
    pattern = (
        r"The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?)\.?"
    )
    match = re.search(pattern, response_str)
    if match:
        status = match.group(1)  # "owed" or "overpaid"
        value = float(match.group(2).replace(",", ""))
        # Return negative value for overpaid (as per RuleArena implementation)
        extracted_answer = str(-value if status == "overpaid" else value)
    else:
        return 0.0, "Failed to extract answer"

    try:
        extracted_val = float(extracted_answer.replace(",", "").replace("$", ""))
        ground_truth_val = float(str(ground_truth).replace(",", "").replace("$", ""))
        # Use numpy.isclose for comparison (matches RuleArena implementation)
        if np.isclose(extracted_val, ground_truth_val):
            return 1.0, "Reward calculation succeeded"
        else:
            return 0.0, "Reward calculation succeeded"
    except (ValueError, TypeError):
        return 0.0, "Reward calculation failed"
