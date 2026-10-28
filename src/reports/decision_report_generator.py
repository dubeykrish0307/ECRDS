from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "decision_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def decision_label(risk_prob: float) -> str:
    if risk_prob < 0.3:
        return "APPROVE"
    if risk_prob < 0.6:
        return "MANUAL REVIEW"
    return "REJECT"


def generate_decision_report_file(applicant_index: int, risk_prob: float, justification_text: str) -> Path:
    now = datetime.now(timezone.utc)

    content = f"""
CREDIT RISK DECISION REPORT
===========================

Applicant Index: {applicant_index}
Timestamp (UTC): {now:%Y-%m-%d %H:%M:%S}

Risk Probability:
-----------------
Estimated Default Risk: {risk_prob:.2%}

Decision:
---------
{decision_label(risk_prob)}

Decision Justification:
-----------------------
{justification_text}

Notes:
------
- This decision was generated using an explainable machine learning model.
- Feature-level contributions were computed using SHAP.
- This report is intended for audit and review purposes.
"""

    report_path = REPORT_DIR / f"decision_report_{applicant_index}_{int(now.timestamp())}.txt"
    report_path.write_text(content.strip())
    return report_path
