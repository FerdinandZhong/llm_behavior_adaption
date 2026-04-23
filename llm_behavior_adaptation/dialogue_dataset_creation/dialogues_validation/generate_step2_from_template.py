#!/usr/bin/env python3
"""
Generate Step 2 QSF files by cloning the working Career-A export and
replacing only the profile/dialogue DB question text with Ann2's 25 samples.

Output:
  DialogueEval_Career_Step2_B.qsf    (career,     Ann2's 25 samples)
  DialogueEval_Investment_Step2_B.qsf (investment, Ann2's 25 samples)
"""

import copy
import html
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
ANN_PKG = BASE / "human_annotation_package"
TEMPLATE = BASE / "Dialogue_Dataset_Review_-_Career_-_A.qsf"

# index of the first real dialogue block in BL Payload (0=Default, 1=Trash, 2=Template)
FIRST_DIALOGUE_BLOCK = 3


# ── HTML helpers ──────────────────────────────────────────────────────────────


def profile_html(profile_text: str) -> str:
    lines = [html.escape(line) for line in profile_text.strip().splitlines()]
    return "User Profile<br />\n<br />\n" + "<br />\n".join(lines)


def dialogue_html(dialogue_text: str) -> str:
    lines = [html.escape(line) for line in dialogue_text.strip().splitlines() if line.strip()]
    return "Dialogue<br />\n<br />\n" + "<br />\n".join(lines)


# ── Main builder ──────────────────────────────────────────────────────────────


def build_step2_qsf(csv_path: Path, survey_name: str, survey_id: str, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    rows = df.to_dict(orient="records")
    assert len(rows) == 25, f"Expected 25 rows, got {len(rows)}"

    with open(TEMPLATE) as f:
        qsf = json.load(f)

    qsf = copy.deepcopy(qsf)

    # Update survey metadata
    qsf["SurveyEntry"]["SurveyName"] = survey_name
    qsf["SurveyEntry"]["SurveyID"] = survey_id

    # Build QID→SQ element index
    elements = qsf["SurveyElements"]
    sq_index = {e["PrimaryAttribute"]: e for e in elements if e["Element"] == "SQ"}

    # Update SurveyID in all elements
    for elem in elements:
        elem["SurveyID"] = survey_id

    # Get BL payload
    bl = next(e for e in elements if e["Element"] == "BL")
    blocks = bl["Payload"]

    for i, row in enumerate(rows):
        blk = blocks[FIRST_DIALOGUE_BLOCK + i]
        sid = int(row["Sample_ID"])

        # Update block description
        blk["Description"] = f"Dialogue {i+1:02d} (ID {sid})"

        # Get the 8 question IDs for this block
        qids = [be["QuestionID"] for be in blk["BlockElements"] if be.get("Type") == "Question"]
        assert len(qids) == 8, f"Block {i} has {len(qids)} questions, expected 8"

        profile_q = sq_index[qids[0]]
        dialogue_q = sq_index[qids[1]]

        profile_q["Payload"]["QuestionText"] = profile_html(str(row["User_Profile"]))
        dialogue_q["Payload"]["QuestionText"] = dialogue_html(str(row["Dialogue"]))

        # Keep SecondaryAttribute (question preview) in sync
        profile_q["SecondaryAttribute"] = profile_q["Payload"]["QuestionText"][:30].replace("\n", " ")
        dialogue_q["SecondaryAttribute"] = dialogue_q["Payload"]["QuestionText"][:30].replace("\n", " ")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(qsf, f, indent=2, ensure_ascii=False)
    print(f"  Written: {out_path}")


def main():
    configs = [
        (
            ANN_PKG / "career_50/career_50_ann_b_new25.csv",
            "Dialogue Quality Evaluation - Career (Step 2)",
            "SV_CareerStep2B001",
            BASE / "DialogueEval_Career_Step2_B.qsf",
        ),
        (
            ANN_PKG / "investment_50/investment_50_ann_b_new25.csv",
            "Dialogue Quality Evaluation - Investment (Step 2)",
            "SV_InvestStep2B001",
            BASE / "DialogueEval_Investment_Step2_B.qsf",
        ),
    ]
    for csv_path, survey_name, survey_id, out_path in configs:
        print(f"{survey_name}")
        build_step2_qsf(csv_path, survey_name, survey_id, out_path)


if __name__ == "__main__":
    main()
