#!/usr/bin/env python3
"""
Generate Qualtrics QSF files for Step 2 human annotation surveys.

Step 2 covers the 25 sample dialogues NOT in Ann1's set (Ann2's samples).
The QSF files can be imported directly into Qualtrics via:
  Create Survey → Import a QSF file

Q-number scheme mirrors Step 1 (Q_BASE=14, Q_STRIDE=8) so the same
parse_qualtrics_tidy() logic works on the Step 2 export unchanged.
"""

import html
import json
from pathlib import Path

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
ANN_PKG = BASE / "human_annotation_package"
OUT_DIR = BASE

Q_BASE = 14
Q_STRIDE = 8  # 6 rating/comment cols + 2 display/gap = 8 per block

METRIC_DEFS = [
    # (label, scale_range, description, choices_dict)
    (
        "Coverage",
        "0 – 5",
        "Does the dialogue cover the user's demographic attributes "
        "(age, education, occupation, SES, region) in contextually appropriate ways?",
        {str(i): {"Display": str(i)} for i in range(6)},
    ),
    (
        "Correctness",
        "0 – 5",
        "Is the demographic information embedded in the dialogue consistent "
        "with the user profile and free from contradictions?",
        {str(i): {"Display": str(i)} for i in range(6)},
    ),
    (
        "Diversity",
        "1 – 5",
        "Does the dialogue show variety in topics and conversational patterns, "
        "avoiding repetitive or formulaic exchanges?",
        {str(i): {"Display": str(i)} for i in range(1, 6)},
    ),
    (
        "Relevance",
        "1 – 5",
        "How well does the dialogue content align with the specified topic?",
        {str(i): {"Display": str(i)} for i in range(1, 6)},
    ),
    (
        "Naturalness",
        "1 – 5",
        "Does the dialogue read as a realistic human-chatbot interaction "
        "with appropriate turn-taking and coherent flow?",
        {str(i): {"Display": str(i)} for i in range(1, 6)},
    ),
]

INSTRUCTIONS_HTML = """\
<strong>Instructions</strong><br><br>
You will read short dialogues and evaluate their quality using a clear scoring rubric.<br><br>
Please read both the <strong>user profile</strong> and the <strong>dialogue</strong> carefully before scoring.<br><br>
<strong>Important scoring rules</strong><br>
- Score <strong>only</strong> based on information that is explicitly stated or clearly inferable from the dialogue.<br>
- Do <strong>not</strong> assume demographic attributes that are not present.<br>
- If you are unsure, give a <strong>lower score</strong>.<br><br>
There are no right or wrong answers. We are interested in your honest judgment.<br><br>
This task takes approximately <strong>6–8 minutes</strong>.
"""

CONSENT_CHOICES = {
    "1": {"Display": "I consent and wish to continue"},
    "2": {"Display": "I do not consent"},
}


# ── HTML helpers ──────────────────────────────────────────────────────────────


def profile_to_html(profile_text: str) -> str:
    lines = [html.escape(line.strip()) for line in profile_text.strip().splitlines() if line.strip()]
    return "<br>".join(lines)


def dialogue_to_html(dialogue_text: str) -> str:
    lines = []
    for line in dialogue_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        esc = html.escape(line)
        if line.startswith("[Turn") and "USER:" in line:
            esc = esc.replace("USER:", "<strong>USER:</strong>", 1)
        elif line.startswith("[Turn") and "ASSISTANT:" in line:
            esc = esc.replace("ASSISTANT:", "<strong>ASSISTANT:</strong>", 1)
        lines.append(esc)
    return "<br><br>".join(lines)


def dialogue_display_html(sample_id: int, topic: str, profile: str, dialogue: str) -> str:
    profile_html = profile_to_html(profile)
    dialogue_html = dialogue_to_html(dialogue)
    return (
        f"<hr><strong>Sample ID: {sample_id} &nbsp;|&nbsp; Topic: {topic}</strong><hr>"
        f"<table width='100%'><tr><td width='35%' valign='top' "
        f"style='background:#f0f4f8;padding:10px;border-radius:4px;'>"
        f"<strong>User Profile</strong><br><br>{profile_html}</td>"
        f"<td width='5%'></td>"
        f"<td width='60%' valign='top' "
        f"style='background:#fff9f0;padding:10px;border-radius:4px;'>"
        f"<strong>Dialogue</strong><br><br>{dialogue_html}</td></tr></table>"
        f"<br><strong>Please rate the dialogue on the five criteria below.</strong>"
    )


# ── QSF element factories ─────────────────────────────────────────────────────


def sq(
    qid: str,
    text: str,
    qtype: str,
    selector: str,
    export_tag: str = None,
    choices: dict = None,
    sub_selector: str = "TX",
    force: bool = True,
) -> dict:
    payload = {
        "QuestionText": text,
        "DataExportTag": export_tag or qid,
        "QuestionType": qtype,
        "Selector": selector,
    }
    if sub_selector:
        payload["SubSelector"] = sub_selector
    if choices is not None:
        payload["Choices"] = choices
        payload["Validation"] = {
            "Settings": {
                "ForceResponse": "ON" if force else "OFF",
                "Type": "None",
            }
        }
    return {
        "SurveyID": "SV_IMPORT",
        "Element": "SQ",
        "PrimaryAttribute": qid,
        "SecondaryAttribute": text[:30].replace("\n", " "),
        "TertiaryAttribute": None,
        "Payload": payload,
    }


def bl(block_id: str, description: str, question_ids: list) -> dict:
    return {
        "SurveyID": "SV_IMPORT",
        "Element": "BL",
        "PrimaryAttribute": description,
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "Type": "Default",
            "Description": description,
            "ID": block_id,
            "BlockElements": [{"Type": "Question", "QuestionID": qid} for qid in question_ids],
        },
    }


# ── QSF builder ───────────────────────────────────────────────────────────────


def build_qsf(topic_label: str, survey_name: str, survey_id: str, rows: list) -> dict:
    """
    rows: list of dicts with keys Sample_ID, User_Profile, Dialogue, Dialogue_Topic
    survey_id must match ^SV_[a-zA-Z0-9]{11,15}$
    """
    all_block_defs = []  # collected into single BL element at the end
    block_ids = []
    all_sq = []

    # ── Survey Options ────────────────────────────────────────────────────────
    so_element = {
        "SurveyID": survey_id,
        "Element": "SO",
        "PrimaryAttribute": "Survey Options",
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "BackButton": "false",
            "SaveAndContinue": "false",
            "AnonymizeResponse": "Yes",
            "SurveyProtection": "PublicSurvey",
            "SurveyExpiration": "None",
            "SurveyTermination": "DefaultMessage",
            "Header": "",
            "Footer": "",
        },
    }

    # ── Block 0: Instructions & Consent ──────────────────────────────────────
    q_instr = sq("QID1", INSTRUCTIONS_HTML, "DB", "TB", export_tag="Instructions", sub_selector=None)
    q_consent = sq(
        "QID2",
        "<strong>Consent</strong><br>I confirm that I am at "
        "least 18 years old and consent to participate in this study.",
        "MC",
        "SAVR",
        export_tag="Consent",
        choices=CONSENT_CHOICES,
        force=True,
    )
    for e in [q_instr, q_consent]:
        e["SurveyID"] = survey_id
    all_sq += [q_instr, q_consent]

    all_block_defs.append(
        {
            "Type": "Default",
            "SubType": "",
            "Description": "Instructions & Consent",
            "ID": "BL_1",
            "BlockElements": [{"Type": "Question", "QuestionID": "QID1"}, {"Type": "Question", "QuestionID": "QID2"}],
        }
    )
    block_ids.append("BL_1")

    # ── Blocks 1–25: one per dialogue ────────────────────────────────────────
    qid_counter = 3  # QID1, QID2 used above

    for block_idx, row in enumerate(rows):
        sid = int(row["Sample_ID"])
        profile = str(row["User_Profile"])
        dialogue = str(row["Dialogue"])
        dtopic = str(row["Dialogue_Topic"])

        blk_id = f"BL_{block_idx + 2}"
        blk_qids = []

        # Display question (not exported)
        disp_qid = f"QID{qid_counter}"
        qid_counter += 1
        disp_html = dialogue_display_html(sid, dtopic, profile, dialogue)
        q_disp = sq(disp_qid, disp_html, "DB", "TB", export_tag=f"{disp_qid}_display", sub_selector=None)
        q_disp["SurveyID"] = survey_id
        all_sq.append(q_disp)
        blk_qids.append(disp_qid)

        # 5 rating questions
        for offset, (metric, scale, desc, choices) in enumerate(METRIC_DEFS):
            export_q_num = Q_BASE + Q_STRIDE * block_idx + offset
            rating_qid = f"QID{qid_counter}"
            qid_counter += 1
            q_text = f"<strong>{metric}</strong> ({scale})<br>" f"<em>{desc}</em>"
            q_rating = sq(rating_qid, q_text, "MC", "SAVR", export_tag=f"Q{export_q_num}", choices=choices, force=True)
            q_rating["SurveyID"] = survey_id
            all_sq.append(q_rating)
            blk_qids.append(rating_qid)

        # Comment
        comment_q_num = Q_BASE + Q_STRIDE * block_idx + 5
        comment_qid = f"QID{qid_counter}"
        qid_counter += 1
        q_comment = sq(
            comment_qid,
            "Comments (optional):",
            "TE",
            "ML",
            export_tag=f"Q{comment_q_num}",
            sub_selector=None,
            choices=None,
            force=False,
        )
        q_comment["SurveyID"] = survey_id
        all_sq.append(q_comment)
        blk_qids.append(comment_qid)

        all_block_defs.append(
            {
                "Type": "Default",
                "SubType": "",
                "Description": f"Dialogue {block_idx + 1} (ID {sid})",
                "ID": blk_id,
                "BlockElements": [{"Type": "Question", "QuestionID": qid} for qid in blk_qids],
            }
        )
        block_ids.append(blk_id)

    # ── Single consolidated BL element ───────────────────────────────────────
    bl_element = {
        "SurveyID": survey_id,
        "Element": "BL",
        "PrimaryAttribute": "Survey Blocks",
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {blk["ID"]: blk for blk in all_block_defs},
    }

    # ── Survey Flow ───────────────────────────────────────────────────────────
    fl_element = {
        "SurveyID": survey_id,
        "Element": "FL",
        "PrimaryAttribute": "Survey Flow",
        "SecondaryAttribute": None,
        "TertiaryAttribute": None,
        "Payload": {
            "Flow": [{"Type": "Block", "ID": bid, "FlowID": f"FL_{i+2}"} for i, bid in enumerate(block_ids)],
            "Properties": {"Count": len(block_ids)},
            "FlowID": "FL_1",
            "Type": "Root",
        },
    }

    # ── Response Set ──────────────────────────────────────────────────────────
    rs_element = {
        "SurveyID": survey_id,
        "Element": "RS",
        "PrimaryAttribute": "RS_000000000000000",
        "SecondaryAttribute": "Default Response Set",
        "TertiaryAttribute": None,
        "Payload": None,
    }

    # Order: SO, BL, SQ elements, FL, RS
    all_elements = [so_element, bl_element] + all_sq + [fl_element, rs_element]

    return {
        "SurveyEntry": {
            "SurveyID": survey_id,
            "SurveyName": survey_name,
            "SurveyDescription": "Human evaluation of synthetic dialogues — Step 2.",
            "SurveyOwnerID": "UR_000000000000000",
            "SurveyBrandID": "qualtrics",
            "DivisionID": "DV_000000000000000",
            "SurveyLanguage": "EN",
            "SurveyActiveResponseSet": "RS_000000000000000",
            "SurveyStatus": "Inactive",
            "SurveyStartDate": "0000-00-00 00:00:00",
            "SurveyExpirationDate": "0000-00-00 00:00:00",
            "SurveyCreationDate": "2026-03-15 00:00:00",
            "CreatorID": "UR_000000000000000",
            "LastModified": "2026-03-15 00:00:00",
            "LastAccessed": "0000-00-00 00:00:00",
            "LastActivated": "0000-00-00 00:00:00",
            "Deleted": None,
        },
        "SurveyElements": all_elements,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    for topic, csv_name, survey_name, survey_id, out_name in [
        (
            "career",
            "career_50/career_50_ann_2_to_fill.csv",
            "Dialogue Quality Evaluation - Career (Step 2)",
            "SV_CareerStep2v01",  # 15 alphanum chars → valid pattern
            "DialogueEval_Career_Step2.qsf",
        ),
        (
            "investment",
            "investment_50/investment_50_ann_2_to_fill.csv",
            "Dialogue Quality Evaluation - Investment (Step 2)",
            "SV_InvestStep2v01",  # 15 alphanum chars → valid pattern
            "DialogueEval_Investment_Step2.qsf",
        ),
    ]:
        csv_path = ANN_PKG / csv_name
        df = pd.read_csv(csv_path)
        rows = df.to_dict(orient="records")
        print(f"{topic}: {len(rows)} samples → {out_name}")

        qsf = build_qsf(topic, survey_name, survey_id, rows)

        out_path = OUT_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(qsf, f, indent=2, ensure_ascii=False)
        print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
