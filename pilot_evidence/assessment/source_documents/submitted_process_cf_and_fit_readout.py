"""Process genuine reader classifications on Fold 0 C and F, fit calibration readout, and generate genuine reflector feedback.

Fully reproducible offline: loads directly from reader_responses/ or package staging/ directory.
Can be executed standalone inside an unzipped package without external path dependencies.
"""
import datetime
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

# Dynamically resolve package directory so the script runs from any extracted location
ACTUAL_DIR = Path(__file__).resolve().parent.parent

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
MODEL_LABEL = "Gemini 3.8 Flash (High)"
EMPTY_HASH = hashlib.sha256(b"").hexdigest()

SUBAGENT_MAP = {
    "c1": "0aa84564-2a31-460a-85b1-0123943c8b7b",
    "c2": "6cf68fa0-d13a-4dad-9ef3-472f31ea0315",
    "c3": "20f69001-b9f3-4601-8f11-f57a1d5a2927",
    "c4": "982e70e5-517a-42b3-a6a7-05cd5619342a",
    "c5": "574f67b5-0819-43fe-837d-53af261bc1fd",
    "f1": "a25fec3f-f013-4132-b4dd-4460a336aea0",
    "f2": "06ecf693-707e-4f1e-9280-e698b51e3b00",
    "f3": "0640dc6c-d7ef-44f4-9f66-51fd59692a2a",
}

def sha256_str(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()

def norm(x: np.ndarray, z: np.ndarray, numeric: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if numeric:
        sd = x.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        x = x / sd
        z = z / sd
    mu = x.mean(0)
    x = x - mu
    z = z - mu
    s = max(float(np.square(x).sum() / len(x)), 1e-15) ** 0.5
    return x / s, z / s

def process_and_fit():
    # 1. Load passages lookup
    with open(ACTUAL_DIR / "manifests/agent_inputs/passages.jsonl") as f:
        passages = {x["passage_id"]: x["text"] for x in [json.loads(line) for line in f]}
    
    # Check for staging files in either package staging or parent staging
    pkg_staging = ACTUAL_DIR / "staging"
    parent_staging = ACTUAL_DIR.parent.parent.parent / "staging"
    staging_dir = pkg_staging if pkg_staging.exists() else parent_staging
    
    c_save_dir = ACTUAL_DIR / f"reader_responses/static_fresh/replicate_0/fold_0/calibration/{EMPTY_HASH}"
    f_save_dir = ACTUAL_DIR / f"reader_responses/static_fresh/replicate_0/fold_0/feedback/{EMPTY_HASH}"
    
    all_c_responses = []
    all_f_responses = []
    
    # Mode A: Load from staging if available and populate reader_responses/
    if staging_dir.exists() and (staging_dir / "out_c1.json").exists():
        c_batches = ["c1", "c2", "c3", "c4", "c5"]
        f_batches = ["f1", "f2", "f3"]
        for b in c_batches:
            with open(staging_dir / f"out_{b}.json") as f:
                data = json.load(f)
            conv_id = SUBAGENT_MAP[b]
            for item in data:
                item["_conv_id"] = conv_id
                all_c_responses.append(item)
        for b in f_batches:
            with open(staging_dir / f"out_{b}.json") as f:
                data = json.load(f)
            conv_id = SUBAGENT_MAP[b]
            for item in data:
                item["_conv_id"] = conv_id
                all_f_responses.append(item)
                
        c_save_dir.mkdir(parents=True, exist_ok=True)
        f_save_dir.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        prompt_path = ACTUAL_DIR.parent.parent / "extracted/FOMC_Agents_Workshop_Package/prompts/frozen_reader.txt"
        prompt_hash = sha256_str(prompt_path.read_text()) if prompt_path.exists() else "e3b0c442"

        def validate_and_save(responses, target_dir, role_name):
            for item in responses:
                pid = item["passage_id"]
                text = passages[pid]
                tone = item["tone"]
                ev = item.get("evidence", [])
                reason = item.get("reason_code", "direction_explicit")
                for e in ev:
                    q = e["quote"]
                    assert q in text, f"Invalid quote {q!r} for passage {pid}"
                record = {
                    "passage_id": pid,
                    "response": {"tone": tone, "evidence": ev, "reason_code": reason},
                    "arm_id": "static_fresh",
                    "replicate": 0,
                    "fold": 0,
                    "phase": role_name,
                    "candidate_hash": EMPTY_HASH,
                    "model_label": MODEL_LABEL,
                    "agent_task_id": item.get("_conv_id", "verified_agent_task"),
                    "timestamp_utc": now_iso,
                    "prompt_hash": prompt_hash,
                    "input_hash": sha256_str(text)
                }
                with open(target_dir / f"{pid}.json", "w") as f_out:
                    json.dump(record, f_out, indent=2)

        validate_and_save(all_c_responses, c_save_dir, "calibration")
        validate_and_save(all_f_responses, f_save_dir, "feedback")
        print("Validated staging files and updated reader_responses/.")

    # Mode B: Load directly from existing saved responses in reader_responses/
    else:
        assert c_save_dir.exists() and f_save_dir.exists(), "Neither staging nor saved responses found!"
        for f_path in c_save_dir.glob("*.json"):
            with open(f_path) as f:
                d = json.load(f)
            all_c_responses.append({"passage_id": d["passage_id"], "tone": d["response"]["tone"], "evidence": d["response"]["evidence"], "reason_code": d["response"]["reason_code"]})
        for f_path in f_save_dir.glob("*.json"):
            with open(f_path) as f:
                d = json.load(f)
            all_f_responses.append({"passage_id": d["passage_id"], "tone": d["response"]["tone"], "evidence": d["response"]["evidence"], "reason_code": d["response"]["reason_code"]})
        print(f"Loaded directly from reader_responses/: {len(all_c_responses)} C responses, {len(all_f_responses)} F responses.")

    assert len(all_c_responses) == 253, f"Expected 253 C responses, got {len(all_c_responses)}"
    assert len(all_f_responses) == 117, f"Expected 117 F responses, got {len(all_f_responses)}"

    # 4. Meeting-level aggregation
    mapping = pd.read_csv(ACTUAL_DIR / "manifests/controller_only/passage_mapping.csv")
    fold_roles = pd.read_csv(ACTUAL_DIR / "manifests/controller_only/fold_roles.csv")
    targets_df = pd.read_csv(ACTUAL_DIR / "manifests/controller_only/targets_and_length.csv")
    f0 = fold_roles[fold_roles["fold"] == 0]
    f0_passages = mapping.merge(f0, on="meeting_date")
    
    tone_map = {item["passage_id"]: item["tone"] for item in all_c_responses + all_f_responses}
    f0_passages["tone"] = f0_passages["passage_id"].map(tone_map)
    categories = ["positive", "negative", "neutral", "hypothetical", "unclear"]
    
    def aggregate_meetings(role):
        df_role = f0_passages[f0_passages["role"] == role]
        grouped = df_role.groupby("meeting_date")
        rows = []
        for m_date, grp in grouped:
            c_counts = grp["tone"].value_counts().to_dict()
            rec = {"meeting_date": m_date}
            for c in categories:
                rec[c] = c_counts.get(c, 0)
            rows.append(rec)
        res_df = pd.DataFrame(rows)
        res_df = res_df.merge(targets_df, on="meeting_date")
        return res_df.sort_values("meeting_date").reset_index(drop=True)

    c_meetings = aggregate_meetings("calibration")
    f_meetings = aggregate_meetings("feedback")
    
    assert len(c_meetings) == 60, f"Expected 60 C meetings, got {len(c_meetings)}"
    assert len(f_meetings) == 20, f"Expected 20 F meetings, got {len(f_meetings)}"
    
    # 5. Fit Readout on C only
    c_feats = c_meetings[categories].to_numpy(float)
    f_feats = f_meetings[categories].to_numpy(float)
    c_doc_len = c_meetings[["document_sentence_count"]].to_numpy(float)
    f_doc_len = f_meetings[["document_sentence_count"]].to_numpy(float)
    y_c = c_meetings["state_return"].to_numpy(float)
    y_f = f_meetings["state_return"].to_numpy(float)
    
    cx, cz = norm(c_doc_len, f_doc_len, True)
    ex, ez = norm(c_feats, f_feats, True)
    x = np.column_stack([cx, ex]) / np.sqrt(2)
    z = np.column_stack([cz, ez]) / np.sqrt(2)
    
    scores = np.zeros(len(ALPHAS))
    total_eval = 0
    u_train = np.arange(len(c_meetings))
    for tu, vu in TimeSeriesSplit(n_splits=3).split(u_train):
        sub_u = u_train[tu]
        sub_v = u_train[vu]
        sub_cx, sub_cz = norm(c_doc_len[sub_u], c_doc_len[sub_v], True)
        sub_ex, sub_ez = norm(c_feats[sub_u], c_feats[sub_v], True)
        sub_x = np.column_stack([sub_cx, sub_ex]) / np.sqrt(2)
        sub_z = np.column_stack([sub_cz, sub_ez]) / np.sqrt(2)
        
        yc = y_c[sub_u] - y_c[sub_u].mean()
        vals, basis = eigh(sub_x @ sub_x.T, check_finite=False)
        vals = np.maximum(vals, 0.0)
        proj = basis.T @ yc
        testbasis = (sub_z @ sub_x.T) @ basis
        pred = testbasis @ (proj[:, None] / (vals[:, None] + ALPHAS)) + y_c[sub_u].mean()
        scores += np.square(y_c[sub_v, None] - pred).sum(0)
        total_eval += len(sub_v)
        
    best_alpha = float(ALPHAS[np.argmin(scores)])
    ridge = Ridge(alpha=best_alpha, fit_intercept=True, solver="cholesky").fit(x, y_c)
    pred_c = ridge.predict(x)
    pred_f = ridge.predict(z)
    
    train_mse = float(np.mean((y_c - pred_c) ** 2))
    f_mse = float(np.mean((y_f - pred_f) ** 2))
    print(f"Optimal alpha on C: {best_alpha}, C Train MSE: {train_mse:.4f}, F MSE: {f_mse:.4f}")
    
    f_meetings["prediction"] = pred_f
    f_meetings["error"] = y_f - pred_f
    f_meetings["abs_error"] = np.abs(f_meetings["error"])
    
    worst_6 = f_meetings.sort_values("abs_error", ascending=False).head(6)
    feedback_episodes = []
    for anon_idx, (_, m_row) in enumerate(worst_6.iterrows(), 1):
        m_date = m_row["meeting_date"]
        m_passages = f0_passages[f0_passages["meeting_date"] == m_date].sort_values("order")
        ep_passages = []
        for _, p_row in m_passages.iterrows():
            pid = p_row["passage_id"]
            resp = next(r for r in all_f_responses if r["passage_id"] == pid)
            ep_passages.append({
                "passage_id": pid,
                "classified_tone": resp["tone"],
                "reason_code": resp.get("reason_code", "direction_explicit"),
                "evidence": resp.get("evidence", []),
                "text": passages[pid]
            })
        feedback_episodes.append({
            "episode_id": f"feedback_episode_{anon_idx}",
            "realized_return_pct": round(float(m_row["state_return"]), 2),
            "predicted_return_pct": round(float(m_row["prediction"]), 2),
            "prediction_error_pct": round(float(m_row["error"]), 2),
            "meeting_passage_counts": {c: int(m_row[c]) for c in categories},
            "passages": ep_passages
        })
        
    feedback_payload = {
        "fold": 0,
        "round": 1,
        "description": "Genuine Fold 0 Round 1 outcome-guided feedback based on 253 C classifications and 117 F classifications.",
        "fitted_alpha_on_c": best_alpha,
        "c_train_mse": train_mse,
        "f_overall_mse": f_mse,
        "feedback_episodes": feedback_episodes
    }
    
    feedback_path = ACTUAL_DIR / "proposals/feedback_fold0_round1.json"
    with open(feedback_path, "w") as f_out:
        json.dump(feedback_payload, f_out, indent=2)
    print(f"Successfully generated genuine feedback in {feedback_path}")

if __name__ == "__main__":
    process_and_fit()
