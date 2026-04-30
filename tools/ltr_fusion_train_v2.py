import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


DROP_COLS = {"query_id", "candidate_id", "label"}


def group_sizes(df: pd.DataFrame) -> List[int]:
    return df.groupby("query_id", sort=True).size().tolist()


def evaluate_rankings(df_scored: pd.DataFrame) -> Dict[str, float]:
    hit1 = hit4 = hit8 = hit16 = 0
    mrr = 0.0
    total_q = 0

    for qid, g in df_scored.groupby("query_id", sort=True):
        total_q += 1
        g = g.sort_values(["pred", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
        truth_rows = g.index[g["label"] == 1].tolist()
        if not truth_rows:
            continue
        rank = truth_rows[0] + 1

        if rank == 1:
            hit1 += 1
        if rank <= 4:
            hit4 += 1
        if rank <= 8:
            hit8 += 1
        if rank <= 16:
            hit16 += 1
        mrr += 1.0 / rank

    if total_q == 0:
        return {"hit@1": 0.0, "hit@4": 0.0, "hit@8": 0.0, "hit@16": 0.0, "MRR": 0.0}

    return {
        "hit@1": float(hit1 / total_q),
        "hit@4": float(hit4 / total_q),
        "hit@8": float(hit8 / total_q),
        "hit@16": float(hit16 / total_q),
        "MRR": float(mrr / total_q),
    }


def split_queries(df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    qids = sorted(df["query_id"].unique())
    rng = np.random.default_rng(seed)
    qids = np.array(qids)
    rng.shuffle(qids)

    n_test = max(1, int(round(len(qids) * test_frac)))
    test_q = set(qids[:n_test].tolist())
    train_q = set(qids[n_test:].tolist())

    train_df = df[df["query_id"].isin(train_q)].copy()
    test_df = df[df["query_id"].isin(test_q)].copy()
    return train_df, test_df


def to_py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def dict_to_py(d: Dict) -> Dict:
    return {str(k): to_py(v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--min-data-in-leaf", type=int, default=10)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--out-json", default="/home/glyph/GLYPH_CPP_BACKEND/out/ltr_train_v2_metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    features = [c for c in df.columns if c not in DROP_COLS]

    train_df, test_df = split_queries(df, args.test_frac, args.seed)

    X_train = train_df[features]
    y_train = train_df["label"].astype(int)
    group_train = group_sizes(train_df)

    X_test = test_df[features]
    y_test = test_df["label"].astype(int)
    group_test = group_sizes(test_df)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        random_state=args.seed,
        n_jobs=-1,
        force_col_wise=True,
    )

    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        eval_at=[1, 4, 8, 16],
    )

    train_scored = train_df.copy()
    train_scored["pred"] = model.predict(X_train)

    test_scored = test_df.copy()
    test_scored["pred"] = model.predict(X_test)

    train_metrics = evaluate_rankings(train_scored)
    test_metrics = evaluate_rankings(test_scored)

    raw_importance = list(model.feature_importances_)
    feature_importance = {
        features[i]: int(raw_importance[i])
        for i in range(len(features))
    }
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: -x[1]))

    used_features = [name for name, imp in feature_importance.items() if imp > 0]
    unused_features = [name for name, imp in feature_importance.items() if imp == 0]

    out = {
        "train_queries": int(train_df["query_id"].nunique()),
        "test_queries": int(test_df["query_id"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "features": list(features),
        "used_features": list(used_features),
        "unused_features": list(unused_features),
        "train_metrics": dict_to_py(train_metrics),
        "test_metrics": dict_to_py(test_metrics),
        "feature_importance": dict_to_py(feature_importance),
    }

    print("=" * 60)
    print(" LTR FUSION TRAIN V2")
    print("=" * 60)
    print(f"train_queries={out['train_queries']}")
    print(f"test_queries={out['test_queries']}")
    print(f"train_rows={out['train_rows']}")
    print(f"test_rows={out['test_rows']}")

    print("\nTRAIN METRICS:")
    for k, v in train_metrics.items():
        print(f"  {k} = {v:.4f}")

    print("\nTEST METRICS:")
    for k, v in test_metrics.items():
        print(f"  {k} = {v:.4f}")

    print("\nUSED FEATURES:")
    for name in used_features:
        print(f"  {name}")

    print("\nTOP FEATURE IMPORTANCE:")
    for k, v in list(feature_importance.items())[:20]:
        print(f"  {k} = {v}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nsaved: {args.out_json}")


if __name__ == "__main__":
    main()