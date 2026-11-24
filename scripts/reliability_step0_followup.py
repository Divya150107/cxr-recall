"""Step-0 follow-up checks on the saved per-query outputs (local, CPU).

1. Recalibrate every model per label on val (Platt scaling on the logit), report test ECE/Brier.
2. Ensemble controls: is probe+retrieval averaging special, or just ensembling?
3. Re-run the routed gates on calibrated probabilities (routing mixes two models' scales,
   so the uncalibrated probe made the original gates unfair).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = Path(sys.argv[1])
SEEDS = range(5)
ARMS = ["probe", "retrieval", "random_control"]
FEATS = ["top1_sim", "mean_sim", "sim_gap", "nb_consensus", "probe_nb_agree", "probe_entropy"]


def ld(name):
    return np.load(D / f"{name}.npy", allow_pickle=False)


Y = {s: ld(f"{s}_lbl") for s in ("val", "test")}
P = {a: {sd: {s: ld(f"{a}_seed{sd}_{s}") for s in ("val", "test")} for sd in SEEDS} for a in ARMS}
F = {f: {s: ld(f"feat_{f}_{s}") for s in ("val", "test")} for f in FEATS}
L = Y["val"].shape[1]


def macro_auroc(y, p):
    return float(np.mean([roc_auc_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if len(np.unique(y[:, c])) == 2]))


def ece(p, y, bins=10):
    out = []
    for c in range(y.shape[1]):
        idx = np.clip((p[:, c] * bins).astype(int), 0, bins - 1)
        out.append(sum(abs(p[idx == b, c].mean() - y[idx == b, c].mean()) * (idx == b).mean()
                       for b in range(bins) if (idx == b).any()))
    return float(np.mean(out))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt(pv, yv, pt):
    out = np.empty_like(pt)
    for c in range(L):
        if len(np.unique(yv[:, c])) < 2:
            out[:, c] = pt[:, c]
            continue
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv[:, [c]]), yv[:, c])
        out[:, c] = m.predict_proba(logit(pt[:, [c]]))[:, 1]
    return out


# calibrated copies (fit on val, applied to val and test; val copy used only for gate selection)
PC = {a: {sd: {"test": platt(P[a][sd]["val"], Y["val"], P[a][sd]["test"]),
               "val": platt(P[a][sd]["val"], Y["val"], P[a][sd]["val"])} for sd in SEEDS} for a in ARMS}
res = {"calibration_test": {}, "ensembles_test_auroc": {}, "gates_calibrated": {}}

print("=== 1. calibration on test (mean over seeds): raw -> Platt-recalibrated on val")
for a in ARMS:
    raw = [ece(P[a][sd]["test"], Y["test"]) for sd in SEEDS]
    cal = [ece(PC[a][sd]["test"], Y["test"]) for sd in SEEDS]
    braw = [((P[a][sd]["test"] - Y["test"]) ** 2).mean() for sd in SEEDS]
    bcal = [((PC[a][sd]["test"] - Y["test"]) ** 2).mean() for sd in SEEDS]
    auc = [macro_auroc(Y["test"], PC[a][sd]["test"]) for sd in SEEDS]
    res["calibration_test"][a] = {"ece_raw": round(np.mean(raw), 4), "ece_cal": round(np.mean(cal), 4),
                                  "brier_raw": round(np.mean(braw), 4), "brier_cal": round(np.mean(bcal), 4),
                                  "auroc_after_cal": round(np.mean(auc), 4)}
    print(f"{a:15s} ECE {np.mean(raw):.4f} -> {np.mean(cal):.4f}   Brier {np.mean(braw):.4f} -> {np.mean(bcal):.4f}   "
          f"AUROC after cal {np.mean(auc):.4f}")
pc = {a: np.array([res["calibration_test"][a]["ece_cal"]]) for a in ARMS}
d = np.array([ece(PC["retrieval"][sd]["test"], Y["test"]) - ece(PC["probe"][sd]["test"], Y["test"]) for sd in SEEDS])
print(f"calibrated ECE retrieval - probe per seed: {np.round(d, 4)}  mean {d.mean():+.4f}  p={stats.ttest_1samp(d, 0).pvalue:.3f}")
res["calibration_test"]["ece_cal_retrieval_minus_probe"] = {"mean": round(float(d.mean()), 4),
                                                           "p": round(float(stats.ttest_1samp(d, 0).pvalue), 4)}

print("\n=== 2. ensembles, test macro AUROC per seed (calibrated probabilities averaged)")
nxt = lambda sd: (sd + 1) % 5
ens = {
    "probe": lambda sd, s: PC["probe"][sd][s],
    "retrieval": lambda sd, s: PC["retrieval"][sd][s],
    "probe + retrieval": lambda sd, s: (PC["probe"][sd][s] + PC["retrieval"][sd][s]) / 2,
    "probe + random_control": lambda sd, s: (PC["probe"][sd][s] + PC["random_control"][sd][s]) / 2,
    "probe + probe(next seed)": lambda sd, s: (PC["probe"][sd][s] + PC["probe"][nxt(sd)][s]) / 2,
    "retrieval + retrieval(next seed)": lambda sd, s: (PC["retrieval"][sd][s] + PC["retrieval"][nxt(sd)][s]) / 2,
    "retrieval + random_control": lambda sd, s: (PC["retrieval"][sd][s] + PC["random_control"][sd][s]) / 2,
    "probe + retrieval (raw probs)": lambda sd, s: (P["probe"][sd][s] + P["retrieval"][sd][s]) / 2,
}
A = {k: np.array([macro_auroc(Y["test"], f(sd, "test")) for sd in SEEDS]) for k, f in ens.items()}
for k, v in A.items():
    res["ensembles_test_auroc"][k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4)}
    print(f"{k:34s} {v.mean():.4f} +- {v.std(ddof=1):.4f}")


def paired(a, b):
    x = A[a] - A[b]
    return {"mean": round(float(x.mean()), 4), "positive": int((x > 0).sum()),
            "p": round(float(stats.ttest_1samp(x, 0).pvalue), 4)}


for a, b in [("probe + retrieval", "probe + random_control"), ("probe + retrieval", "probe + probe(next seed)"),
             ("probe + random_control", "probe + probe(next seed)"), ("probe + retrieval", "retrieval + retrieval(next seed)")]:
    res["ensembles_test_auroc"][f"{a} vs {b}"] = paired(a, b)
    print(f"  {a} vs {b}: {paired(a, b)}")

print("\n=== 3. routed gates on calibrated probabilities (threshold/direction chosen on val)")
base_val = np.mean([macro_auroc(Y["val"], PC["probe"][sd]["val"]) for sd in SEEDS])
for f in FEATS:
    best = None
    for thr in np.quantile(F[f]["val"], np.linspace(0.1, 0.9, 9)):
        for direc in ("above", "below"):
            def route(sd, s, thr=thr, direc=direc):
                use = F[f][s] > thr if direc == "above" else F[f][s] <= thr
                return np.where(use[:, None], PC["retrieval"][sd][s], PC["probe"][sd][s])
            v = np.mean([macro_auroc(Y["val"], route(sd, "val")) for sd in SEEDS])
            if best is None or v > best[0]:
                best = (v, route, thr, direc)
    t = np.array([macro_auroc(Y["test"], best[1](sd, "test")) for sd in SEEDS])
    dp, dr = t - A["probe"], t - A["retrieval"]
    res["gates_calibrated"][f] = {"test_mean": round(float(t.mean()), 4), "vs_probe": round(float(dp.mean()), 4),
                                  "vs_retrieval": round(float(dr.mean()), 4),
                                  "p_vs_retrieval": round(float(stats.ttest_1samp(dr, 0).pvalue), 4),
                                  "rule": f"retrieval when {best[3]} {best[2]:.4f}"}
    print(f"{f:15s} test {t.mean():.4f}  vs probe {dp.mean():+.4f}  vs retrieval {dr.mean():+.4f} "
          f"(p={stats.ttest_1samp(dr, 0).pvalue:.2f})   rule: retrieval when {best[3]} {best[2]:.4f}")

json.dump(res, open(sys.argv[2], "w"), indent=2, default=float)
print("\nsaved", sys.argv[2])
