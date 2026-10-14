"""Broader exploratory power-feature search, holding the speed head fixed.

All feature searches and fusion weights use only the 46 labeled SampleIDs.
Sample 49 is predicted after each candidate is fixed; its provisional 425 W
reference is used to make a retrospective table, never to optimize a model.
The reused 46 groups mean the search scores are not independent validation.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from 训练策略_统一五折筛选前五并预测 import (
    BASE, NEW_EARLY, ONSET, STRATEGIES, TAIL, assemble,
)
from 双头融合_训练与预测未知组 import matrix, regressor


OUT = Path(__file__).resolve().parent.parent / "结果表" / "多候选搜索复现"
REFERENCE_49_W = 425.0  # Retrospective comparison only, never an optimization target.
SEED = 5106
TRAIN_IDS = sorted(set(range(1, 50)) - {41, 43, 49})
FAMILIES = ("浅层GBDT", "三段岭回归", "两段加权岭回归")
NAMES = ("浅层GBDT", "普通岭回归", "加权岭回归")
FEATURES = tuple(BASE) + (ONSET,) + NEW_EARLY + TAIL + (
    "CrestFactor_Mean", "SpectralCentroid_Hz_SD")
BEAM_WIDTH = 4
MAX_COUNT = 10
FINAL_COUNTS = range(3, MAX_COUNT + 1)
WEIGHT_STEPS = 3000
ORIGINAL = (
    tuple(STRATEGIES["五项加尖峰加0至10与10至25ms两段"][1]),
    tuple(STRATEGIES["五项加尖峰加0至10与10至25与尾段三段"][1]),
    tuple(STRATEGIES["五项加尖峰加0至10与10至25ms两段"][1]),
)


def fit_regressor(name, x, y):
    scaler = StandardScaler().fit(x)
    center, scale = float(y.mean()), float(y.std()) or 1.0
    estimator = regressor(name)
    standardized = scaler.transform(x)
    target = (y - center) / scale
    if name == "两段加权岭回归":
        counts = {value: int(np.sum(y == value)) for value in np.unique(y)}
        weights = np.array([1 / np.sqrt(counts[value]) for value in y])
        weights /= weights.mean()
        estimator.fit(standardized, target, sample_weight=weights)
    else:
        estimator.fit(standardized, target)
    return scaler, estimator, center, scale


def predict_regressor(fitted, x):
    scaler, estimator, center, scale = fitted
    return np.asarray(estimator.predict(scaler.transform(x))).reshape(-1) * scale + center


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_simplex(vector):
    sorted_vector = np.sort(vector)[::-1]
    cutoff = np.cumsum(sorted_vector) - 1
    index = np.arange(1, len(vector) + 1)
    active = np.nonzero(sorted_vector - cutoff / index > 0)[0]
    threshold = cutoff[active[-1]] / (active[-1] + 1)
    return np.maximum(vector - threshold, 0)


def fit_weights(oof, actual):
    scaled = oof / actual[:, None]
    lipschitz = 2 * np.linalg.eigvalsh(scaled.T @ scaled).max() / len(actual)
    weights = np.ones(3) / 3
    for _ in range(WEIGHT_STEPS):
        gradient = 2 * scaled.T @ (scaled @ weights - 1) / len(actual)
        updated = project_simplex(weights - gradient / lipschitz)
        if np.max(np.abs(updated - weights)) < 1e-11:
            weights = updated
            break
        weights = updated
    return weights


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {int(r["SampleID"]): r for r in assemble()["整段"]}
    training = [rows[sid] for sid in TRAIN_IDS]
    unknown = [rows[49]]
    if np.isfinite(float(unknown[0]["LaserPower_W"])):
        raise ValueError("49 must not have a formal power label")
    actual = np.array([float(r["LaserPower_W"]) for r in training])
    partitions = list(KFold(n_splits=4, shuffle=True,
                            random_state=SEED + 4000).split(training))

    @lru_cache(maxsize=None)
    def heldout(family_index, columns):
        output = np.full(len(training), np.nan)
        for train_idx, test_idx in partitions:
            train = [training[i] for i in train_idx]
            test = [training[i] for i in test_idx]
            y = np.array([float(r["LaserPower_W"]) for r in train])
            fitted = fit_regressor(FAMILIES[family_index], matrix(train, columns), y)
            output[test_idx] = predict_regressor(fitted, matrix(test, columns))
        if not np.isfinite(output).all():
            raise ValueError("Incomplete held-out predictions")
        return output

    @lru_cache(maxsize=None)
    def fullfit_49(family_index, columns):
        fitted = fit_regressor(FAMILIES[family_index], matrix(training, columns), actual)
        return float(predict_regressor(fitted, matrix(unknown, columns))[0])

    alternatives = []
    for family_index, name in enumerate(NAMES):
        beam = [()]
        best_by_count = {}
        for count in range(1, MAX_COUNT + 1):
            expanded = set()
            for subset in beam:
                for feature in FEATURES:
                    if feature not in subset:
                        expanded.add(tuple(sorted(subset + (feature,))))
            ranked = sorted(expanded, key=lambda columns: (
                float(np.mean(((heldout(family_index, columns) - actual) / actual) ** 2)),
                columns))
            beam = ranked[:BEAM_WIDTH]
            best_by_count[count] = beam[0]
            print(f"{name}: {count} features, {len(expanded)} extensions", flush=True)
        candidates = [(f"搜索_{count}项", best_by_count[count])
                      for count in FINAL_COUNTS]
        candidates.append(("原主模型特征", ORIGINAL[family_index]))
        local = {
            0: ("PooledRMS", "Band40to80k_Fraction_Mean", "CrestFactor_Mean",
                "PeakAbs_Mean", "尾25ms至终点对中10至25ms_RMS比_五轨中位数"),
            1: ("前0至10ms对中10至25ms_RMS比_五轨中位数",
                "尾25ms至终点对中10至25ms_RMS比_五轨中位数",
                "SpectralCentroid_Hz_SD", "PooledRMS"),
            2: ("CrestFactor_Mean",
                "前0至10ms减中10至25ms_高频比例_五轨中位数",
                "PooledRMS", "尾25ms至终点对中10至25ms_RMS比_五轨中位数",
                "前0至10ms对中10至25ms_RMS比_五轨中位数",
                "Band8to20k_Fraction_Mean", "SpectralCentroid_Hz_Mean",
                "SpectralCentroid_Hz_SD"),
        }
        candidates.append(("上一轮局部搜索", local[family_index]))
        deduplicated = {}
        for label, columns in candidates:
            key = tuple(sorted(columns))
            # Preserve the original column order: tree split tie-breaking can
            # depend on it, so sorting the baseline changed its prediction.
            deduplicated.setdefault(key, (label, tuple(columns)))
        family_rows = []
        for label, columns in deduplicated.values():
            prediction = heldout(family_index, columns)
            relative = abs(prediction - actual) / actual
            family_rows.append({"模型": name, "候选": label, "特征数": len(columns),
                                "特征": "; ".join(columns),
                                "46组内部留出_相对平方误差": float(np.mean(((prediction - actual) / actual) ** 2)),
                                "46组内部留出_平均相对误差_pct": 100 * float(relative.mean()),
                                "46组内部留出_20pct内": int(np.sum(relative <= 0.2)),
                                "49预测功率_W": fullfit_49(family_index, columns)})
        family_rows.sort(key=lambda row: (row["46组内部留出_相对平方误差"], row["特征数"]))
        alternatives.append([(row, tuple(row["特征"].split("; ")))
                             for row in family_rows])
        write_csv(OUT / f"{name}_候选特征比较.csv", family_rows)

    combinations = []
    for a in alternatives[0]:
        for b in alternatives[1]:
            for c in alternatives[2]:
                chosen = (a, b, c)
                oof = np.column_stack([
                    heldout(i, chosen[i][1]) for i in range(3)])
                weights = fit_weights(oof, actual)
                internal = oof @ weights
                rel = abs(internal - actual) / actual
                sample49 = sum(weights[i] * chosen[i][0]["49预测功率_W"]
                               for i in range(3))
                combinations.append({
                    "GBDT候选": a[0]["候选"], "GBDT特征数": a[0]["特征数"],
                    "岭回归候选": b[0]["候选"], "岭回归特征数": b[0]["特征数"],
                    "加权岭候选": c[0]["候选"], "加权岭特征数": c[0]["特征数"],
                    "GBDT权重": float(weights[0]), "岭回归权重": float(weights[1]),
                    "加权岭权重": float(weights[2]),
                    "46组内部留出_相对平方误差": float(np.mean(((internal - actual) / actual) ** 2)),
                    "46组内部留出_平均相对误差_pct": 100 * float(rel.mean()),
                    "46组内部留出_20pct内": int(np.sum(rel <= 0.2)),
                    "49预测功率_W": float(sample49),
                    "49速度_沿用原头_mm_s": 850,
                    "49相对425W事后误差_pct": 100 * abs(sample49 - REFERENCE_49_W) / REFERENCE_49_W,
                })
    # Keep the actual pre-existing main model visible. Re-optimizing weights
    # changes the experiment, so it must not silently replace this baseline.
    for label, feature_label, preset in (
        ("原主模型固定权重", "原主模型特征", np.array([0.25, 0.25, 0.50])),
        ("49已知后调权对照_非盲测", "原主模型特征", np.array([0.25, 0.0, 0.75])),
        ("上轮局部搜索固定权重", "上一轮局部搜索",
         np.array([0.26205218079024717, 0.20271763109333793,
                   0.5352301881164148])),
    ):
        chosen = [next(item for item in family if item[0]["候选"] == feature_label)
                  for family in alternatives]
        oof = np.column_stack([heldout(i, chosen[i][1]) for i in range(3)])
        internal = oof @ preset
        rel = abs(internal - actual) / actual
        sample49 = sum(preset[i] * chosen[i][0]["49预测功率_W"] for i in range(3))
        combinations.append({
            "GBDT候选": label, "GBDT特征数": chosen[0][0]["特征数"],
            "岭回归候选": label, "岭回归特征数": chosen[1][0]["特征数"],
            "加权岭候选": label, "加权岭特征数": chosen[2][0]["特征数"],
            "GBDT权重": float(preset[0]), "岭回归权重": float(preset[1]),
            "加权岭权重": float(preset[2]),
            "46组内部留出_相对平方误差": float(np.mean(((internal - actual) / actual) ** 2)),
            "46组内部留出_平均相对误差_pct": 100 * float(rel.mean()),
            "46组内部留出_20pct内": int(np.sum(rel <= 0.2)),
            "49预测功率_W": float(sample49),
            "49速度_沿用原头_mm_s": 850,
            "49相对425W事后误差_pct": 100 * abs(sample49 - REFERENCE_49_W) / REFERENCE_49_W,
        })
    combinations.sort(key=lambda row: (row["46组内部留出_相对平方误差"],
                                       row["46组内部留出_平均相对误差_pct"]))
    for rank, row in enumerate(combinations, 1):
        row["46组内搜索排名"] = rank
    write_csv(OUT / "全部融合组合比较.csv", combinations)
    write_csv(OUT / "46组内排名前二十.csv", combinations[:20])
    retrospective = sorted(combinations, key=lambda row: row["49相对425W事后误差_pct"])
    write_csv(OUT / "49事后误差前二十_不可当盲测.csv", retrospective[:20])
    summary = {
        "候选组合数": len(combinations), "其中梯度调权组合数": len(combinations) - 3,
        "每模型候选数": [len(x) for x in alternatives],
        "选择范围": "12项池，每模型3–10项各保留宽度4束搜索最佳集；另含原主模型和上轮局部搜索特征；全部笛卡尔积融合",
        "训练数据": "只用46个有正式功率标签的SampleID，四份轮流留出用于特征和权重探索",
        "权重": "每组融合候选用投影梯度下降优化46组内部留出的相对平方误差，非负且和为1",
        "速度": "原速度头保持不变，49号仍预测850 mm/s",
        "限制": "特征和权重在同46组上搜索，内部留出误差非独立验证；49号425W仅作事后对照，若按其排名挑选即是事后调参",
        "46组内最优": combinations[0], "49事后最优": retrospective[0],
    }
    (OUT / "实验设置与摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
