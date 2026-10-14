"""Fit the selected exploratory P/V heads on 46 labeled groups and predict 41/43/49.

The 425 W slide value for Sample 49 is not read by this program. It is a
provisional reference for a separate, retrospective calculation, not a label.
Run from any directory: python 代码/双头融合_训练与预测未知组.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from 训练策略_统一五折筛选前五并预测 import SPEED, STRATEGIES, assemble


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "结果表" / "未知三组_双头融合候选预测.csv"
UNKNOWN_IDS = (41, 43, 49)
POWER_WEIGHTS = np.array([0.25, 0.25, 0.50])
SPEED_LEVELS = np.array([550.0, 700.0, 850.0])
TWO_SEGMENTS = STRATEGIES["五项加尖峰加0至10与10至25ms两段"][1]
THREE_SEGMENTS = STRATEGIES["五项加尖峰加0至10与10至25与尾段三段"][1]
POWER_COMPONENTS = (
    ("浅层GBDT", TWO_SEGMENTS, False),
    ("三段岭回归", THREE_SEGMENTS, False),
    ("两段加权岭回归", TWO_SEGMENTS, True),
)
SPEED_COMPONENTS = (
    ("八特征岭分类", SPEED, "classifier"),
    ("三段MLP", THREE_SEGMENTS, "mlp"),
    ("三段随机森林", THREE_SEGMENTS, "forest"),
)


def matrix(rows: list[dict], features: tuple[str, ...]) -> np.ndarray:
    result = np.array([[float(row[name]) for name in features] for row in rows])
    if not np.isfinite(result).all():
        raise ValueError("模型输入特征含缺失值或非有限数")
    return result


def regressor(name: str):
    if name in ("三段岭回归", "两段加权岭回归"):
        return Ridge(alpha=1.0)
    if name == "浅层GBDT":
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                         max_depth=2, min_samples_leaf=3,
                                         random_state=5106)
    if name == "三段MLP":
        return MLPRegressor(hidden_layer_sizes=(12,), activation="relu",
                            solver="lbfgs", alpha=1.0, max_iter=800,
                            random_state=5106)
    if name == "三段随机森林":
        return RandomForestRegressor(n_estimators=160, max_depth=3,
                                     min_samples_leaf=3, random_state=5106,
                                     n_jobs=1)
    raise ValueError(name)


def fit_predict_regression(name: str, features: tuple[str, ...],
                           training: list[dict], unknown: list[dict],
                           target: str, weighted: bool = False) -> np.ndarray:
    x_train = matrix(training, features)
    x_unknown = matrix(unknown, features)
    y = np.array([float(row[target]) for row in training])
    scaler = StandardScaler().fit(x_train)
    z_train = scaler.transform(x_train)
    z_unknown = scaler.transform(x_unknown)
    center = float(y.mean())
    scale = float(y.std()) or 1.0
    model = regressor(name)
    if weighted:
        counts = {value: int(np.sum(y == value)) for value in np.unique(y)}
        sample_weight = np.array([1 / np.sqrt(counts[value]) for value in y])
        sample_weight /= sample_weight.mean()
        model.fit(z_train, (y - center) / scale, sample_weight=sample_weight)
    else:
        model.fit(z_train, (y - center) / scale)
    return np.asarray(model.predict(z_unknown)).reshape(-1) * scale + center


def speed_votes(training: list[dict], unknown: list[dict]) -> np.ndarray:
    columns = []
    for name, features, family in SPEED_COMPONENTS:
        if family == "classifier":
            x_train = matrix(training, features)
            scaler = StandardScaler().fit(x_train)
            labels = np.array([float(row["ScanSpeed_mm_s"]) for row in training])
            model = RidgeClassifier(alpha=1.0).fit(scaler.transform(x_train), labels)
            columns.append(model.predict(scaler.transform(matrix(unknown, features))).astype(float))
        else:
            values = fit_predict_regression(name, features, training, unknown,
                                            "ScanSpeed_mm_s")
            nearest = np.argmin(abs(values[:, None] - SPEED_LEVELS[None, :]), axis=1)
            columns.append(SPEED_LEVELS[nearest])
    return np.column_stack(columns)


def vote(row: np.ndarray) -> float:
    counts = np.array([np.sum(row == level) for level in SPEED_LEVELS])
    winners = SPEED_LEVELS[counts == counts.max()]
    return float(winners[0] if len(winners) == 1 else row[0])


def main() -> None:
    rows = {int(row["SampleID"]): row for row in assemble()["整段"]}
    if set(rows) != set(range(1, 50)):
        raise ValueError("需要 1–49 号共 49 组特征")
    training = [rows[sid] for sid in sorted(set(rows) - set(UNKNOWN_IDS))]
    unknown = [rows[sid] for sid in UNKNOWN_IDS]
    if len(training) != 46:
        raise ValueError("训练组应为 46 组")
    for row in unknown:
        if np.isfinite(float(row["LaserPower_W"])):
            raise ValueError("未知组功率列不应有正式标签")
    power_parts = np.column_stack([
        fit_predict_regression(name, features, training, unknown,
                               "LaserPower_W", weighted)
        for name, features, weighted in POWER_COMPONENTS
    ])
    votes = speed_votes(training, unknown)
    predictions = []
    for index, sid in enumerate(UNKNOWN_IDS):
        predictions.append({
            "SampleID": sid,
            "预测功率_W": float(power_parts[index] @ POWER_WEIGHTS),
            "预测速度_mm_s": vote(votes[index]),
            "GBDT功率_W": float(power_parts[index, 0]),
            "三段岭功率_W": float(power_parts[index, 1]),
            "加权岭功率_W": float(power_parts[index, 2]),
            "岭分类速度票_mm_s": float(votes[index, 0]),
            "MLP速度票_mm_s": float(votes[index, 1]),
            "随机森林速度票_mm_s": float(votes[index, 2]),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    for row in predictions:
        print(f"{row['SampleID']}: P={row['预测功率_W']:.2f} W, "
              f"V={row['预测速度_mm_s']:.0f} mm/s")
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
