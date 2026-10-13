"""中文训练策略：统一五折筛选八候选，训练前五并预测现有 SampleID。

运行环境：Python 3.10+、NumPy。无须原始 .mat 或本机绝对路径。
执行 `python 训练策略_统一五折筛选前五并预测.py train` 重新生成结果；
执行 `python 训练策略_统一五折筛选前五并预测.py predict --ids 49`
只读取已经保存的模型。41、43、49 始终不参与拟合与排名。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "输入特征"
OUTPUT = HERE.parent / "结果表"
MODEL = HERE / "已训练模型_前五方案.json"
EXCLUDED = {41, 43, 49}
SPEED_CLASSES = np.array([550.0, 700.0, 850.0])
BASE = ("PooledRMS", "PeakAbs_Mean", "SpectralCentroid_Hz_Mean",
        "Band8to20k_Fraction_Mean", "Band40to80k_Fraction_Mean")
ONSET = "起始尖峰比_最突出两条均值"
PHASE = ("起始对后续RMS比_五轨中位数", "起始减后续高频比例_五轨中位数")
COMPACT = ("四份摘要_RMS峰值位置", "四份摘要_RMS线性趋势",
           "四份摘要_前半能量占比", "四份摘要_高频后半减前半")
NEW_EARLY = ("前0至10ms对中10至25ms_RMS比_五轨中位数",
             "前0至10ms减中10至25ms_高频比例_五轨中位数")
TAIL = ("尾25ms至终点对中10至25ms_RMS比_五轨中位数",
        "尾25ms至终点减中10至25ms_高频比例_五轨中位数")
SPEED = ("Duration_s_Mean", "Duration_s_SD", "PooledRMS", "RMS_SD",
         "CrestFactor_Mean", "SpectralCentroid_Hz_Mean",
         "Band8to20k_Fraction_Mean", "Band20to40k_Fraction_Mean")

STRATEGIES = {
    "原五项基准": ("整段", BASE),
    "五项加起始尖峰": ("整段", BASE + (ONSET,)),
    "五项加旧时段演化": ("整段", BASE + PHASE),
    "五项加尖峰加旧时段演化": ("整段", BASE + (ONSET,) + PHASE),
    "五项加整段四份摘要": ("整段", BASE + COMPACT),
    "五项加尖峰加0至10与10至25ms两段": ("整段", BASE + (ONSET,) + NEW_EARLY),
    "五项加尖峰加0至10与10至25与尾段三段":
        ("整段", BASE + (ONSET,) + NEW_EARLY + TAIL),
    "核心加工段五项加尖峰加旧时段演化": ("核心", BASE + (ONSET,) + PHASE),
}


def read_csv(name: str) -> list[dict]:
    with (INPUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(name: str, rows: list[dict]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rows_by_id(rows: list[dict], key: str) -> dict[int, dict]:
    result = {int(r[key]): r for r in rows}
    if set(result) != set(range(1, 50)):
        raise ValueError(f"{key} 必须包含 1–49 号各一行")
    return result


def assemble() -> dict[str, list[dict]]:
    base = rows_by_id(read_csv("特征表.csv"), "SampleID")
    onset = rows_by_id(read_csv("每组起始尖峰汇总.csv"), "样本编号")
    phase = rows_by_id(read_csv("每组时段演化.csv"), "样本编号")
    compact = rows_by_id(read_csv("每组整段与摘要特征.csv"), "SampleID")
    windows = rows_by_id(read_csv("每组两三段特征.csv"), "SampleID")
    core_source = [r for r in read_csv("每组整段与核心特征.csv") if r["处理"] == "核心"]
    core = rows_by_id(core_source, "SampleID")
    grouped = {"整段": [], "核心": []}
    for sid in range(1, 50):
        full = dict(base[sid])
        full[ONSET] = onset[sid][ONSET]
        full.update({name: phase[sid][name] for name in PHASE})
        full.update({name: compact[sid][name] for name in COMPACT})
        full.update({name: windows[sid][name] for name in NEW_EARLY + TAIL})
        trimmed = dict(core[sid])
        # Labels are used only when sid belongs to a training or validation fold.
        trimmed["LaserPower_W"] = base[sid]["LaserPower_W"]
        trimmed["ScanSpeed_mm_s"] = base[sid]["ScanSpeed_mm_s"]
        grouped["整段"].append(full)
        grouped["核心"].append(trimmed)
    return grouped


def matrix(rows: list[dict], features: tuple[str, ...]) -> np.ndarray:
    values = np.asarray([[float(r[name]) for name in features] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("特征含空值或非有限数")
    return values


def fit_ridge(x: np.ndarray, y: np.ndarray, ddof: int) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=ddof)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    target = y[:, None] if y.ndim == 1 else y
    target_mean = target.mean(axis=0)
    weights = np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ (target - target_mean))
    return {"center": mean.tolist(), "scale": scale.tolist(),
            "target_center": target_mean.tolist(), "coefficients": weights.tolist()}


def apply_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    return np.asarray(model["target_center"]) + (
        (x - np.asarray(model["center"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficients"])


def fixed_folds() -> dict[int, int]:
    result = {int(r["SampleID"]): int(r["折编号"])
              for r in read_csv("固定五折分配.csv")}
    if set(result) != set(range(1, 50)) - EXCLUDED or set(result.values()) != set(range(1, 6)):
        raise ValueError("固定折分配应只包含 46 个已知样本")
    return result


def evaluate_and_rank(grouped: dict[str, list[dict]], folds: dict[int, int]) -> tuple[list[dict], list[dict]]:
    summaries, predictions = [], []
    for title, (processing, features) in STRATEGIES.items():
        known = [r for r in grouped[processing] if int(r["SampleID"]) not in EXCLUDED]
        for fold in range(1, 6):
            training = [r for r in known if folds[int(r["SampleID"])] != fold]
            held = [r for r in known if folds[int(r["SampleID"])] == fold]
            y = np.array([float(r["LaserPower_W"]) for r in training])
            model = fit_ridge(matrix(training, features), y, ddof=1)
            predicted = apply_ridge(model, matrix(held, features))[:, 0]
            for row, p in zip(held, predicted):
                truth = float(row["LaserPower_W"])
                err = abs(float(p) - truth)
                predictions.append({"方案": title, "处理": processing,
                                    "SampleID": int(row["SampleID"]), "折编号": fold,
                                    "真实功率_W": truth, "预测功率_W": float(p),
                                    "功率绝对误差_W": err, "功率相对误差": err / truth,
                                    "功率20pct内": int(err / truth <= .2)})
        subset = [r for r in predictions if r["方案"] == title]
        if len(subset) != 46:
            raise ValueError("每套方案应有 46 个折外预测")
        summaries.append({"方案": title, "处理": processing, "功率特征数": len(features),
                          "同折验证组数": 46,
                          "功率20pct内组数": sum(r["功率20pct内"] for r in subset),
                          "功率平均相对误差_pct": 100 * float(np.mean([r["功率相对误差"] for r in subset])),
                          "功率MAE_W": float(np.mean([r["功率绝对误差_W"] for r in subset]))})
    summaries.sort(key=lambda r: (-r["功率20pct内组数"],
                                  r["功率平均相对误差_pct"], r["功率MAE_W"], r["方案"]))
    for rank, row in enumerate(summaries, 1):
        row["排名"] = rank
        row["入选前五"] = int(rank <= 5)
    predictions.sort(key=lambda r: (r["方案"], r["SampleID"]))
    return summaries, predictions


def train_final(grouped: dict[str, list[dict]], ranking: list[dict]) -> dict:
    record = {"训练策略": "统一固定五折，按±20%内组数优先选前五，再用全部46组各训练一次",
              "训练SampleID": [i for i in range(1, 50) if i not in EXCLUDED],
              "排除SampleID": sorted(EXCLUDED), "功率模型": {}, "速度模型": {}}
    for item in ranking[:5]:
        name = item["方案"]
        processing, features = STRATEGIES[name]
        rows = [r for r in grouped[processing] if int(r["SampleID"]) not in EXCLUDED]
        power = np.asarray([float(r["LaserPower_W"]) for r in rows])
        record["功率模型"][name] = {"排名": item["排名"], "处理": processing,
                                     "特征": list(features),
                                     "参数": fit_ridge(matrix(rows, features), power, 1)}
        if processing not in record["速度模型"]:
            speed = np.asarray([float(r["ScanSpeed_mm_s"]) for r in rows])
            onehot = (speed[:, None] == SPEED_CLASSES).astype(float)
            record["速度模型"][processing] = {"特征": list(SPEED),
                                            "类别_mm_s": SPEED_CLASSES.tolist(),
                                            "参数": fit_ridge(matrix(rows, SPEED), onehot, 0)}
    return record


def predict_ids(record: dict, grouped: dict[str, list[dict]], ids: list[int]) -> list[dict]:
    result = []
    for name, model in record["功率模型"].items():
        processing = model["处理"]
        rows = [r for r in grouped[processing] if int(r["SampleID"]) in set(ids)]
        if len(rows) != len(set(ids)):
            raise ValueError("请求的 SampleID 不存在")
        power = apply_ridge(model["参数"], matrix(rows, tuple(model["特征"])))[:, 0]
        speed_model = record["速度模型"][processing]
        scores = apply_ridge(speed_model["参数"], matrix(rows, tuple(speed_model["特征"])))
        speed = SPEED_CLASSES[np.argmax(scores, axis=1)]
        for row, p, v in zip(rows, power, speed):
            result.append({"排名": model["排名"], "方案": name, "处理": processing,
                           "SampleID": int(row["SampleID"]),
                           "预测功率_W": float(p), "预测速度_mm_s": float(v)})
    return sorted(result, key=lambda r: (r["排名"], r["SampleID"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "predict"))
    parser.add_argument("--ids", nargs="+", type=int, default=[49])
    args = parser.parse_args()
    grouped = assemble()
    if args.mode == "train":
        ranking, cv_rows = evaluate_and_rank(grouped, fixed_folds())
        save_csv("八候选同折比较与排名.csv", ranking)
        save_csv("八候选固定五折逐组预测.csv", cv_rows)
        record = train_final(grouped, ranking)
        MODEL.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        predictions = predict_ids(record, grouped, [49])
        for row in predictions:
            # 说明 PDF 第 4 页明确标注 Sample 49 为 425 W、850 mm/s；
            # 该值只用于拟合后的对照，不进入训练、排名或标准化。
            row.update({"题目第4页参考功率_W": 425, "题目第4页参考速度_mm_s": 850,
                        "功率绝对误差_W": abs(row["预测功率_W"] - 425),
                        "功率相对误差_pct": 100 * abs(row["预测功率_W"] - 425) / 425,
                        "功率20pct内": int(abs(row["预测功率_W"] - 425) / 425 <= .2),
                        "速度正确": int(row["预测速度_mm_s"] == 850)})
        save_csv("入选五方案预测第49组.csv", predictions)
        print(json.dumps({"前五": ranking[:5], "49号": predictions}, ensure_ascii=False, indent=2))
    else:
        record = json.loads(MODEL.read_text(encoding="utf-8"))
        for row in predict_ids(record, grouped, args.ids):
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

