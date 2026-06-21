"""
샘플 데이터 준비 스크립트.
GitHub에서 SMD, PSM 원본 데이터를 받아 앱 형식으로 가공합니다.

실행 방법: python scripts/make_sample_data.py
"""
import io, urllib.request
import numpy as np
import pandas as pd
import os

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "sample_data")
os.makedirs(OUTPUT, exist_ok=True)

def prepare_smd():
    base = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    test_raw  = urllib.request.urlopen(f"{base}/test/machine-1-1.txt", timeout=30).read().decode()
    label_raw = urllib.request.urlopen(f"{base}/test_label/machine-1-1.txt", timeout=30).read().decode()
    test  = pd.read_csv(io.StringIO(test_raw),  header=None)
    label = pd.read_csv(io.StringIO(label_raw), header=None, names=["is_anomaly"])
    test.columns = [f"server_metric_{i:02d}" for i in range(test.shape[1])]
    timestamps = pd.date_range("2019-01-01", periods=len(test), freq="min")
    test.insert(0, "timestamp", timestamps)
    test["is_anomaly"] = label["is_anomaly"].values
    path = os.path.join(OUTPUT, "smd_server_machine.csv")
    test.to_csv(path, index=False)
    print(f"[SMD] {len(test):,}행 저장 → {path}")

def prepare_psm():
    base = "https://raw.githubusercontent.com/eBay/RANSynCoders/main/data"
    test_raw  = urllib.request.urlopen(f"{base}/test.csv",       timeout=30).read().decode()
    label_raw = urllib.request.urlopen(f"{base}/test_label.csv", timeout=30).read().decode()
    test  = pd.read_csv(io.StringIO(test_raw))
    label = pd.read_csv(io.StringIO(label_raw))
    base_ts = pd.Timestamp("2020-01-01")
    test["timestamp"] = base_ts + pd.to_timedelta(test["timestamp_(min)"], unit="min")
    test = test.drop(columns=["timestamp_(min)"])
    test["is_anomaly"] = label["label"].values
    subset = test.head(5000)
    cols = ["timestamp"] + [c for c in subset.columns if c not in ("timestamp","is_anomaly")] + ["is_anomaly"]
    subset = subset[cols]
    path = os.path.join(OUTPUT, "psm_server_metrics.csv")
    subset.to_csv(path, index=False)
    print(f"[PSM] {len(subset):,}행 저장 → {path}")

if __name__ == "__main__":
    prepare_smd()
    prepare_psm()
    print("샘플 데이터 준비 완료")
