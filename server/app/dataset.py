from datasets import load_dataset


def load_aci():
    return load_dataset("mkieffer/ACI-Bench", "aci")


def _trunc(value, n=200):
    s = str(value).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


if __name__ == "__main__":
    ds = load_aci()
    print(ds)
    split = list(ds)[0]
    print(f"\nsplit: {split} | columns: {ds[split].column_names}")
    for i in range(2):
        print(f"\n=== row {i} ===")
        for k, v in ds[split][i].items():
            print(f"{k}: {_trunc(v)}")
