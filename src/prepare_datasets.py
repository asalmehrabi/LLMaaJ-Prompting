"""
Download datasets from HuggingFace and save as parquet files.

This script is meant to run once on Snellius (CPU partition) to cache
all datasets locally before running GPU inference jobs.

Usage:
    python -m src.prepare_datasets                     # all datasets
    python -m src.prepare_datasets --dataset flores200  # single dataset
    python -m src.prepare_datasets --list               # show what would run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.config import DATASETS, RAW_DATA, LANG_NAMES


def download_dataset(ds_name: str, ds_cfg: dict) -> int:
    """Download a single dataset, all languages, save as parquet.
    
    Returns number of files saved.
    """
    from datasets import load_dataset
    import pandas as pd

    hf_id = ds_cfg["hf_id"]
    split = ds_cfg["split"]
    sample_n = ds_cfg["sample_n"]
    trust = ds_cfg.get("trust_remote", False)

    out_dir = RAW_DATA / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for lang_code, hf_config in ds_cfg["configs"].items():
        out_path = out_dir / f"{lang_code}.parquet"

        # Skip if already exists
        if out_path.exists():
            df = pd.read_parquet(out_path)
            print(f"  📂 {lang_code}: cached ({len(df)} rows)")
            saved += 1
            continue

        try:
            kwargs = {"split": f"{split}[:{sample_n}]"}
            if hf_config is not None:
                kwargs["name"] = hf_config
            if trust:
                kwargs["trust_remote_code"] = True

            ds = load_dataset(hf_id, **kwargs)
            df = ds.to_pandas()
            df.to_parquet(out_path, index=False)
            print(f"  ✅ {lang_code} ({LANG_NAMES.get(lang_code, lang_code)}): "
                  f"{len(df)} rows saved")
            saved += 1

        except Exception as e:
            print(f"  ❌ {lang_code}: {str(e)[:120]}")

    return saved


def main():
    parser = argparse.ArgumentParser(description="Download thesis datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to download")
    parser.add_argument("--list", action="store_true",
                        help="List datasets without downloading")
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS

    if args.list:
        for name, cfg in targets.items():
            langs = list(cfg["configs"].keys())
            print(f"  {name:20s} | {cfg['task']:15s} | {cfg['hf_id']:35s} | "
                  f"{len(langs)} langs: {', '.join(langs)}")
        return

    print("=" * 70)
    print(f"DOWNLOADING DATASETS → {RAW_DATA}")
    print("=" * 70)

    total = 0
    for name, cfg in targets.items():
        print(f"\n{'─'*60}")
        print(f"📦 {name} ({cfg['task']}) — {cfg['hf_id']}")
        print(f"{'─'*60}")
        total += download_dataset(name, cfg)

    print(f"\n{'='*70}")
    print(f"Done: {total} language files saved")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
