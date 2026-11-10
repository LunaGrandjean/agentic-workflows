"""Download immutable GEM/E2E Parquet splits, audit them and optionally upload to GCS."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_workflows.restaurant_data import write_restaurant_splits


def main():
    import pandas as pd
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/restaurant")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Reuse downloaded, checksum-verified Parquet files.")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    args = parser.parse_args()
    raw = args.output_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_file = raw / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    elif args.offline:
        parser.error("No downloaded manifest. Run once without --offline.")
    else:
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        def get_json(url):
            response = session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        source = get_json("https://huggingface.co/api/datasets/GEM/e2e_nlg")
        converted = get_json("https://huggingface.co/api/datasets/GEM/e2e_nlg/revision/refs%2Fconvert%2Fparquet")
        manifest = {"dataset": "GEM/e2e_nlg", "source_revision": source["sha"],
                    "parquet_revision": converted["sha"], "license": "CC-BY-SA-4.0", "files": {}}
        for split in ("train", "validation", "test"):
            url = f"https://huggingface.co/datasets/GEM/e2e_nlg/resolve/{converted['sha']}/default/{split}/0000.parquet"
            response = session.get(url, timeout=120)
            response.raise_for_status()
            path = raw / f"{split}.parquet"
            path.write_bytes(response.content)
            manifest["files"][split] = {"url": url, "sha256": hashlib.sha256(response.content).hexdigest(), "rows": len(pd.read_parquet(path))}
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths = {}
    for split, metadata in manifest["files"].items():
        path = raw / f"{split}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        paths[split] = str(path)
    audit = write_restaurant_splits(paths, str(args.output_dir / "prepared"),
        {split: getattr(args, f"{split}_limit") for split in ("train", "validation", "test")})
    if args.upload:
        from google.cloud import storage
        from agentic_workflows.constants import GCP_BUCKET_NAME, GCP_PROJECT_ID
        prefix = f"restaurant/e2e_nlg/{manifest['parquet_revision']}"
        bucket = storage.Client(project=GCP_PROJECT_ID).bucket(GCP_BUCKET_NAME)
        for path in [*raw.glob("*.parquet"), manifest_file]:
            blob = bucket.blob(f"{prefix}/{path.name}")
            if blob.exists():
                if hashlib.sha256(blob.download_as_bytes(timeout=120)).hexdigest() != hashlib.sha256(path.read_bytes()).hexdigest():
                    raise ValueError(f"Existing frozen object differs from the local file: {blob.name}")
            else:
                blob.upload_from_filename(str(path), timeout=120, if_generation_match=0)
        uri = f"gs://{GCP_BUCKET_NAME}/{prefix}/manifest.json"
        print(f"Dataset manifest: {uri}", flush=True)
        (args.output_dir / "dataset_uri.txt").write_text(uri, encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
