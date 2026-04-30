#!/usr/bin/env bash
set -euo pipefail

HF_OWNER="neashton"
HF_PREFIX="drivaerml"
BASE_URL="https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main"
LOCAL_DIR="data/raw/drivaerml"
MODE="aggregate"
RUN_START=1
RUN_END=20
WITH_STL=0

usage() {
  cat <<'USAGE'
Download a small DrivAerML subset.

Usage:
  bash scripts/download_drivaerml.sh [--aggregate] [--per-run] [--run-start N] [--run-end N] [--with-stl] [--out DIR]

Default:
  --aggregate downloads root CSV files only. This is enough for tabular XGBoost training.

Options:
  --aggregate      Download aggregate root CSVs.
  --per-run        Download per-run CSVs for a range of run IDs.
  --run-start N    First run ID for --per-run. Default: 1.
  --run-end N      Last run ID for --per-run. Default: 20.
  --with-stl       Include drivaer_i.stl in --per-run downloads. STLs are large.
  --out DIR        Output directory. Default: data/raw/drivaerml.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aggregate)
      MODE="aggregate"
      shift
      ;;
    --per-run)
      MODE="per-run"
      shift
      ;;
    --run-start)
      RUN_START="$2"
      shift 2
      ;;
    --run-end)
      RUN_END="$2"
      shift 2
      ;;
    --with-stl)
      WITH_STL=1
      shift
      ;;
    --out)
      LOCAL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$LOCAL_DIR"
MANIFEST="$LOCAL_DIR/download_manifest.jsonl"
: > "$MANIFEST"

download_file() {
  local url="$1"
  local out="$2"
  local required="${3:-required}"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then
    echo "exists: $out"
  else
    echo "download: $url"
    if ! curl -L --fail --retry 3 --retry-delay 2 "$url" -o "$out"; then
      rm -f "$out"
      if [[ "$required" == "optional" ]]; then
        echo "optional file unavailable, skipping: $url" >&2
        return 0
      fi
      echo "required file unavailable: $url" >&2
      return 1
    fi
  fi

  local size sha
  size=$(wc -c < "$out" | tr -d ' ')
  sha=$(shasum -a 256 "$out" | awk '{print $1}')
  printf '{"url":"%s","path":"%s","size_bytes":%s,"sha256":"%s"}\n' \
    "$url" "$out" "$size" "$sha" >> "$MANIFEST"
}

if [[ "$MODE" == "aggregate" ]]; then
  for file in geo_parameters_all.csv force_mom_all.csv force_mom_constref_all.csv; do
    download_file "$BASE_URL/$file" "$LOCAL_DIR/$file" required
  done
  for file in geo_ref_all.csv LICENSE.txt README.md; do
    download_file "$BASE_URL/$file" "$LOCAL_DIR/$file" optional
  done
else
  for i in $(seq "$RUN_START" "$RUN_END"); do
    run_dir="$LOCAL_DIR/run_$i"
    for file in "geo_parameters_${i}.csv" "geo_ref_${i}.csv" "force_mom_${i}.csv" "force_mom_constref_${i}.csv"; do
      download_file "$BASE_URL/run_$i/$file" "$run_dir/$file"
    done
    if [[ "$WITH_STL" == "1" ]]; then
      download_file "$BASE_URL/run_$i/drivaer_${i}.stl" "$run_dir/drivaer_${i}.stl"
    fi
  done
fi

echo "Manifest written to $MANIFEST"
