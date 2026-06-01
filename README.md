# SLC-PFM Feature Extraction

This repo contains a streaming extractor for zipped WebP tile archives with this layout:

```text
/project/rrg-jma/shared/SLC-PFM/
  chunk_8/
    chunk8_id000_....zip
      1.webp
      2.webp
      ...
```

The extractor writes one `.pt` file per input zip and encoder:

```text
<output_root>/
  virchow2/chunk_68/chunk68_id441_63347379.pt
  hoptimus1/chunk_68/chunk68_id441_63347379.pt
  uni_v2/chunk_68/chunk68_id441_63347379.pt
```

Each `.pt` file is a `torch.save` dictionary with:

- `features`: per-tile CLS embeddings, shape `[num_tiles, dim]`
- `tile_names`: tile names inside the source zip, aligned with `features`
- `source_zip`, `encoder`, `feature_kind`, `errors`
- `token_maps` and `token_grid_size` when `--include-token-maps` is used

## Setup

Use a Python environment with PyTorch, then install the lightweight dependencies:

```bash
pip install -r requirements.txt
```

The three requested models are gated Hugging Face repositories. Before the first run, request/accept access for:

- `paige-ai/Virchow2`
- `bioptimus/H-optimus-1`
- `MahmoodLab/UNI2-h`

Then authenticate on the machine that will run extraction:

```bash
huggingface-cli login
```

On Fir, keep model caches out of home storage. Use `--hf-cache-dir` or set `HF_HOME` to project/turbo storage.

## Smoke Test

Run a tiny extraction before launching the full job:

```bash
python scripts/extract_zip_tile_features.py \
  --input-root /project/rrg-jma/shared/SLC-PFM \
  --output-root /project/rrg-jma/shared/SLC-PFM_features \
  --encoders virchow2 \
  --chunks chunk_8 \
  --limit-zips 1 \
  --limit-tiles 8 \
  --batch-size 8 \
  --hf-cache-dir /project/rrg-jma/${USER}/hf_cache
```

## Full Extraction

```bash
python scripts/extract_zip_tile_features.py \
  --input-root /project/rrg-jma/shared/SLC-PFM \
  --output-root /project/rrg-jma/shared/SLC-PFM_features \
  --encoders virchow2,hoptimus1,uni_v2 \
  --batch-size 64 \
  --hf-cache-dir /project/rrg-jma/${USER}/hf_cache
```

Add `--include-token-maps` only if you truly need patch-token grids. They are much larger than per-tile embeddings.

## Fir Slurm

Use the included template for GPU extraction on Fir:

```bash
sbatch --array=0-0 slurm/extract_features_fir.sbatch
```

The command above is a one-chunk smoke test. Once it succeeds, increase the array range to cover all chunk directories. The script discovers `chunk_*` folders under `INPUT_ROOT` and selects one by `SLURM_ARRAY_TASK_ID`.
