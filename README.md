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

The default Fir request is intentionally modest:

```text
1x 40 GB H100 MIG slice, 4 CPUs, 32 GB RAM, 12 hours
```

If a run fails because a model does not fit, override only what is needed at submit time:

```bash
sbatch \
  --gpus-per-node=h100:1 \
  --mem=64G \
  --cpus-per-task=4 \
  --array=0-0 \
  slurm/extract_features_fir.sbatch
```

After each completed run, inspect usage and keep trimming requests:

```bash
seff <jobid_or_array_task>
```

The Slurm scripts also start a resource monitor by default. For distillation jobs, it samples CPU, CPU memory, GPU, and
GPU memory every 5 seconds and prints both 60-second window peaks and a final whole-job summary to the Slurm `.out` log:

```text
[resource-summary] ... max_cpu_utilization=... max_cpu_memory_utilization=... max_gpu_utilization=... max_gpu_memory_utilization=...
```

After a smoke test finishes, get the final peak utilization line with:

```bash
grep '\[resource-summary\]' logs/slc-pfm-distill_<jobid>.out
```

You can change or disable monitoring at submit time:

```bash
sbatch \
  --export=ALL,RESOURCE_MONITOR_REPORT_SECONDS=60,RESOURCE_MONITOR_SAMPLE_SECONDS=5 \
  slurm/train_cradio_distill_fir.sbatch

sbatch \
  --export=ALL,RESOURCE_MONITOR=0 \
  slurm/train_cradio_distill_fir.sbatch
```

## C-RADIO Distillation

After extracting `virchow2`, `hoptimus1`, and `uni_v2` features, continue training from NVIDIA's C-RADIOv4 checkpoint
with one new projection head per pathology teacher:

```bash
python scripts/train_cradio_distill.py \
  --input-root /project/rrg-jma/shared/SLC-PFM \
  --feature-root /project/rrg-jma/shared/SLC-PFM_features \
  --output-dir /project/rrg-jma/shared/SLC-PFM_distill/cradio_v4_so400m_virchow_hoptimus_uni \
  --encoders virchow2,hoptimus1,uni_v2 \
  --radio-version c-radio_v4-so400m \
  --batch-size 64 \
  --max-steps 100000
```

The implementation uses the actual `NVlabs/RADIO` TorchHub model as the student backbone and initializes it from
`c-radio_v4-so400m` by default. Use `RADIO_VERSION=c-radio_v4-h` to start from C-RADIOv4-H instead. The current feature
files contain CLS/summary embeddings by default, so this trains the C-RADIO summary distillation path for our three
encoders.

To reproduce the dense C-RADIO objective without storing the huge patch-token maps, use the saved summary embeddings and
run the three pathology teachers frozen inside each training step. This extracts patch tokens on the fly and applies the
dense spatial loss immediately:

```bash
sbatch \
  --time=02:00:00 \
  --export=ALL,CODE_DIR=/scratch/atatc/app/SLC-PFM,INPUT_ROOT=/project/rrg-jma/shared/SLC-PFM,FEATURE_ROOT=/project/rrg-jma/shared/SLC-PFM_features,OUTPUT_DIR=/project/rrg-jma/shared/SLC-PFM_distill/cradio_v4_so400m_online_patch_smoke,ONLINE_TOKEN_TEACHERS=1,LIMIT_ZIPS=4,STATS_MAX_FILES=4,MAX_STEPS=20,ESTIMATE_TOTAL_STEPS=100000,BATCH_SIZE=8,RESOURCE_MONITOR_SAMPLE_SECONDS=1,RESOURCE_MONITOR_REPORT_SECONDS=60 \
  slurm/train_cradio_distill_fir.sbatch
```

Full online patch-token run:

```bash
sbatch \
  --time=36:00:00 \
  --export=ALL,CODE_DIR=/scratch/atatc/app/SLC-PFM,INPUT_ROOT=/project/rrg-jma/shared/SLC-PFM,FEATURE_ROOT=/project/rrg-jma/shared/SLC-PFM_features,OUTPUT_DIR=/project/rrg-jma/shared/SLC-PFM_distill/cradio_v4_so400m_online_patch_virchow_hoptimus_uni,ONLINE_TOKEN_TEACHERS=1,BATCH_SIZE=16,EPOCHS=3,AUTO_RESUME=1,SAVE_EVERY=1000 \
  slurm/train_cradio_distill_fir.sbatch
```

On-the-fly patch-token training is much slower than summary-only distillation because every batch runs the C-RADIO
student plus all three frozen teachers. The current Slurm script requests one full H100, 8 CPUs, and 32 GB memory for
the larger `BATCH_SIZE=16` run.

Checkpoints are written as both `checkpoint_step*.pt` and `checkpoint_latest.pt`. With `AUTO_RESUME=1`, a restarted job
continues from the latest checkpoint in `OUTPUT_DIR`. If a job stops mid-epoch, resume restarts that epoch from the
beginning while preserving model, optimizer, scaler, and global step state.

If you prefer to spend storage instead of recomputing teacher patch tokens every epoch, you can still extract only dense
patch-token maps to a separate token-map root:

```bash
N=$(find /project/rrg-jma/shared/SLC-PFM -mindepth 1 -maxdepth 1 -type d -name 'chunk_*' | wc -l)

sbatch \
  --array=0-$((N-1))%2 \
  --export=ALL,CODE_DIR=/scratch/atatc/app/SLC-PFM,INPUT_ROOT=/project/rrg-jma/shared/SLC-PFM,OUTPUT_ROOT=/project/rrg-jma/shared/SLC-PFM_token_maps,ENCODERS=virchow2,hoptimus1,uni_v2,BATCH_SIZE=32,TOKEN_MAPS_ONLY=1 \
  slurm/extract_features_fir.sbatch
```

Smoke test dense extraction first:

```bash
sbatch \
  --array=0-0 \
  --time=00:30:00 \
  --export=ALL,CODE_DIR=/scratch/atatc/app/SLC-PFM,INPUT_ROOT=/project/rrg-jma/shared/SLC-PFM,OUTPUT_ROOT=/project/rrg-jma/shared/SLC-PFM_token_maps,ENCODERS=virchow2,LIMIT_ZIPS=1,LIMIT_TILES=128,BATCH_SIZE=16,TOKEN_MAPS_ONLY=1 \
  slurm/extract_features_fir.sbatch
```

Dense C-RADIO run after `SLC-PFM_token_maps` is ready. `FEATURE_ROOT` points to your existing summary embeddings, and
`TOKEN_FEATURE_ROOT` points to the token-map-only files:

```bash
sbatch \
  --export=ALL,CODE_DIR=/scratch/atatc/app/SLC-PFM,FEATURE_ROOT=/project/rrg-jma/shared/SLC-PFM_features,TOKEN_FEATURE_ROOT=/project/rrg-jma/shared/SLC-PFM_token_maps,OUTPUT_DIR=/project/rrg-jma/shared/SLC-PFM_distill/cradio_v4_so400m_dense_virchow_hoptimus_uni,INCLUDE_TOKEN_MAPS=1,BATCH_SIZE=16,MAX_STEPS=100000 \
  slurm/train_cradio_distill_fir.sbatch
```

Checkpoints and cached teacher statistics are written under `OUTPUT_DIR`.
