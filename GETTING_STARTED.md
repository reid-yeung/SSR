# Getting started

This guide covers the public reconstruction and generation pipeline. Run commands from the repository root. Use Python 3.10 and a CUDA-enabled PyTorch/torchvision build suitable for your GPUs. The code was import-checked with PyTorch 2.8.0 and torchvision 0.23.0. Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

Training and reconstruction use ImageNet-1K in `torchvision.datasets.ImageFolder` layout, with one subdirectory per class. The examples below use `/path/to/imagenet/train` and `/path/to/imagenet/val`; replace these paths with your dataset. Prepare the DINOv3-B Hugging Face-format model under `models/dinov3b/`, and the discriminator weight at `models/discs/dino_vit_small_patch8_224.pth`. Model weights and ImageNet images are not stored in this repository. The LPIPS weight is downloaded by the training code on first use.

The `torchrun` examples launch eight GPUs on one node. The paper's Stage 1 run used 16 H100 GPUs; use a multi-node launcher if reproducing that hardware setup. The YAML fixes the global batch size independently of the GPU count.

The public configs use these stable checkpoint names:

| File | Purpose |
| --- | --- |
| `checkpoints/ssr_stage1.pt` | Merged Stage 1 EMA autoencoder |
| `checkpoints/ssr_latent_stats.pt` | Latent normalization statistics for the same autoencoder |
| `checkpoints/ssr_dit_xl.pt` | EMA DiT-DH-XL generator |
| `checkpoints/ssr_dit_s.pt` | EMA DiT-DH-S AutoGuidance model |

Place released weights at these paths to run reconstruction or sampling directly. To train from scratch, follow the steps below and link your outputs to these names.

## 1. Train the representation autoencoder

Stage 1 trains the decoder for 12 epochs, then adapts the DINOv3-B student encoder with MLP LoRA for 18 more epochs. The frozen teacher supplies latent alignment, patch self-similarity, and Gaussian high-pass structural alignment. The released configuration uses 30 total epochs and a high-pass cutoff ratio of 0.05.

```bash
EXPERIMENT_NAME=ssr-stage1 torchrun --standalone --nproc_per_node=8 \
  src/train_stage1.py --config configs/stage1/train.yaml \
  --data-path /path/to/imagenet/train --results-dir outputs --precision bf16

mkdir -p checkpoints
ln -s ../outputs/ssr-stage1/checkpoints/ep-last.pt checkpoints/ssr_stage1.pt
```

The final `ep-last.pt` is the merged, Stage 2-compatible EMA checkpoint. If you already have the released checkpoint, place it directly at `checkpoints/ssr_stage1.pt` and skip training.

## 2. Reconstruct and score offline

The reconstruction script writes both reconstructed PNGs and the matching center-cropped input PNGs. It also packs each set into an NPZ file in the same ImageFolder index order.

```bash
SAVE_FOLDER=ssr-reconstruction torchrun --standalone --nproc_per_node=8 \
  src/stage1_sample_ddp.py --config configs/stage1/reconstruct.yaml \
  --data-path /path/to/imagenet/val --sample-dir outputs/reconstruction \
  --precision fp32

python src/evaluate_reconstruction.py \
  --reference outputs/reconstruction/ssr-reconstruction-input.npz \
  --reconstruction outputs/reconstruction/ssr-reconstruction.npz \
  --output outputs/reconstruction/metrics.json
```

The offline evaluator reports PSNR, SSIM, and reconstruction FID using `torch-fidelity`. For a comparable score, use the same validation split, center crop, sample count, and metric implementation. Training does not compute an online reconstruction score.

## 3. Compute latent statistics

Use the same merged Stage 1 checkpoint for latent statistics and Stage 2 training.

```bash
SAVE_FOLDER=ssr torchrun --standalone --nproc_per_node=8 \
  src/calculate_stat.py --config configs/stage1/reconstruct.yaml \
  --data-path /path/to/imagenet/train --sample-dir outputs/stats \
  --per-proc-batch-size 256 --precision fp32

ln -s ../outputs/stats/ssr/normalization_stats.pt checkpoints/ssr_latent_stats.pt
```

## 4. Train the generator

Train the main DiT-DH-XL model for 80 epochs and the smaller AutoGuidance model for 14 epochs. Both configs preserve the linear learning-rate schedule with an 800-epoch decay horizon used for these checkpoints; `epochs` sets the stopping point. Sampling loads EMA weights from the saved checkpoints.

```bash
EXPERIMENT_NAME=ssr-dit-xl torchrun --standalone --nproc_per_node=8 \
  src/train.py --config configs/stage2/train_xl.yaml \
  --data-path /path/to/imagenet/train --results-dir outputs --precision bf16

EXPERIMENT_NAME=ssr-dit-s torchrun --standalone --nproc_per_node=8 \
  src/train.py --config configs/stage2/train_s.yaml \
  --data-path /path/to/imagenet/train --results-dir outputs --precision bf16

ln -s ../outputs/ssr-dit-xl/checkpoints/ep-last.pt checkpoints/ssr_dit_xl.pt
ln -s ../outputs/ssr-dit-s/checkpoints/ep-last.pt checkpoints/ssr_dit_s.pt
```

## 5. Sample and score offline

Both sampling configs use 50-step Euler integration. The AutoGuidance config uses scale 1.42 and the small model. These commands produce 50,000 PNGs and an NPZ per run; `sample_ddp.py` prints the NPZ path.

```bash
torchrun --standalone --nproc_per_node=8 src/sample_ddp.py \
  --config configs/stage2/sample_xl.yaml --sample-dir outputs/samples/unguided \
  --num-fid-samples 50000 --per-proc-batch-size 125 \
  --precision bf16 --label-sampling equal

torchrun --standalone --nproc_per_node=8 src/sample_ddp.py \
  --config configs/stage2/sample_xl_autoguidance.yaml \
  --sample-dir outputs/samples/autoguidance --num-fid-samples 50000 \
  --per-proc-batch-size 125 --precision bf16 --label-sampling equal
```

Score each generated NPZ separately with the [ADM evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) and its ImageNet 256 reference batch:

```bash
python /path/to/guided-diffusion/evaluations/evaluator.py \
  /path/to/VIRTUAL_imagenet256_labeled.npz /path/to/generated_samples.npz
```

Generation FID is reported from this offline evaluator, not from training-time sampling.
