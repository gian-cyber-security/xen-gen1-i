---
license: mit
library_name: pytorch
pipeline_tag: text-to-image
language:
  - en
tags:
  - xens
  - xen
  - pytorch
  - custom-architecture
---

# XEN-GEN1-I

XEN-GEN1-I is the **image generation model** in the XEN family. It is designed for local text-to-image generation using a small conditional diffusion model trained from scratch.

## What it can do

- Text-to-image generation
- Prompt-conditioned image generation
- Image-to-image editing with a reference image
- Adjustable edit strength
- Local CUDA/CPU inference
- Train on your own image + caption dataset
- Generate PNG images locally

## Supported platforms

- Windows
- Linux
- macOS

## Requirements

Recommended development hardware:

- NVIDIA GPU with CUDA support
- RTX 4060 8GB is the current baseline
- 32GB RAM recommended
- Python 3.10+
- PyTorch 2.4+

### Windows

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your system uses `python` for Python 3, you can use `python` instead of `python3`.

## Generate an image

After you have a trained checkpoint in `outputs/xen-gen1-i`, run:

```bash
python inference/image_generate.py --model-dir outputs/xen-gen1-i --prompt "a futuristic city at night"
```

The generated image is saved to:

```
outputs/xen-gen1-i/generated.png
```

### Useful generation options

```bash
python inference/image_generate.py \
  --model-dir outputs/xen-gen1-i \
  --prompt "a small orange cat sitting in a garden" \
  --output outputs/cat.png \
  --size 256 \
  --steps 50 \
  --seed 42
```

Windows CMD version:

```bat
python inference/image_generate.py --model-dir outputs/xen-gen1-i --prompt "a small orange cat sitting in a garden" --output outputs/cat.png --size 256 --steps 50 --seed 42
```

### Important

XEN-GEN1-I is a from-scratch research/development model. The repository does **not** contain a pretrained commercial checkpoint. You need to train the model first or provide a compatible checkpoint.

## Train the model

Create:

`datasets/image_data.jsonl`

Example:

```json
{"image":"datasets/images/cat.jpg","caption":"a small orange cat sitting in a garden"}
{"image":"datasets/images/car.jpg","caption":"a red sports car driving through a city at night"}
```

Put the image files at the paths used by the JSONL file.

Start training:

```bash
python training/image_train.py --data datasets/image_data.jsonl --output outputs/xen-gen1-i --steps 10000 --batch-size 2
```

For an RTX 4060 8GB, start with a small batch size. If VRAM is insufficient, reduce the batch size and image size according to the training script options.

## Dataset format

Each line must contain:

- `image`: path to an image file
- `caption`: text description of the image

Example:

```json
{"image":"datasets/images/dog.jpg","caption":"a brown dog running on grass"}
```

## Folder structure

```text
xen-gen1-i/
├── configs/
│   └── system_prompt_i.txt
├── datasets/
│   └── image_data.jsonl
├── inference/
│   └── image_generate.py
├── model/
│   ├── image_conditioner.py
│   ├── image_model.py
│   └── tokenizer.py
├── training/
│   └── image_train.py
├── requirements.txt
└── README.md
```

## Local usage flow

1. Install Python dependencies.
2. Prepare image/caption JSONL data.
3. Train XEN-GEN1-I.
4. Keep the resulting checkpoint in `outputs/xen-gen1-i`.
5. Run `inference/image_generate.py` with your prompt.
6. Open the generated PNG.

## Hugging Face

The model can be packaged for a Hugging Face Space or other compatible deployment. A GPU is recommended because image diffusion inference is computationally expensive.

## Notes

This is a research/development model, not a pretrained Stable Diffusion/Flux-style model. Generation quality depends heavily on the dataset, model capacity, training steps, and available GPU memory.

## License

MIT
