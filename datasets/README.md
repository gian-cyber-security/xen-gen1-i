# Dataset template (not training data)

This directory is intentionally empty. Provide a real image-caption JSONL manifest and the referenced files. Supported image formats are JPG, JPEG, PNG, WEBP, and BMP. Images must be readable and at least 2x2 pixels.

Conceptual record (EXAMPLE ONLY; do not use as a dataset):
`{"image":"images/example.png","caption":"EXAMPLE caption"}`

Validate with:
`python tools/validate_dataset.py --kind image --data datasets/image_data.jsonl`

The validator is read-only and reports invalid JSONL, missing files, unsupported formats, corrupt images, empty captions, and duplicate references.
