# sft_vlm S3 PNG demo dataset

A tiny multimodal SFT dataset whose images live in S3 as PNGs, for exercising
the `s3://` image support in `sft_vlm`.

## Files

- `images/*.png` — 5 demo images (cat, bee, cats, beignets), capped at 1024 px
  on the long side. These are the local source files you upload to your bucket.
- `example_data_s3.csv` — `image_url,question,response` rows where each
  `image_url` is an `s3://your-bucket/datasets/vlm_images/<name>.png` URI.
- `upload_example_to_s3.sh` — uploads the PNGs to your bucket and writes a
  bucket-specific copy of the CSV to `s3://<bucket>/datasets/vlm_train.csv`.

## Usage

1. Upload images + dataset CSV to your bucket (requires the AWS CLI):

   ```bash
   ./upload_example_to_s3.sh my-vlm-bucket
   # or with a key prefix:
   ./upload_example_to_s3.sh my-vlm-bucket team-a/experiments
   ```

2. Train. `qwen2_5_vl_3b_lora_s3_file.yaml` already points at
   `s3://your-bucket/datasets/vlm_train.csv` — change `your-bucket` to your
   bucket name (and add the optional prefix if you used one):

   ```bash
   python -m trainers.train --config recipes/training/sft_vlm/qwen2_5_vl_3b_lora_s3_file.yaml
   ```

   The dataset file is downloaded from S3, and each row's `image_url` `s3://`
   PNG is fetched via boto3 at load time (cached under `image_cache_dir`).

## S3 connection settings

`s3://` images use the boto3 default credential/region chain. To override the
region or use an S3-compatible endpoint (e.g. MinIO), set in the recipe's
`data:` block:

```yaml
data:
  image_s3_region: "us-west-2"
  image_s3_endpoint_url: null   # e.g. http://minio:9000
```

These are independent of each dataset's `s3_region` / `s3_endpoint_url`, which
govern the dataset *file*.
