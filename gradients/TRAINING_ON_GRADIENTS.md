# Training Affine Models on Gradients

## Overview

Want to train models for Affine's SAT & ABD challenges? Use these scripts to generate datasets, upload to S3, and train on Gradients via SFT or GRPO.

## Prerequisites

Before running these scripts, ensure you have:

1. **Gradients API key** - Sign up at [gradients.io](https://gradients.io)
2. **S3 credentials** for dataset storage
3. **Affine module** installed in your environment

### Environment Setup (.env)

Create a `.env` file in the affine project root directory (not in the gradients folder) with:

```bash
# S3 Configuration (Required for dataset upload)
S3_ENDPOINT=https://s3.region.amazonaws.com  # Your S3 endpoint
S3_ACCESS_KEY=your_access_key_here
S3_SECRET_KEY=your_secret_key_here
S3_REGION=us-east-1  # Your S3 region
S3_BUCKET_NAME=your_bucket_name

# Required for generate_grpo_data.py (ABD uses LLM to create diverse inputs)
CHUTES_API_KEY=your_chutes_api_key

# Optional: Affine configuration for SFT data extraction
SUBTENSOR_ENDPOINT=finney  # or your custom endpoint
```

## Dataset Generation Scripts

### 1. `generate_sft_data.py` - Extract Real Affine Data for SFT

This script collects actual evaluation data from the Affine subnet, perfect for supervised fine-tuning.

**Usage:**
```bash
python -m gradients.generate_sft_data --max-results 1000 --min-score 0.5 --delete-local
```

**Parameters:**
- `--max-results` - Number of examples to collect (default: unlimited)
- `--min-score` - Filter threshold - only include responses scoring above this value
  - Example: `0.5` = only train on successful responses
  - Use this to ensure your model learns from high-quality examples
- `--tail` - Process last N blocks from the chain (default: 100)
- `--presign-hours` - S3 URL expiration time in hours (default: 168 = 1 week)
- `--delete-local` - Remove local JSON file after successful S3 upload
- `--dry-run` - Test without uploading to S3

**Output:** 
- Uploads dataset to S3 automatically
- Returns a presigned URL ready for use with Gradients
- Dataset format: Flat JSON array with prompts, responses, and scores

### 2. `generate_grpo_data.py` - Create Synthetic Data for GRPO

This script generates synthetic SAT and ABD problems with known solutions, ideal for reinforcement learning with GRPO.

**Usage:**
```bash
python -m gradients.generate_grpo_data -n 1000 --validate --delete-local
```

**Parameters:**
- `-n, --num-examples` - Number of synthetic examples to generate
- `--sat-ratio` - Ratio of SAT vs ABD problems (0.0-1.0, default: 0.5)
- `--validate` - Verify dataset structure before upload
- `--presign-hours` - S3 URL expiration time in hours (default: 168 = 1 week)
- `--delete-local` - Remove local JSON file after successful S3 upload
- `--dry-run` - Test without uploading to S3

**Output:**
- Uploads dataset to S3 automatically
- Returns a presigned URL ready for use with Gradients
- Dataset format: JSON with `env` field containing `sat` and `abd` keys for reward calculation

## Training on Gradients

### Stage 1: Supervised Fine-Tuning (SFT)

Use real Affine data to teach your model the task structure and successful response patterns.

**Dataset:** Use the S3 URL from `generate_sft_data.py`

**API Endpoint:** `/v1/tasks/create_custom_dataset_text`

**Example Request:**
```bash
curl https://api.gradients.io/v1/tasks/create_custom_dataset_text \
  --request POST \
  --header 'Authorization: Bearer YOUR_GRADIENTS_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "field_instruction": "prompt",
    "field_output": "response",
    "model_repo": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "training_data": "[S3_URL_from_generate_sft_data]",
    "file_format": "s3",
    "hours_to_complete": 2
  }'
```

**Key Fields:**
- `model_repo` - Any HuggingFace model as your starting point
- `training_data` - The S3 presigned URL from `affine_upload.py`
- `field_instruction` - Set to `"prompt"` for Affine data
- `field_output` - Set to `"response"` for Affine data

### Stage 2: Group Relative Policy Optimization (GRPO)

Use synthetic data with exact solutions to optimize your model through reinforcement learning.

**Dataset:** Use the S3 URL from `generate_grpo_data.py`

**API Endpoint:** `/v1/tasks/create_grpo`

**Example Request:**
```bash
curl https://api.gradients.io/v1/tasks/create_grpo \
  --request POST \
  --header 'Authorization: Bearer YOUR_GRADIENTS_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
    "field_prompt": "prompt",
    "ds_repo": "[S3_URL_from_generate_grpo_data]",
    "model_repo": "YOUR_FINETUNED_MODEL_OR_BASE_MODEL",
    "file_format": "s3",
    "reward_function_ids": ["sat_reward_fn", "abd_reward_fn"],
    "hours_to_complete": 2
  }'
```

**Key Fields:**
- `model_repo` - Use your SFT model from Stage 1 or any HuggingFace model
- `ds_repo` - The S3 presigned URL from `generate_synthetic_dataset.py`
- `reward_function_ids` - Specific functions that score SAT and ABD responses
- `field_prompt` - Set to `"prompt"` for the synthetic dataset

## Workflow Example

1. **Generate SFT training data from real Affine evaluations:**
   ```bash
   python -m gradients.generate_sft_data --max-results 5000 --min-score 0.7 --delete-local
   ```
   This collects 5000 high-quality examples (score > 0.7) and uploads to S3.

2. **Train with SFT on Gradients:**
   Use the returned S3 URL in the SFT API call to create your initial fine-tuned model.

3. **Generate synthetic data for GRPO:**
   ```bash
   python -m gradients.generate_grpo_data -n 10000 --sat-ratio 0.5 --validate --delete-local
   ```
   This creates 10,000 examples (50% SAT, 50% ABD) with ground truth for rewards.

4. **Optimize with GRPO on Gradients:**
   Use the synthetic dataset URL and your SFT model to further optimize performance.

## Important Notes

- **S3 Upload:** Both scripts automatically upload to S3 (required for Gradients to access your data)
- **URL Expiration:** S3 presigned URLs expire after the specified time (default: 7 days). Generate new URLs by re-running scripts if needed.
- **Model Selection:** You can use any HuggingFace model as your `model_repo` starting point
- **Quality Filtering:** Use `--min-score` parameter to filter training data quality. For example:
  - `--min-score 0.0` - Include all data
  - `--min-score 0.5` - Only successful responses
  - `--min-score 0.9` - Only highest quality responses
- **Dataset Formats:**
  - SFT data: Simple prompt-response pairs with scores
  - GRPO data: Includes `env` field with problem specifications for reward calculation

## Troubleshooting

- **S3 Upload Errors:** Check your `.env` file has correct S3 credentials
- **Affine Module Errors:** Ensure the affine module is properly installed
- **Gradients API Errors:** Verify your API key and endpoint URLs
- **Expired URLs:** Re-run the script to generate fresh presigned URLs