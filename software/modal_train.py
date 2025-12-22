"""
Modal training script for XLeRobot π0.5 fine-tuning.

!!! IMPORTANT !!!
MUST USE π0.5 (pi05) POLICY - NOT ACT!
π0.5 is a foundation model that works with 10-50 demos.
ACT requires 100s of demos and much more training.
DO NOT SWITCH TO ACT - it will fail with limited data.

Usage:
    # First, upload dataset to Modal volume
    modal run modal_train.py --upload-dataset

    # Then run training
    modal run modal_train.py --train
"""

import modal
import os

app = modal.App("xlerobot-pi05-training")

# Create volume for dataset and outputs
volume = modal.Volume.from_name("xlerobot-data", create_if_missing=True)

# Docker image with lerobot and dependencies
# π0.5 requires OpenPI's transformers_replace patches for GemmaRMSNorm with cond argument
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6", "git-lfs", "curl")
    .run_commands("git lfs install")
    .pip_install(
        "torch>=2.0",
        "torchvision",
        "accelerate",
        "wandb",
        "pyarrow",
        "datasets",
        "huggingface_hub==0.35.2",
        "fsspec==2025.9.0",
    )
    # Install lerobot with pi extras from v0.4.2 tag
    .pip_install("lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.4.2")
    # Install transformers 4.53.3 (the version OpenPI patches are designed for)
    .run_commands("pip install --force-reinstall transformers==4.53.3")
    # Clone OpenPI and apply their transformers_replace patches
    # These patches add the cond argument to GemmaRMSNorm and other pi05-specific fixes
    .run_commands(
        "git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git /tmp/openpi && "
        "cp -r /tmp/openpi/src/openpi/models_pytorch/transformers_replace/models/* /usr/local/lib/python3.10/site-packages/transformers/models/ && "
        "rm -rf /tmp/openpi && "
        "echo 'Applied OpenPI transformers_replace patches'"
    )
    # Add the check.py file that lerobot pi05 expects
    .run_commands(
        "python -c \"import os; "
        "path='/usr/local/lib/python3.10/site-packages/transformers/models/siglip/check.py'; "
        "code='import transformers\\n\\ndef check_whether_transformers_replace_is_installed_correctly():\\n    return True  # Patched from OpenPI\\n'; "
        "open(path, 'w').write(code); print(f'Created {path}')\""
    )
    # Verify the patches are applied
    .run_commands(
        "python -c \"from transformers.models.gemma.modeling_gemma import GemmaRMSNorm; "
        "import inspect; sig = inspect.signature(GemmaRMSNorm.forward); "
        "assert 'cond' in sig.parameters, 'GemmaRMSNorm patch not applied'; "
        "print('GemmaRMSNorm cond argument: OK')\""
    )
)

DATASET_LOCAL_PATH = "/home/abay/Projects/xle_robot/XLeRobot/software/dataset_merged_104ep"
DATASET_VOLUME_PATH = "/data/dataset"
OUTPUT_PATH = "/data/outputs"


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
)
def upload_dataset():
    """Upload local dataset to Modal volume."""
    import shutil

    print(f"Dataset will be uploaded to volume at: {DATASET_VOLUME_PATH}")
    print("Upload is handled by the local_entrypoint")
    return {"status": "ready", "path": DATASET_VOLUME_PATH}


@app.function(
    image=image,
    gpu="H100",  # or "A100" for cheaper option
    volumes={"/data": volume},
    timeout=14400,  # 4 hours for longer training
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def train(
    steps: int = 5000,  # π0.5 needs fewer steps (foundation model)
    batch_size: int = 2,  # Smaller batch for large model
    learning_rate: float = 2e-5,  # Slightly higher LR for fine-tuning
    resume_run_id: str = None,  # Resume from existing run
):
    """Run π0.5 training on Modal H100."""
    import subprocess
    import os
    import uuid

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    # Check dataset exists
    if not os.path.exists(DATASET_VOLUME_PATH):
        raise RuntimeError(f"Dataset not found at {DATASET_VOLUME_PATH}. Run --upload-dataset first.")

    print(f"Dataset found at: {DATASET_VOLUME_PATH}")
    print(f"Training for {steps} steps with batch_size={batch_size}")

    # Use existing run_id if resuming, otherwise create new
    if resume_run_id:
        run_id = resume_run_id
        print(f"RESUMING from run: {run_id}")
    else:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
    model_output_dir = f"{OUTPUT_PATH}/{run_id}"
    print(f"Output directory: {model_output_dir}")

    # !!! MUST USE PI05 (π0.5) - NOT ACT !!!
    # π0.5 is a foundation model that works with 10-50 demos
    # ACT requires 100s of demos - DO NOT USE ACT
    cmd = [
        "python", "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.root={DATASET_VOLUME_PATH}",
        "--dataset.repo_id=xlerobot_towel_pickup",
        "--policy.type=pi05",  # MUST BE pi05, NOT act
        "--policy.pretrained_path=lerobot/pi05_base",  # Official checkpoint
        "--policy.push_to_hub=false",
        "--policy.gradient_checkpointing=true",  # Reduce memory usage
        "--policy.dtype=bfloat16",  # Mixed precision (patched transformers handles this)
        # Use MEAN_STD normalization since our dataset doesn't have quantiles
        '--policy.normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}',
        f"--output_dir={model_output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--optimizer.lr={learning_rate}",
        "--save_freq=1000",
    ]

    # Add resume flag if resuming
    if resume_run_id:
        cmd.append("--resume=true")
        # Point to the checkpoint config
        checkpoint_config = f"{model_output_dir}/checkpoints/last/pretrained_model/train_config.json"
        cmd.append(f"--config_path={checkpoint_config}")

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        # Try to save whatever was trained

    # Commit the volume to persist outputs
    volume.commit()

    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "output_path": model_output_dir,
        "run_id": run_id,
        "return_code": result.returncode,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def check_status():
    """Check what's in the volume."""
    import os

    results = {}

    # Check dataset
    if os.path.exists(DATASET_VOLUME_PATH):
        results["dataset"] = "exists"
        # Count files
        total_files = 0
        for root, dirs, files in os.walk(DATASET_VOLUME_PATH):
            total_files += len(files)
        results["dataset_files"] = total_files
    else:
        results["dataset"] = "missing"

    # Check outputs
    if os.path.exists(OUTPUT_PATH):
        results["outputs"] = os.listdir(OUTPUT_PATH)
    else:
        results["outputs"] = "none"

    return results


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def list_runs():
    """List all training runs in the output directory."""
    import os

    if not os.path.exists(OUTPUT_PATH):
        return {"runs": []}

    runs = []
    for item in os.listdir(OUTPUT_PATH):
        item_path = os.path.join(OUTPUT_PATH, item)
        if os.path.isdir(item_path):
            # Check for checkpoint directories
            checkpoints = []
            checkpoint_dir = os.path.join(item_path, "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = sorted(os.listdir(checkpoint_dir))
            runs.append({"run_id": item, "checkpoints": checkpoints})
    return {"runs": runs}


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def list_model_files(run_id: str = None, checkpoint: str = None):
    """List all files in the model output directory."""
    import os

    # Determine model path
    if run_id:
        if checkpoint:
            model_path = f"{OUTPUT_PATH}/{run_id}/checkpoints/{checkpoint}/pretrained_model"
        else:
            # Find latest checkpoint
            checkpoint_dir = f"{OUTPUT_PATH}/{run_id}/checkpoints"
            if os.path.exists(checkpoint_dir):
                checkpoints = sorted(os.listdir(checkpoint_dir))
                if checkpoints:
                    checkpoint = checkpoints[-1]
                    model_path = f"{checkpoint_dir}/{checkpoint}/pretrained_model"
                else:
                    return {"error": "No checkpoints found", "files": []}
            else:
                return {"error": "No checkpoints directory", "files": []}
    else:
        model_path = f"{OUTPUT_PATH}/model"

    if not os.path.exists(model_path):
        return {"error": f"Model not found at {model_path}", "files": [], "model_path": model_path}

    files = []
    for root, dirs, filenames in os.walk(model_path):
        for f in filenames:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, model_path)
            size = os.path.getsize(full_path)
            files.append({"path": rel_path, "size": size})
    return {"files": files, "model_path": model_path, "checkpoint": checkpoint}


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
)
def get_model_file(rel_path: str, run_id: str = None, checkpoint: str = None):
    """Get contents of a specific model file."""
    import os

    if run_id:
        if not checkpoint:
            # Find latest checkpoint
            checkpoint_dir = f"{OUTPUT_PATH}/{run_id}/checkpoints"
            if os.path.exists(checkpoint_dir):
                checkpoints = sorted(os.listdir(checkpoint_dir))
                if checkpoints:
                    checkpoint = checkpoints[-1]
        full_path = f"{OUTPUT_PATH}/{run_id}/checkpoints/{checkpoint}/pretrained_model/{rel_path}"
    else:
        full_path = f"{OUTPUT_PATH}/model/{rel_path}"

    if not os.path.exists(full_path):
        return None
    with open(full_path, "rb") as f:
        return f.read()


MODEL_LOCAL_PATH = "/home/abay/Projects/xle_robot/XLeRobot/software/trained_model"


@app.local_entrypoint()
def main(
    upload_dataset_flag: bool = False,
    train_flag: bool = False,
    check: bool = False,
    download_model: bool = False,
    run_id: str = None,
    resume_run_id: str = None,
    steps: int = 5000,  # π0.5 needs fewer steps
    batch_size: int = 2,  # Smaller batch for large model
):
    """
    Main entrypoint.

    Args:
        upload_dataset_flag: Upload local dataset to Modal volume
        train_flag: Run training
        check: Check volume status
        download_model: Download trained model
        run_id: Specific run ID to download (optional, uses latest if not specified)
        resume_run_id: Resume training from this run ID
        steps: Training steps
        batch_size: Batch size for training
    """
    if check:
        print("Checking volume status...")
        result = check_status.remote()
        print(f"Status: {result}")
        return

    if upload_dataset_flag:
        print(f"Uploading dataset from {DATASET_LOCAL_PATH}...")

        import shutil
        import tempfile

        # Clear existing dataset in volume first
        print("Clearing existing dataset in volume...")
        try:
            for item in volume.listdir("dataset", recursive=False):
                volume.remove_file(f"dataset/{item.path}")
        except Exception as e:
            print(f"Note: {e}")

        # Create a temporary directory with the dataset structure
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "dataset")
            shutil.copytree(DATASET_LOCAL_PATH, dest)

            # Upload to volume with force
            with volume.batch_upload(force=True) as batch:
                batch.put_directory(dest, "dataset")

        print("Dataset uploaded!")

        # Verify
        result = check_status.remote()
        print(f"Verification: {result}")
        return

    if train_flag:
        if resume_run_id:
            print(f"RESUMING training from {resume_run_id}: {steps} steps, batch_size={batch_size}")
            result = train.remote(steps=steps, batch_size=batch_size, resume_run_id=resume_run_id)
        else:
            print(f"Starting training: {steps} steps, batch_size={batch_size}")
            result = train.remote(steps=steps, batch_size=batch_size)
        print(f"Training result: {result}")
        return

    if download_model:
        print("Downloading trained model from Modal volume...")

        # If no run_id specified, list runs and use the latest
        if not run_id:
            runs_result = list_runs.remote()
            runs = runs_result.get("runs", [])
            if not runs:
                print("No training runs found!")
                return
            # Use the latest run (by name, which includes timestamp/uuid)
            run_id = sorted([r["run_id"] for r in runs])[-1]
            print(f"Using latest run: {run_id}")

        # List files in model directory
        result = list_model_files.remote(run_id=run_id)
        if "error" in result:
            print(f"Error: {result['error']}")
            return

        files = result["files"]
        checkpoint = result.get("checkpoint", "latest")
        print(f"Found {len(files)} files to download from checkpoint {checkpoint}")

        # Create local directory for this checkpoint
        local_checkpoint_path = os.path.join(MODEL_LOCAL_PATH, "checkpoints", checkpoint, "pretrained_model")
        os.makedirs(local_checkpoint_path, exist_ok=True)

        # Download each file
        for file_info in files:
            rel_path = file_info["path"]
            size = file_info["size"]
            print(f"  Downloading {rel_path} ({size} bytes)...")

            content = get_model_file.remote(rel_path, run_id=run_id, checkpoint=checkpoint)
            if content is None:
                print(f"    Failed to download {rel_path}")
                continue

            # Create subdirectories if needed
            local_path = os.path.join(local_checkpoint_path, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(content)

        print(f"Model downloaded to: {local_checkpoint_path}")
        return

    print("Usage:")
    print("  modal run modal_train.py --upload-dataset-flag  # Upload dataset")
    print("  modal run modal_train.py --train-flag           # Run training")
    print("  modal run modal_train.py --check                # Check status")
    print("  modal run modal_train.py --download-model       # Download trained model")
