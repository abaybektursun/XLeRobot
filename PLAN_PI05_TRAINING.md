# XLeRobot π0.5 VLA Training Plan

## Status: Using π0.5 (π0.6 NOT YET AVAILABLE)
As of Dec 2025, [π0.6 is not open-sourced](https://github.com/Physical-Intelligence/openpi/issues/791).
We use **π0.5** ([lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)) with open-world generalization.

---

## Overview

| Phase | Task | Time |
|-------|------|------|
| 1 | Environment Setup | 1-2h |
| 2 | Camera & Hardware Verification | 30min |
| 3 | Data Collection (50-100 episodes) | 2-4h |
| 4 | Data Preparation & Augmentation | 30min |
| 5 | Cloud Training | 1-2h |
| 6 | Validation & Evaluation | 30min |
| 7 | Deployment | 30min |

**Total: ~6-10 hours**

---

## Phase 1: Environment Setup

### 1.1 Local Machine (Data Collection & Deployment)

```bash
# Activate lerobot environment
conda activate lerobot

# Verify lerobot version (need 0.4.2+)
python -c "import lerobot; print(lerobot.__version__)"

# Install π0.5 dependencies
pip install -e ".[pi]"
# OR for lerobot 0.4.x:
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"

# Install async inference dependencies
pip install -e ".[async]"

# Verify xlerobot connection
cd /home/abay/Projects/xle_robot/XLeRobot/software
source /home/abay/miniconda3/etc/profile.d/conda.sh && conda activate lerobot
echo "" | python examples/simple_xlerobot_test.py
```

### 1.2 HuggingFace Setup

```bash
# Login to HuggingFace
huggingface-cli login

# Set your username
export HF_USER=$(huggingface-cli whoami | head -1)
echo "HF_USER: $HF_USER"
```

### 1.3 Cloud Setup (Training)

| Provider | GPU | Price/hr | Best For |
|----------|-----|----------|----------|
| [Modal](https://modal.com) | H100 80GB | ~$4 | Simple, serverless |
| [RunPod](https://runpod.io) | A100 80GB | ~$2 | Cost-effective |
| [Lambda](https://lambdalabs.com) | H100/H200 | ~$3 | Enterprise |

**GPU Requirements:**

| Method | VRAM | Recommended GPU |
|--------|------|-----------------|
| LoRA fine-tune | 22.5+ GB | A100 40GB |
| Full fine-tune | 70+ GB | H100 80GB |
| Multi-GPU (2x) | 2×40GB | 2× A100 40GB |

```bash
# Modal setup
pip install modal
modal setup
modal secret create huggingface-token HF_TOKEN=hf_xxxxx
```

---

## Phase 2: Camera & Hardware Verification

### 2.1 Detect Available Cameras

```bash
# Find all cameras
python -c "
import subprocess
result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
print(result.stdout)
"

# Or use lerobot's camera finder
python -m lerobot.find_cameras
```

### 2.2 Test Each Camera

```bash
# Test cameras individually (adjust indices based on your setup)
python -c "
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

cameras = {
    'head': 0,      # Adjust these indices
    'left_wrist': 2,
    'right_wrist': 4,
}

for name, idx in cameras.items():
    try:
        cfg = OpenCVCameraConfig(index_or_path=idx, width=640, height=480, fps=30)
        cam = OpenCVCamera(cfg)
        cam.connect()
        frame = cam.async_read()
        print(f'{name} (index {idx}): OK - shape {frame.shape}')
        cam.disconnect()
    except Exception as e:
        print(f'{name} (index {idx}): FAILED - {e}')
"
```

### 2.3 Document Your Camera Setup

Create a camera config file for consistency:

```bash
# Save your camera config
cat > camera_config.json << 'EOF'
{
    "head": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30},
    "left_wrist": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    "right_wrist": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30}
}
EOF
```

---

## Phase 3: Data Collection

### 3.1 Option A: VR Teleoperation (Recommended)

**Prerequisites:**
- Quest 3 headset with XLeVR app
- XLeRobot VR teleop module installed

```bash
# Copy xlerobot VR teleop to lerobot (one-time setup)
cp -r /home/abay/Projects/xle_robot/XLeRobot/software/src/teleporators/xlerobot_vr \
      /home/abay/miniconda3/envs/lerobot/lib/python3.10/site-packages/lerobot/teleoperators/

# Start VR server (in separate terminal)
# Follow XLeVR docs: https://xlerobot.readthedocs.io/en/latest/simulation/getting_started/vr_sim.html

# Record dataset
python /home/abay/miniconda3/envs/lerobot/lib/python3.10/site-packages/lerobot/scripts/record.py \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --robot.id=my_xlerobot \
  --robot.cameras='{
    head: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    left_wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30},
    right_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=xlerobot_vr \
  --dataset.repo_id=${HF_USER}/xlerobot_task \
  --dataset.num_episodes=50 \
  --dataset.single_task="Fold the shirt" \
  --dataset.push_to_hub=true \
  --display_data=true
```

### 3.2 Option B: Single Arm with Leader Arm

```bash
# For single-arm tasks (right arm example)
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=right_arm \
  --robot.cameras='{
    head: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30},
    wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM2 \
  --teleop.id=leader_arm \
  --dataset.repo_id=${HF_USER}/xlerobot_single_arm \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick up the cup" \
  --dataset.push_to_hub=true
```

### 3.3 Data Quality Checklist

- [ ] **Consistent head camera angle** - Keep servos at same position
- [ ] **50-100 episodes minimum** - More is better
- [ ] **Vary starting positions** - Slight variations improve generalization
- [ ] **Include recovery from errors** - Don't reset on small mistakes
- [ ] **Good lighting** - Consistent, no harsh shadows
- [ ] **Clean background** - Minimize distractions

---

## Phase 4: Data Preparation & Augmentation

### 4.1 Verify Dataset

```bash
# Check dataset on HuggingFace
python -c "
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset('${HF_USER}/xlerobot_task')
print(f'Episodes: {ds.num_episodes}')
print(f'Frames: {len(ds)}')
print(f'Features: {list(ds.meta.features.keys())}')
"
```

### 4.2 Add Quantile Statistics (Required for π0.5)

```bash
# π0.5 requires quantile normalization
python -m lerobot.scripts.augment_dataset_quantile_stats \
  --repo-id=${HF_USER}/xlerobot_task
```

**Alternative:** Train with explicit normalization mapping:
```bash
--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
```

### 4.3 Data Augmentation (Optional, 4x Data)

```bash
# Flip and mirror augmentation
python -m lerobot.scripts.augment_dataset \
  --dataset.repo_id=${HF_USER}/xlerobot_task \
  --output.repo_id=${HF_USER}/xlerobot_task_augmented
```

---

## Phase 5: Cloud Training with π0.5

### 5.0 Cloud Abstraction Architecture

> **Note:** LeRobot has [async inference](https://huggingface.co/docs/lerobot/en/async) for deployment,
> but **no built-in cloud training** ([issue #2172](https://github.com/huggingface/lerobot/issues/2172)).

```
xlerobot_cloud/
├── __init__.py
├── base.py              # Abstract CloudProvider interface
├── config.py            # TrainingConfig, InferenceConfig
├── providers/
│   ├── modal_provider.py    # Modal (default)
│   ├── runpod_provider.py   # RunPod (future)
│   └── lambda_provider.py   # Lambda Labs (future)
└── cli.py               # xlerobot-train, xlerobot-serve
```

### 5.1 Training Config

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Dataset
    dataset_repo_id: str

    # Model
    policy_type: str = "pi05"  # NOT "pi0"!
    pretrained_path: str = "lerobot/pi05_base"
    output_repo_id: Optional[str] = None

    # Training
    batch_size: int = 32
    steps: int = 3000
    learning_rate: float = 1e-5

    # Hardware
    gpu_type: str = "H100"
    num_gpus: int = 1
    timeout_hours: int = 2

    # Checkpointing
    save_freq: int = 500
    resume_from: Optional[str] = None
```

### 5.2 Modal Training Script

Create `xlerobot_cloud/providers/modal_provider.py`:

```python
import modal

app = modal.App("xlerobot-training")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "transformers",
        "accelerate",
        "wandb",
        "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
    )
)

@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-token")],
    volumes={"/outputs": modal.Volume.from_name("xlerobot-outputs", create_if_missing=True)}
)
def train(
    dataset_repo_id: str,
    output_repo_id: str,
    steps: int = 3000,
    batch_size: int = 32,
):
    import subprocess
    import os

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    cmd = [
        "python", "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={dataset_repo_id}",
        "--policy.type=pi05",
        "--policy.pretrained_path=lerobot/pi05_base",
        "--output_dir=/outputs/model",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=true",
        "--save_freq=500",  # Checkpoint every 500 steps
    ]

    result = subprocess.run(cmd, check=True)

    # Push to HuggingFace
    subprocess.run([
        "huggingface-cli", "upload",
        output_repo_id,
        "/outputs/model",
        "--repo-type=model"
    ], check=True)

    return {"status": "completed", "repo_id": output_repo_id}

@app.local_entrypoint()
def main(
    dataset: str,
    output: str,
    steps: int = 3000,
    batch_size: int = 32,
):
    result = train.remote(dataset, output, steps, batch_size)
    print(f"Training complete: {result}")
```

### 5.3 Run Training

```bash
# Single GPU
modal run xlerobot_cloud/providers/modal_provider.py \
  --dataset=${HF_USER}/xlerobot_task \
  --output=${HF_USER}/pi05_xlerobot \
  --steps=3000

# Multi-GPU (modify modal script to use gpu="H100:2")
modal run xlerobot_cloud/providers/modal_provider.py \
  --dataset=${HF_USER}/xlerobot_task \
  --output=${HF_USER}/pi05_xlerobot \
  --steps=1500 \  # Halved for 2x GPUs
  --batch_size=32  # Effective: 64
```

### 5.4 Local Multi-GPU Training (Alternative)

```bash
# If you have local GPUs
accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  --mixed_precision=bf16 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/xlerobot_task \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.repo_id=${HF_USER}/pi05_xlerobot \
  --output_dir=outputs/pi05_xlerobot \
  --batch_size=16 \
  --steps=1500 \
  --policy.gradient_checkpointing=true
```

### 5.5 Training Parameters

| Parameter | Single H100 | 2× A100 40GB | Notes |
|-----------|-------------|--------------|-------|
| `batch_size` | 32 | 16 (eff: 32) | Reduce if OOM |
| `steps` | 3000 | 1500 | Scale with GPUs |
| `learning_rate` | 1e-5 | 1e-5 | Don't scale |
| `save_freq` | 500 | 500 | Checkpoint frequency |

### 5.6 Error Handling & Resume

```bash
# Resume from checkpoint
modal run xlerobot_cloud/providers/modal_provider.py \
  --dataset=${HF_USER}/xlerobot_task \
  --output=${HF_USER}/pi05_xlerobot \
  --resume=/outputs/model/checkpoint-1500
```

---

## Phase 6: Validation & Evaluation

### 6.1 Simulation Evaluation (Quick)

```bash
# Test on LIBERO benchmark
python -m lerobot.scripts.eval \
  --policy.type=pi05 \
  --policy.pretrained_path=${HF_USER}/pi05_xlerobot \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=10
```

### 6.2 Real Robot Smoke Test

```bash
# Quick test with 5 episodes
lerobot-record \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --policy.type=pi05 \
  --policy.pretrained_path=${HF_USER}/pi05_xlerobot \
  --dataset.repo_id=${HF_USER}/pi05_eval \
  --dataset.num_episodes=5 \
  --dataset.single_task="Fold the shirt"
```

### 6.3 Success Metrics

- [ ] **Simulation success rate > 80%** on similar tasks
- [ ] **Real robot completes 3/5 test episodes**
- [ ] **No erratic movements** during execution
- [ ] **Inference latency < 100ms** per action chunk

---

## Phase 7: Deployment

### 7.1 Local GPU Deployment

```bash
# If you have local GPU (RTX 3090+, 24GB+ VRAM)
lerobot-record \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --policy.type=pi05 \
  --policy.pretrained_path=${HF_USER}/pi05_xlerobot \
  --policy.device=cuda
```

### 7.2 Remote GPU with Async Inference (Recommended)

π0.5 is ~3B params, async inference eliminates latency.

**On GPU Server (Modal/RunPod):**

```bash
# Start policy server
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080
```

**On Robot (XLeRobot):**

```bash
python -m lerobot.async_inference.robot_client \
  --server_address=GPU_SERVER_IP:8080 \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --robot.cameras='{...}' \
  --policy_type=pi05 \
  --pretrained_name_or_path=${HF_USER}/pi05_xlerobot \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --task="Fold the shirt"
```

### 7.3 Modal Inference Server

```python
# xlerobot_cloud/providers/modal_inference.py
import modal

app = modal.App("xlerobot-inference")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "lerobot[pi,async]@git+https://github.com/huggingface/lerobot.git"
)

@app.function(
    image=image,
    gpu="A100",
    keep_warm=1,  # Keep 1 instance warm for low latency
    secrets=[modal.Secret.from_name("huggingface-token")]
)
@modal.web_server(port=8080)
def serve():
    import subprocess
    subprocess.run([
        "python", "-m", "lerobot.async_inference.policy_server",
        "--host=0.0.0.0",
        "--port=8080"
    ])
```

---

## Rollback Plan

### If Training Fails

1. **Check logs** for OOM or NaN loss
2. **Reduce batch_size** (32 → 16 → 8)
3. **Resume from checkpoint** with `--resume`
4. **Try LoRA** instead of full fine-tune

### If Deployment Fails

1. **Test with ACT first** (lighter model)
2. **Check async latency** with `--debug_visualize_queue_size=true`
3. **Reduce actions_per_chunk** (50 → 25)
4. **Fallback to teleoperation** for demos

### Quick Rollback Commands

```bash
# Revert to base model
--policy.pretrained_path=lerobot/pi05_base

# Use smaller model
--policy.type=act

# Disable cloud, run local
python examples/5_xlerobot_teleop_xbox.py
```

---

## Cost Estimate

| Phase | Resource | Cost |
|-------|----------|------|
| Training (H100, 2hr) | Modal | ~$8 |
| Inference (A100, 1hr) | Modal | ~$2 |
| HuggingFace storage | Hub | Free |
| **Total** | | **~$10-15** |

---

## Checkpoints

### Phase 1: Setup
- [ ] lerobot 0.4.2+ installed
- [ ] `pip install -e ".[pi,async]"` complete
- [ ] HuggingFace login working
- [ ] Modal account configured

### Phase 2: Hardware
- [ ] All 3 cameras detected
- [ ] Camera config saved
- [ ] Robot motors responding

### Phase 3: Data Collection
- [ ] 50+ episodes recorded
- [ ] Dataset pushed to HuggingFace
- [ ] Video playback verified

### Phase 4: Data Prep
- [ ] Quantile stats added
- [ ] (Optional) Augmentation complete

### Phase 5: Training
- [ ] Training launched
- [ ] Checkpoints saving
- [ ] Training complete
- [ ] Model pushed to HuggingFace

### Phase 6: Validation
- [ ] Simulation eval > 80%
- [ ] Real robot test passed

### Phase 7: Deployment
- [ ] Inference server running
- [ ] Robot executing policy
- [ ] Latency acceptable

---

## References

- [π0.5 Docs](https://huggingface.co/docs/lerobot/en/pi05) - Official LeRobot π0.5 documentation
- [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) - Base model checkpoint
- [OpenPI Repo](https://github.com/Physical-Intelligence/openpi) - Physical Intelligence source
- [LeRobot Async Inference](https://huggingface.co/docs/lerobot/en/async) - Remote GPU deployment
- [LeRobot Multi-GPU](https://huggingface.co/docs/lerobot/multi_gpu_training) - Distributed training
- [XLeRobot VLA Guide](https://xlerobot.readthedocs.io/en/latest/software/getting_started/RL_VLA.html) - XLeRobot-specific docs
- [Modal GPU Docs](https://modal.com/docs/guide/gpu) - Cloud training setup
