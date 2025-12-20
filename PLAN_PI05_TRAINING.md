# XLeRobot π0.5 VLA Training Plan

## Status: π0.6 NOT YET AVAILABLE
As of Dec 2025, [π0.6 is not open-sourced](https://github.com/Physical-Intelligence/openpi/issues/791).
We will use **π0.5** (released Sept 2025) which has better open-world generalization than π0.

---

## Overview

| Phase | Task | Time Estimate |
|-------|------|---------------|
| 1 | Environment Setup | 1-2 hours |
| 2 | Data Collection (50-100 episodes) | 2-4 hours |
| 3 | Data Augmentation | 30 min |
| 4 | Cloud Training | 1-2 hours |
| 5 | Deployment & Testing | 1 hour |

---

## Phase 1: Environment Setup

### 1.1 Local Machine (Data Collection)

```bash
# Activate lerobot environment
conda activate lerobot

# Install π0 dependencies
pip install -e ".[pi]"

# Verify xlerobot connection
python examples/simple_xlerobot_test.py
```

### 1.2 Cloud Setup (Training)

**Recommended: [Modal](https://modal.com)** - Simple, mature, pay-per-use
- H100 80GB: ~$4/hour
- A100 80GB: ~$2/hour

**Alternatives:**
- [RunPod](https://runpod.io) - A100 from $1.89/hr, good availability
- [Lambda Labs](https://lambdalabs.com) - H100/H200, enterprise-focused

**GPU Requirements:**
| Method | VRAM Required |
|--------|---------------|
| LoRA fine-tune | 22.5+ GB (A100 40GB works) |
| Full fine-tune | 70+ GB (H100 80GB recommended) |

### 1.3 Modal Setup

```bash
pip install modal
modal setup  # Authenticate

# Create modal app for training
modal run train_pi05.py
```

---

## Phase 2: Data Collection

### 2.1 Camera Setup

XLeRobot uses 3 cameras:
- Left wrist camera
- Right wrist camera
- Head camera

```bash
# Verify cameras
python -c "
from lerobot.cameras.opencv import OpenCVCamera
for i in range(10):
    try:
        cam = OpenCVCamera(i)
        cam.connect()
        print(f'Camera {i}: OK')
        cam.disconnect()
    except: pass
"
```

### 2.2 Recording with VR Teleoperation (Recommended)

Per [XLeRobot docs](https://xlerobot.readthedocs.io/en/latest/software/getting_started/RL_VLA.html):

```bash
# Start VR server first (on Quest 3)
# Then record:

python -m lerobot.record \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --robot.id=my_xlerobot \
  --robot.cameras='{
    left_wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480},
    right_wrist: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480},
    head: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480}
  }' \
  --teleop.type=xlerobot_vr \
  --dataset.repo_id=${HF_USER}/xlerobot_laundry \
  --dataset.num_episodes=50 \
  --dataset.single_task="Fold the shirt"
```

### 2.3 Recording with Leader Arms (Alternative)

If no VR, use dual leader arms:

```bash
python -m lerobot.record \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --teleop.type=xlerobot_leader \
  --teleop.left_port=/dev/ttyACM2 \
  --teleop.right_port=/dev/ttyACM3 \
  --dataset.repo_id=${HF_USER}/xlerobot_laundry \
  --dataset.num_episodes=50 \
  --dataset.single_task="Fold the shirt"
```

### 2.4 Data Quality Tips

1. **Keep head camera angle consistent** - Policy performance drops otherwise
2. **Record 50-100 episodes** - More is better, but 50 can work
3. **Vary starting positions slightly** - Improves generalization
4. **Include failure recovery** - Helps policy learn corrections

---

## Phase 3: Data Augmentation (4x Data)

```bash
python lerobot/scripts/augment_dataset.py \
  --dataset.repo_id=${HF_USER}/xlerobot_laundry \
  --output.repo_id=${HF_USER}/xlerobot_laundry_augmented
```

This flips footage and reverses action polarities, effectively 4x your data.

---

## Phase 4: Cloud Training with π0.5

### 4.0 Cloud Abstraction Architecture

> **Note:** LeRobot has [async inference](https://huggingface.co/docs/lerobot/en/async) for remote GPU deployment,
> but **no built-in cloud training** ([issue #2172](https://github.com/huggingface/lerobot/issues/2172)).
> We build our own training abstraction here.

We use a **provider-agnostic abstraction** so training can run on any cloud:

```
xlerobot_cloud/
├── __init__.py
├── base.py              # Abstract CloudProvider interface
├── config.py            # TrainingConfig dataclass
├── providers/
│   ├── __init__.py
│   ├── modal_provider.py    # Modal implementation (default)
│   ├── runpod_provider.py   # RunPod (future)
│   └── lambda_provider.py   # Lambda Labs (future)
└── cli.py               # Unified CLI: xlerobot-train --provider=modal
```

**Abstract Interface (`base.py`):**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    dataset_repo_id: str
    output_repo_id: str
    policy_type: str = "pi0"
    pretrained_path: str = "lerobot/pi05_base"
    batch_size: int = 32
    steps: int = 3000
    gpu_type: str = "H100"
    timeout_hours: int = 2

class CloudProvider(ABC):
    @abstractmethod
    def setup(self) -> None:
        """Authenticate and configure provider."""
        pass

    @abstractmethod
    def train(self, config: TrainingConfig) -> str:
        """Launch training job. Returns job ID."""
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> dict:
        """Get job status."""
        pass

    @abstractmethod
    def download_model(self, job_id: str, output_path: str) -> None:
        """Download trained model."""
        pass
```

**Usage:**

```bash
# Train with Modal (default)
xlerobot-train --provider=modal --dataset=user/xlerobot_laundry

# Train with RunPod (future)
xlerobot-train --provider=runpod --dataset=user/xlerobot_laundry
```

---

### 4.1 Modal Provider Implementation

Create `xlerobot_cloud/providers/modal_provider.py`:

```python
import modal
from ..base import CloudProvider, TrainingConfig

class ModalProvider(CloudProvider):
    def __init__(self):
        self.app = modal.App("xlerobot-training")
        self.image = (
            modal.Image.debian_slim(python_version="3.10")
            .pip_install("torch", "transformers", "accelerate")
            .pip_install("lerobot[pi]@git+https://github.com/huggingface/lerobot.git")
        )

    def setup(self) -> None:
        # modal setup is done via CLI: `modal setup`
        pass

    def train(self, config: TrainingConfig) -> str:
        @self.app.function(
            image=self.image,
            gpu=config.gpu_type,
            timeout=config.timeout_hours * 3600,
            secrets=[modal.Secret.from_name("huggingface-token")]
        )
        def _train():
            import subprocess
            subprocess.run([
                "python", "-m", "lerobot.train",
                f"--policy.type={config.policy_type}",
                f"--policy.pretrained_path={config.pretrained_path}",
                f"--dataset.repo_id={config.dataset_repo_id}",
                "--output_dir=/outputs/model",
                f"--batch_size={config.batch_size}",
                f"--steps={config.steps}",
                "--policy.dtype=bfloat16",
                "--policy.gradient_checkpointing=true",
            ], check=True)

            # Push to HuggingFace
            subprocess.run([
                "huggingface-cli", "upload",
                config.output_repo_id,
                "/outputs/model"
            ], check=True)

        with self.app.run():
            _train.remote()
        return "modal-job"  # Modal doesn't expose job IDs easily

    def get_status(self, job_id: str) -> dict:
        # Modal runs synchronously in .remote()
        return {"status": "completed"}

    def download_model(self, job_id: str, output_path: str) -> None:
        # Model is pushed to HuggingFace, download from there
        pass
```

### 4.2 Run Training

```bash
# First time setup
pip install modal
modal setup
modal secret create huggingface-token HF_TOKEN=hf_xxxxx

# Run training via abstraction
xlerobot-train \
  --provider=modal \
  --dataset=${HF_USER}/xlerobot_laundry_augmented \
  --output=${HF_USER}/pi05_xlerobot_policy \
  --gpu=H100 \
  --steps=3000

# Or run Modal directly (legacy)
modal run xlerobot_cloud/providers/modal_provider.py
```

### 4.3 Training Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `batch_size` | 32 | Reduce if OOM |
| `steps` | 3000-5000 | 50 episodes × 60 steps/ep |
| `learning_rate` | 1e-5 | Default is fine |
| `dtype` | bfloat16 | Saves memory |
| `gradient_checkpointing` | true | Saves memory |

---

## Phase 5: Deployment

> LeRobot has built-in [async inference](https://huggingface.co/docs/lerobot/en/async) for remote GPU deployment.

### 5.1 Local Deployment (if GPU available)

```bash
python -m lerobot.record \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --policy.path=${HF_USER}/pi05_xlerobot_policy
```

### 5.2 Remote GPU Server (Recommended for π0.5)

π0.5 is heavy (~3B params), so use **LeRobot's async inference**:

**On GPU Server (Modal/RunPod/Lambda):**
```bash
# Using lerobot's built-in policy server
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --policy.path=${HF_USER}/pi05_xlerobot_policy
```

**On Robot (XLeRobot):**
```bash
# Stream to remote GPU for inference
python -m lerobot.record \
  --robot.type=xlerobot \
  --policy.remote_server=http://GPU_SERVER_IP:8080
```

See [LeRobot async docs](https://huggingface.co/docs/lerobot/en/async#tuning-async-inference-for-your-setup) for tuning latency.

---

## Cost Estimate

| Item | Cost |
|------|------|
| Modal H100 (2 hours) | ~$8 |
| HuggingFace storage | Free |
| Total | **~$10** |

---

## Checkpoints

- [ ] lerobot + pi dependencies installed
- [ ] Cameras detected (3 cameras)
- [ ] VR or leader arms working
- [ ] 50+ episodes recorded
- [ ] Dataset uploaded to HuggingFace
- [ ] Modal account + secrets configured
- [ ] Training completed
- [ ] Policy deployed and tested

---

## References

- [π0.5 OpenPI Repo](https://github.com/Physical-Intelligence/openpi)
- [LeRobot π0 Docs](https://huggingface.co/docs/lerobot/pi0)
- [XLeRobot VLA Training](https://xlerobot.readthedocs.io/en/latest/software/getting_started/RL_VLA.html)
- [Teddy Warner's π0 Tutorial](https://teddywarner.org/Projects/pi0/)
- [Modal GPU Docs](https://modal.com/docs/guide/gpu)
