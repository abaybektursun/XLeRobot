# XLeRobot π0.5 Training Pipeline

Complete guide for training a π0.5 (Pi Zero Five) vision-language-action model on XLeRobot.

## Repository Structure

### GitHub Repositories

```
abaybektursun/XLeRobot          (this repo - robot project)
    └── software/               (π0.5 training pipeline)

abaybektursun/lerobot           (fork of huggingface/lerobot)
    └── feature/xlerobot        (branch with XLeRobot driver)
```

### XLeRobot Software Directory

```
software/
├── modal_train.py              # Cloud training on Modal (H100 GPU)
├── run_policy.py               # Run trained policy on robot
├── pi05train.md                # This documentation
├── PI05_TROUBLESHOOTING.md     # Troubleshooting log
├── camera_config.json          # Camera device mappings
├── perm.sh                     # USB permissions script
│
├── examples/
│   ├── 9_kinesthetic_teaching.py    # Data collection by demonstration
│   ├── scan_motors.py               # Motor diagnostics
│   ├── simple_xlerobot_test.py      # Quick robot test
│   └── ...                          # Other teleop examples
│
├── src/
│   ├── robots/
│   │   ├── xlerobot/                # XLeRobot driver (→ lerobot fork)
│   │   └── xlerobot_2wheels/        # 2-wheel variant
│   ├── teleporators/
│   │   └── xlerobot_vr/             # VR teleoperator (→ lerobot fork)
│   └── record.py                    # Custom recording script
│
├── dataset_merged_104ep/       # Training dataset (104 episodes)
├── trained_model_pi05/         # Downloaded trained model (7GB)
└── trained_model_104ep/        # Alternative checkpoint
```

### LeRobot Fork (abaybektursun/lerobot)

Branch: `feature/xlerobot`

```
src/lerobot/
├── robots/
│   └── xlerobot/               # XLeRobot robot driver
│       ├── __init__.py
│       ├── config_xlerobot.py  # Robot configuration
│       ├── xlerobot.py         # Main driver (648 lines)
│       ├── xlerobot_client.py  # Remote client
│       └── xlerobot_host.py    # Remote host
│
└── teleoperators/
    └── xlerobot_vr/            # VR teleoperator
        ├── __init__.py
        ├── configuration_xlerobot_vr.py
        ├── xlerobot_vr.py
        └── vr_monitor.py
```

---

## Hardware Configuration

### XLeRobot Components

| Component | Motors | IDs | Bus |
|-----------|--------|-----|-----|
| Left Arm | 6x STS3215 | 1-6 | /dev/ttyACM0 |
| Head | 2x STS3215 | 7-8 | /dev/ttyACM0 |
| Right Arm | 6x STS3215 | 1-6 | /dev/ttyACM1 |
| Base Wheels | 3x STS3215 | 7-9 | /dev/ttyACM1 (optional) |

### Cameras

| Camera | Device | Resolution | FPS |
|--------|--------|------------|-----|
| Head | /dev/video6 | 640x480 | 30 |
| Left Wrist | /dev/video4 | 640x480 | 30 |
| Right Wrist | /dev/video2 | 640x480 | 30 |

### Motor Names (Right Arm - Used in Training)

```python
RIGHT_ARM_MOTORS = [
    "right_arm_shoulder_pan",
    "right_arm_shoulder_lift",
    "right_arm_elbow_flex",
    "right_arm_wrist_flex",
    "right_arm_wrist_roll",
    "right_arm_gripper",
]
```

---

## Installation

### 1. Install LeRobot with XLeRobot Support

```bash
# Create conda environment
conda create -n lerobot python=3.10
conda activate lerobot

# Install from fork (includes xlerobot driver)
pip install "lerobot[pi] @ git+https://github.com/abaybektursun/lerobot.git@feature/xlerobot"
```

### 2. Apply OpenPI Patches (Required for π0.5)

π0.5 requires patched transformers with `GemmaRMSNorm.cond` argument:

```bash
# Install correct transformers version
pip install transformers==4.53.3

# Clone OpenPI and apply patches
cd /tmp && git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git

# Copy patches to transformers
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
cp -r /tmp/openpi/src/openpi/models_pytorch/transformers_replace/models/* $TRANSFORMERS_PATH/models/

# Create check.py
cat > $TRANSFORMERS_PATH/models/siglip/check.py << 'EOF'
import transformers
def check_whether_transformers_replace_is_installed_correctly():
    return True  # Patched from OpenPI
EOF

# Verify patches
python -c "from transformers.models.gemma.modeling_gemma import GemmaRMSNorm; \
           import inspect; sig = inspect.signature(GemmaRMSNorm.forward); \
           assert 'cond' in sig.parameters; print('Patches OK')"

# Cleanup
rm -rf /tmp/openpi
```

### 3. USB Permissions

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
# Or run: ./perm.sh
```

### 4. HuggingFace Access

```bash
# Login to HuggingFace (required for π0.5 base model)
huggingface-cli login

# Accept PaliGemma license at:
# https://huggingface.co/google/paligemma-3b-pt-224
```

---

## Data Collection

### Kinesthetic Teaching (Recommended)

```bash
cd software
python examples/9_kinesthetic_teaching.py
```

**Controls:**
- `WASD` - Position head camera
- `SPACE` - Start/stop recording
- `ENTER` - Save episode
- `R` - Reset (discard)
- `Q` - Quit

**Output:** LeRobot dataset in `dataset_TIMESTAMP/`

### Dataset Format

```
dataset_merged_104ep/
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # Joint positions + metadata
├── videos/
│   └── observation.images.head/
│       └── chunk-000/
│           └── file-000.mp4    # Camera frames
└── meta/
    ├── info.json               # Dataset metadata
    ├── episodes.jsonl          # Episode index
    └── tasks.jsonl             # Task descriptions
```

**Features:**
```json
{
  "action": {"shape": [6], "dtype": "float32"},
  "observation.state": {"shape": [6], "dtype": "float32"},
  "observation.images.head": {"shape": [480, 640, 3], "dtype": "video"}
}
```

---

## Training on Modal

### Prerequisites

```bash
# Install Modal
pip install modal

# Setup Modal account
modal setup

# Create HuggingFace secret in Modal dashboard
# Name: huggingface-token
# Key: HF_TOKEN = your_token
```

### Training Commands

```bash
cd software

# 1. Upload dataset to Modal volume
modal run modal_train.py --upload-dataset-flag

# 2. Check upload status
modal run modal_train.py --check

# 3. Run training (5000 steps on H100)
modal run modal_train.py --train-flag --steps 5000 --batch-size 2

# 4. Download trained model
modal run modal_train.py --download-model
```

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Policy | `pi05` | NOT ACT - π0.5 is foundation model |
| Base Model | `lerobot/pi05_base` | 7GB pretrained |
| Steps | 5000 | Fewer than ACT (foundation model) |
| Batch Size | 2 | Limited by GPU memory |
| Learning Rate | 2e-5 | Fine-tuning rate |
| Normalization | MEAN_STD | Dataset lacks quantiles |
| dtype | bfloat16 | Required for π0.5 |
| GPU | H100 | Or A100 (slower) |

### Resume Training

```bash
# List runs
modal run modal_train.py --check

# Resume from specific run
modal run modal_train.py --train-flag --resume-run-id run_XXXXXXXX --steps 10000
```

---

## Inference

### Run Trained Policy

```bash
cd software

# Dry run (test without robot)
python run_policy.py --dry-run

# Run on robot
python run_policy.py --model-path trained_model_pi05/pretrained_model --steps 500

# Move to training start pose first
python run_policy.py --go-to-start --steps 500
```

### Safety Limits

```python
ACTION_LIMITS = {
    "right_arm_shoulder_pan": (90, 100),
    "right_arm_shoulder_lift": (25, 45),
    "right_arm_elbow_flex": (-70, -55),
    "right_arm_wrist_flex": (85, 100),
    "right_arm_wrist_roll": (-30, 30),
    "right_arm_gripper": (0, 100),
}
```

### Inference Pipeline

```
Camera Frame (640x480)
        ↓
    Preprocess (resize to 224x224, normalize)
        ↓
    Robot State (6 joint positions)
        ↓
    Task Description ("Pick up the towel")
        ↓
┌───────────────────────────────────────┐
│  PolicyProcessorPipeline (preprocess) │
│  - Batching                           │
│  - Normalization (MEAN_STD)           │
│  - Tokenization                       │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  π0.5 Policy (7GB model)              │
│  - PaliGemma vision encoder           │
│  - Gemma 2B language model            │
│  - 300M action expert                 │
│  - bfloat16 inference                 │
└───────────────────────────────────────┘
        ↓
┌───────────────────────────────────────┐
│  PolicyProcessorPipeline (postprocess)│
│  - Unnormalize actions                │
└───────────────────────────────────────┘
        ↓
    Safety Clamp (per-joint limits)
        ↓
    Robot Action (6 joint positions)
```

---

## Model Architecture

### π0.5 (Pi Zero Five)

A Vision-Language-Action (VLA) foundation model:

| Component | Model | Size |
|-----------|-------|------|
| Vision | SigLIP (from PaliGemma) | - |
| Language | Gemma 2B | 2B params |
| Action Expert | Gemma 300M | 300M params |
| **Total** | - | **~7GB** |

**Key Features:**
- Works with 10-50 demonstrations (vs 100s for ACT)
- Language-conditioned (task description as prompt)
- Diffusion-based action generation
- Chunk size: 50 actions per inference

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `GemmaRMSNorm.forward() got unexpected keyword argument 'cond'` | Apply OpenPI patches (see Installation) |
| `403 Gated repo error for google/paligemma` | Accept license at HuggingFace |
| `expected scalar type Float but found BFloat16` | Use `torch.autocast(dtype=torch.bfloat16)` |
| USB permission denied | Run `sudo chmod 666 /dev/ttyACM*` |
| Motor not detected | Check power (12V 8A) and connections |

### Verify Installation

```bash
# Check transformers patches
python -c "from transformers.models.gemma.modeling_gemma import GemmaRMSNorm; \
           import inspect; print('cond' in inspect.signature(GemmaRMSNorm.forward).parameters)"

# Check xlerobot driver
python -c "from lerobot.robots.xlerobot import XLerobot; print('OK')"

# Check π0.5 policy
python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy; print('OK')"
```

---

## Training Results

### Towel Pickup Task

| Metric | Value |
|--------|-------|
| Episodes | 104 |
| Total Frames | 16,095 |
| Training Steps | 5,000 |
| Final Loss | ~0.112 |
| Training Time | ~2 hours (H100) |
| Model Size | 7.47 GB |

### Loss Progression

```
Step  200: 0.291
Step  400: 0.210
Step  600: 0.155
Step  800: 0.140
Step 1000: 0.130
...
Step 5000: 0.112
```

---

## Remote Deployment (Raspberry Pi + GPU Laptop)

π0.5 is a 7GB model requiring GPU inference. The only practical deployment uses:
- **Raspberry Pi**: Robot control (motors, cameras)
- **Laptop/Desktop with GPU**: Policy inference

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Raspberry Pi (Robot)                        │
│                                                                 │
│  xlerobot_host.py                                               │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Motor Control  │    │  Camera Capture │                    │
│  │  (STS3215 bus)  │    │  (OpenCV)       │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│              ┌───────────────┐                                  │
│              │  ZMQ Server   │                                  │
│              │  Port 5555    │◄─── Commands (actions)           │
│              │  Port 5556    │───► Observations (state+images)  │
│              └───────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
                           │
                      WiFi/Ethernet
                           │
┌─────────────────────────────────────────────────────────────────┐
│                  Laptop with GPU                                │
│                                                                 │
│  run_policy_remote.py                                           │
│  ┌───────────────┐      ┌─────────────────────────────────┐    │
│  │  ZMQ Client   │      │  π0.5 Policy (7GB)              │    │
│  │  xlerobot_    │◄────►│  - Preprocessor                 │    │
│  │  client.py    │      │  - PI05Policy.select_action()   │    │
│  └───────────────┘      │  - Postprocessor                │    │
│                         └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Raspberry Pi Setup

```bash
# On Raspberry Pi
# 1. Install lerobot (CPU only, no [pi] extras)
pip install "lerobot @ git+https://github.com/abaybektursun/lerobot.git@feature/xlerobot"

# 2. Set USB permissions
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1

# 3. Run host (controls robot, streams observations)
python -m lerobot.robots.xlerobot.xlerobot_host
```

### GPU Laptop Setup

```bash
# On laptop with GPU
# 1. Install lerobot with π0.5 support
pip install "lerobot[pi] @ git+https://github.com/abaybektursun/lerobot.git@feature/xlerobot"

# 2. Apply OpenPI patches (see Installation section)

# 3. Run policy client
python run_policy_remote.py --remote-ip <PI_IP_ADDRESS>
```

### Configuration

Edit `config_xlerobot.py` for remote settings:

```python
@dataclass
class XLerobotHostConfig:
    port_zmq_cmd: int = 5555           # Receive actions
    port_zmq_observations: int = 5556  # Send observations
    connection_time_s: float = 3600    # Session duration
    watchdog_timeout_ms: int = 1000    # Stop if no commands
    max_loop_freq_hz: float = 30       # Control loop rate

@dataclass
class XLerobotClientConfig:
    remote_ip: str = "192.168.1.100"   # Pi IP address
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    polling_timeout_ms: int = 100
    connect_timeout_s: float = 10
```

### Network Requirements

| Parameter | Requirement |
|-----------|-------------|
| Latency | < 50ms recommended |
| Bandwidth | ~5 Mbps (JPEG-compressed images) |
| Protocol | ZMQ over TCP |
| Ports | 5555 (commands), 5556 (observations) |

### Data Flow

1. **Pi → Laptop** (Observations):
   - Joint positions (14 floats)
   - Camera frames (JPEG base64 encoded)
   - Velocity state

2. **Laptop → Pi** (Actions):
   - Target joint positions (JSON)
   - Velocity commands (if using base)

---

## References

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [π0.5 Documentation](https://huggingface.co/docs/lerobot/en/pi05)
- [OpenPI Repository](https://github.com/Physical-Intelligence/openpi)
- [XLeRobot Hardware Docs](https://xlerobot.readthedocs.io)

### Related Issues

- [lerobot#2305](https://github.com/huggingface/lerobot/issues/2305) - transformers compatibility
- [lerobot#1406](https://github.com/huggingface/lerobot/issues/1406) - π0.5 key mismatches
- [lerobot#2307](https://github.com/huggingface/lerobot/issues/2307) - training issues
