# π0.5 Training Troubleshooting Log

## Issue Summary
Training π0.5 on Modal with lerobot requires careful dependency management.

## What We've Tried

### Attempt 1: Standard lerobot[pi] install
- **Command**: `pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"`
- **Result**: FAILED - "An incorrect transformer version is used"
- **Cause**: The check.py file in transformers was missing

### Attempt 2: Install patched transformers BEFORE lerobot
- **Command**: Install transformers from fix/lerobot_openpi branch first, then lerobot
- **Result**: FAILED - lerobot[pi] overwrites transformers with standard version

### Attempt 3: Install patched transformers AFTER lerobot with --force-reinstall
- **Command**: `pip install --force-reinstall git+...transformers...@fix/lerobot_openpi`
- **Result**: FAILED - check.py expects version 4.53.x but branch is 4.57.x

### Attempt 4: Install transformers 4.53.3 from PyPI + add check.py manually
- **Command**: `pip install transformers==4.53.3` + create check.py
- **Result**: FAILED - 403 Gated repo error for google/paligemma-3b-pt-224

### Attempt 5: After accepting PaliGemma license
- **Result**: FAILED - bfloat16 dtype mismatch with SigLIP
- **Error**: "expected scalar type Float but found BFloat16"

### Attempt 6: Use float32 instead of bfloat16
- **Result**: FAILED - GemmaRMSNorm.forward() got unexpected keyword argument 'cond'
- **Cause**: PyPI transformers 4.53.3 doesn't have pi05-specific modifications

### Attempt 7: Real patched transformers + bypass version check
- **Command**: Install from fix/lerobot_openpi + update check.py to always return True
- **Result**: FAILED - Same GemmaRMSNorm error
- **Error**: `TypeError: GemmaRMSNorm.forward() got an unexpected keyword argument 'cond'`
- **Cause**: The fix/lerobot_openpi branch doesn't contain the modified GemmaRMSNorm that lerobot expects

### Attempt 8: Apply OpenPI transformers_replace patches - SUCCESS!
- **What we did**:
  1. Install transformers 4.53.3
  2. Clone OpenPI repo
  3. Copy `src/openpi/models_pytorch/transformers_replace/models/*` to transformers
  4. Add check.py that returns True
- **Result**: SUCCESS! Training started with loss decreasing:
  - Step 200: 0.291
  - Step 400: 0.210
  - Step 600: 0.155
  - Step 800: 0.140

## WORKING CONFIGURATION

```python
# In Modal image definition:
.pip_install("lerobot[pi]@git+https://github.com/huggingface/lerobot.git@v0.4.2")
.run_commands("pip install --force-reinstall transformers==4.53.3")
.run_commands(
    "git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git /tmp/openpi && "
    "cp -r /tmp/openpi/src/openpi/models_pytorch/transformers_replace/models/* /usr/local/lib/python3.10/site-packages/transformers/models/ && "
    "rm -rf /tmp/openpi"
)
# Add check.py that returns True
```

## Key Insights

1. **PaliGemma Access Required**: Must accept license at https://huggingface.co/google/paligemma-3b-pt-224

2. **Transformers Version Incompatibility** (GitHub Issue #1406):
   - transformers >= 4.52.0 has key mismatches with pi05 checkpoints
   - The fix/lerobot_openpi branch has modified GemmaRMSNorm with `cond` argument
   - Standard PyPI transformers doesn't have these modifications

3. **check.py Version Check**:
   - Located at `transformers/models/siglip/check.py`
   - Expects version 4.53.2 or 4.53.3
   - But the branch itself is now at 4.57.x

## Training Completed Successfully!
- **Total steps**: 5000
- **Final loss**: ~0.112
- **Model size**: 7GB (7,473,096,344 bytes)
- **Checkpoints saved**: 1000, 2000, 3000, 4000, 5000, last

## Local Inference Setup

### Step 1: Install OpenPI Patches Locally
```bash
# Install transformers 4.53.3
pip install transformers==4.53.3

# Clone OpenPI and apply patches
cd /tmp && git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git

# Copy patches to transformers installation
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
cp -r /tmp/openpi/src/openpi/models_pytorch/transformers_replace/models/* $TRANSFORMERS_PATH/models/

# Create check.py
cat > $TRANSFORMERS_PATH/models/siglip/check.py << 'EOF'
import transformers
def check_whether_transformers_replace_is_installed_correctly():
    return True  # Patched from OpenPI
EOF

# Verify patches
python -c "from transformers.models.gemma.modeling_gemma import GemmaRMSNorm; import inspect; sig = inspect.signature(GemmaRMSNorm.forward); assert 'cond' in sig.parameters; print('OK')"
```

### Step 2: Download Model from Modal
```bash
# Download π0.5 model (7GB)
modal volume get xlerobot-data "outputs/run_a8b4daaf/checkpoints/005000/pretrained_model" ./trained_model_pi05 --force
```

### Step 3: Inference Requirements
π0.5 inference requires:
1. **PolicyProcessorPipeline**: Load preprocessor and postprocessor from model directory
2. **Task description**: VLM requires text prompt (e.g., "Pick up the towel")
3. **bfloat16 dtype**: Use `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
4. **Image preprocessing**: Original camera size (480x640) → preprocessor resizes to 224x224

### Inference Example
```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline

# Load policy, preprocessor, postprocessor
policy = PI05Policy.from_pretrained(MODEL_PATH)
policy.to("cuda", dtype=torch.bfloat16)
preprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_PATH, config_filename="policy_preprocessor.json")
postprocessor = PolicyProcessorPipeline.from_pretrained(MODEL_PATH, config_filename="policy_postprocessor.json")

# Create observation with task description
observation = {
    "observation.images.head": image_tensor,  # [3, H, W]
    "observation.state": state_tensor,         # [6]
    "task": "Pick up the towel",
}

# Process and infer
batch = preprocessor(observation)
for k in batch:
    if isinstance(batch[k], torch.Tensor) and batch[k].is_floating_point():
        batch[k] = batch[k].to(torch.bfloat16)

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    action = policy.select_action(batch)

action = postprocessor({"action": action})["action"]
```

## References
- https://github.com/huggingface/lerobot/issues/2305
- https://github.com/huggingface/lerobot/issues/1406
- https://github.com/huggingface/lerobot/issues/2307
- https://huggingface.co/docs/lerobot/en/pi05
