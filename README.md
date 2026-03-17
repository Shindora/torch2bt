# torch2bt

**The bridge between PyTorch research and the Bittensor decentralized intelligence network.**

Turn any `torch.nn.Module` into a revenue-generating Bittensor miner — zero boilerplate.

## How it works

1. **Inspect** — Analyzes your model's `forward()` signature via reflection
2. **Synthesize** — Generates `protocol.py`, `miner.py`, and `Dockerfile` using Python 3.14 t-strings
3. **Deploy** — Drop the output into any GPU host and start mining

## Install

```bash
uv add torch2bt
```

## Usage

```python
import torch2bt as t2b
from my_models import SuperNeuralNet

t2b.package(
    model=SuperNeuralNet(),
    target_subnet=18,
    optimization="fp16",
    wallet_name="mining_key",
)
```

Output: `torch2bt_output/protocol.py`, `miner.py`, `Dockerfile`, `pyproject.toml`

## Supported subnets

| NetUID | Name            | Optimizations     |
|--------|-----------------|-------------------|
| 1      | Text Prompting  | FP32/FP16/BF16/INT8/INT4 |
| 18     | Cortex          | FP16/BF16         |

## Local testing

```python
from torch2bt.testing import MockValidator

validator = MockValidator(MySynapse, subnet_id=18, forward_fn=my_forward)
result = validator.query({"prompt": "a red cat"})
```

## Stack

- Python 3.14+ (free-threaded)
- uv · ruff · ZubanLS

## Roadmap

- **Alpha** — `package()` for SN1 + SN18, mock validator
- **Beta** — RunPod/Lambda deploy API, auto-quantization
- **Production** — Multi-subnet mining, self-healing miners
