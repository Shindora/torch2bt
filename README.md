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

## Examples

See [`examples/`](examples/) for full runnable scripts:

| Script | Subnet | Description |
|--------|--------|-------------|
| [`sn1_text_prompting.py`](examples/sn1_text_prompting.py) | SN1 | Transformer LM → Text Prompting miner |
| [`sn18_image_generation.py`](examples/sn18_image_generation.py) | SN18 | Diffusion model → Cortex image miner |

## TODO

### Phase A — Alpha (current)

- [x] `inspector.py` — extract model `forward()` signature via reflection
- [x] `codegen.py` — generate `protocol.py`, `miner.py`, `Dockerfile`, `pyproject.toml` using Python 3.14 t-strings
- [x] `subnets/` — protocol registry for SN1 (Text Prompting) and SN18 (Cortex)
- [x] `testing/` — `MockValidator` + `MockSynapse` for offline miner testing
- [x] `t2b.package()` — end-to-end packaging API
- [x] CI — ruff lint/format, zuban type check, pytest
- [x] PyPI metadata — version `0.1.0a1`, classifiers, license, URLs
- [ ] Publish `0.1.0a1` to PyPI

### Phase B — Beta

- [ ] `t2b.deploy(platform="runpod")` — provision GPU instance via RunPod API
- [ ] `t2b.deploy(platform="lambda")` — Lambda Labs GPU support
- [ ] Auto-register hotkey with `btcli` post-deploy
- [ ] Dynamic TAO (dTAO / BIT001) profitability dashboard integration
- [ ] Auto-quantization — convert FP32 models to INT4/INT8 on the fly with bitsandbytes
- [ ] `uv.lock` generation for deterministic miner environments

### Phase C — Production

- [ ] Multi-subnet mining — host multiple models on a single Axon
- [ ] Self-healing miners — auto-restart on OOM or network failure
- [ ] Expand subnet registry beyond SN1 + SN18
- [ ] `t2b.benchmark()` — measure model latency vs subnet timeout requirements
