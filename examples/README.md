# Examples

## SN1 — Text Prompting

Package a transformer language model as a miner on Subnet 1.

```bash
uv run python examples/sn1_text_prompting.py
```

```python
import torch
import torch.nn as nn
import torch2bt as t2b
from torch2bt.testing import MockValidator, MockSynapse

class TinyLM(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

t2b.package(TinyLM(), target_subnet=1, optimization="fp16", wallet_name="my_wallet")

# Local test — no live network needed
async def my_forward(synapse: MockSynapse) -> MockSynapse:
    synapse.completion = "Paris"
    return synapse

validator = MockValidator("Prompting", subnet_id=1, forward_fn=my_forward)
result = await validator.query({"roles": ["user"], "messages": ["Capital of France?"]})
print(result.completion)  # Paris
```

## SN18 — Image Generation (Cortex)

Package a diffusion model as a miner on Subnet 18.

```bash
uv run python examples/sn18_image_generation.py
```

```python
import torch
import torch.nn as nn
import torch2bt as t2b
from torch2bt.testing import MockValidator, MockSynapse

class TinyDiffusion(nn.Module):
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        ...

# SN18 only supports fp16 / bf16
t2b.package(TinyDiffusion(), target_subnet=18, optimization="fp16", wallet_name="my_wallet")

async def my_forward(synapse: MockSynapse) -> MockSynapse:
    synapse.image_data = [0.0] * (64 * 64 * 3)
    synapse.image_shape = (64, 64, 3)
    return synapse

validator = MockValidator("ImageResponse", subnet_id=18, forward_fn=my_forward)
result = await validator.query({
    "prompt": "a red cat",
    "seed": 42,
    "width": 64,
    "height": 64,
    "num_inference_steps": 20,
})
print(result.image_shape)  # (64, 64, 3)
```
