# rhythmic-pattern-transcription
[![ArXiv](https://img.shields.io/badge/arXiv-2510.05756-b31b1b?logo=arxiv&labelColor=gray)](https://arxiv.org/abs/2510.05756)

Companion resources for the paper 'Transcribing Rhythmic Patterns of the Guitar Track in Polyphonic Music'

## Downbeat processing
Install dependencies with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

Run the downbeat processing script:
```python
import numpy as np
from downbeat import process_downbeats

downbeats = np.sort(np.asarray([...], dtype=np.float64))
processed = process_downbeats(downbeats)
```

## Dataset excerpts
[excerpts](excerpts/) folder contains audio excerpts from the held‑out test split used in the paper, 2 excerpts per difficulty category (simplified, intermediate, advanced, original).
Each excerpt represents a 10-second clip of polyphonic music with guitar and has two associated files:
- `{difficulty}_{i}_full_mix.ogg`: The full mix audio file containing all instruments.
- `{difficulty}_{i}_guitar_with_clicks`: The isolated target guitar track before mixing with added click sounds at the strum onsets predictions. All predictions are made on `other` stem using the MERT-based model fine-tuned on `other` stems.
