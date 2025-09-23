# rhythmic-pattern-transcription
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

downbeats = np.asarray([...], dtype=np.float64)
processed = process_downbeats(downbeats)
```

## TODO
- [ ] Add dataset excerpts
- [x] Add BeatThis post-processing source code
