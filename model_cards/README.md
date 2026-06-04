# Model cards

Optional, human-readable disclosure pages — **one per model**, named `<model_id>.md`.
They are for reviewers; **nothing here is parsed** and they do **not** affect the leaderboard
(size, scores, and provenance all come from `meta` in the submission / `model_meta.json`).

When you submit a model, add `model_cards/<model_id>.md` with:

```markdown
# <model_id>

- **Display name:**
- **Parameters:**
- **Architecture:**
- **Weights / URL:**
- **Tokenizer / input:**
- **Pooling:** mean over tokens (or describe)

## Training data
<what the model was pretrained on>

## Disclosure
- **Zero-shot?** no / yes — <explain>
- **Train/test contamination with benchmark tasks?** none known / <describe any overlap>
- **Harness version used:** GENEB-0.1.0
- **Submitted by:**
```
