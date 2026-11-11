# Results packages

| Folder | Provenance |
|--------|------------|
| `00_paper_reference/` | Transcription of the IEEE Access 2024 baseline paper |
| `01_expected_illustrative_scenario/` | **Synthetic** M0–M4 teaching scenario |

Do not mix CSVs or figures between the two.

## Regenerate

```bash
.venv/bin/python scripts/build_paper_reference.py
.venv/bin/python -m gfer.teaching_scenario
```
