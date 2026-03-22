# Contributing to PkVision

Thank you for your interest in contributing. PkVision is a community-driven project and benefits from many types of contributions: code, trick definitions, training clips, translations, bug reports, and documentation improvements.

---

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Code Contributions](#code-contributions)
- [Trick Definitions](#trick-definitions)
- [Training Clips](#training-clips)
- [Translations](#translations)
- [Code Style](#code-style)
- [Reporting Issues](#reporting-issues)

---

## Ways to Contribute

| Type | Skill Level | Description |
|------|-------------|-------------|
| Training clips | No coding required | Submit parkour videos to improve detection accuracy |
| Trick proposals | No coding required | Propose new tricks for the catalog |
| Translations | No coding required | Add trick names in new languages |
| Bug reports | No coding required | Report detection errors or software bugs |
| Trick definitions | Basic JSON | Add new tricks as JSON config files |
| Code | Python | Bug fixes, features, tests, performance |
| Documentation | Markdown | Improve guides, add examples, fix errors |

---

## Code Contributions

### Setup

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/pkvision.git
   cd pkvision
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Run the test suite to verify everything works:
   ```bash
   pytest
   ```

### Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes.
3. Run tests:
   ```bash
   pytest
   ```
4. Commit with a descriptive message following conventional commit format:
   ```bash
   git commit -m "feat: add lazy vault trick definition"
   git commit -m "fix: correct angle range for gainer takeoff phase"
   git commit -m "test: add unit tests for scoring engine deduplication"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request against `main`.

### Branch naming

- `feature/description` -- New functionality
- `fix/description` -- Bug fixes
- `refactor/description` -- Code restructuring
- `test/description` -- Test additions or changes
- `docs/description` -- Documentation changes

### Pull Request guidelines

- Keep PRs focused on a single change.
- Include a clear description of what the PR does and why.
- Reference any related issues.
- Ensure all tests pass.
- Add tests for new functionality when applicable.

---

## Trick Definitions

Tricks are defined as JSON files in `data/tricks/catalog/<lang>/`. No code changes are required to add a new trick.

### Steps

1. Choose the trick you want to add.
2. Create a JSON file in `data/tricks/catalog/en/` following the schema in `data/tricks/schema.json`.
3. The file name should match the `trick_id` (e.g., `lazy_vault.json` for trick_id `lazy_vault`).
4. Optionally, create a matching file in `data/tricks/catalog/fr/` for the French translation.

### Required fields

| Field | Type | Description |
|-------|------|-------------|
| `trick_id` | string | Unique identifier in snake_case (`^[a-z][a-z0-9_]*$`) |
| `category` | string | One of: `flip`, `vault`, `twist`, `combo`, `spin`, `precision` |
| `difficulty` | number | 0.0 to 10.0 |
| `detection_method` | string | `angle_threshold` or `temporal_model` |
| `phases` | array | At least one phase with angle rules |
| `names` | object | At least `en` key with the English name |

### Optional fields

| Field | Type | Description |
|-------|------|-------------|
| `tags` | array | Searchable tags (e.g., `["acrobatic", "aerial"]`) |
| `composable_with` | array | Trick IDs this trick can combine with |

### Phase structure

Each phase represents a stage of the trick (approach, takeoff, execution, landing, recovery). Phases contain:

- `name` -- One of: `approach`, `takeoff`, `execution`, `landing`, `recovery`
- `duration_range_ms` -- Expected duration as `[min_ms, max_ms]`
- `angle_rules` -- Joint angle constraints for detection
- `velocity_rules` -- Joint velocity constraints (optional)
- `trajectory` -- Rotation trajectory constraints (optional)

### Angle rules

Each angle rule specifies a joint, a valid angle range, and an optional description:

```json
{
  "joint": "knee",
  "min": 130,
  "max": 165,
  "description": "Slight knee bend for takeoff"
}
```

Available joints: `knee`, `hip`, `elbow`, `shoulder`, `spine`, `neck`, and their left/right variants (`left_knee`, `right_hip`, etc.).

### Validation

Validate your JSON file against the schema:

```bash
python -c "
import json, jsonschema
with open('data/tricks/schema.json') as f: schema = json.load(f)
with open('data/tricks/catalog/en/your_trick.json') as f: trick = json.load(f)
jsonschema.validate(trick, schema)
print('Valid')
"
```

### Guidelines for difficulty ratings

- 1-2: Basic vaults and movements (kong vault, lazy vault)
- 3-4: Single flips and standard acrobatics (front flip, back flip, side flip)
- 5-6: Advanced single tricks (webster, gainer, 360 flip)
- 7-8: Double rotations and complex combos (double front, double kong)
- 9-10: Elite-level tricks (triple cork, multi-axis rotations)

When in doubt, reference the FIG Parkour Code of Points or discuss in the GitHub issue.

---

## Training Clips

Training clips are the fuel for the ST-GCN model. The more diverse the dataset, the more accurate the detections.

### How to submit

1. Go to [Issues > New Issue > Submit Training Clip](../../issues/new?template=clip_submission.yml).
2. Fill out the form: trick name, video link, timestamp, camera angle.
3. A maintainer will review and add the clip to the training pipeline.

You can also submit clips via the API:

```bash
curl -X POST http://localhost:8000/api/v1/submissions \
  -H "Content-Type: application/json" \
  -d '{
    "submission_type": "clip",
    "trick_name": "Front Flip",
    "video_url": "https://example.com/video.mp4",
    "description": "Clean front flip on grass, side angle",
    "submitter_name": "Your Name"
  }'
```

### Filming guidelines

See [CLIP_GUIDELINES.md](CLIP_GUIDELINES.md) for detailed filming requirements. The key points:

- Minimum 720p, 30fps
- Side or diagonal camera angle
- Full body visible throughout
- Contrasting clothing
- One trick per clip when possible

---

## Translations

Trick names are stored in the `names` field of each trick JSON file:

```json
"names": {
  "en": "Front Flip",
  "fr": "Salto Avant"
}
```

To add a new language:

1. Add the language code and translated name to the `names` object in the English catalog file (e.g., `"de": "Vorwartssalto"`).
2. Optionally, create a full translated catalog directory at `data/tricks/catalog/<lang>/` with localized description fields.

Language codes follow ISO 639-1 (e.g., `en`, `fr`, `de`, `es`, `pt`, `ja`).

---

## Code Style

### Requirements

- Python 3.11 or higher
- Type hints on all function signatures
- Tests for new functionality (pytest)

### Conventions

- Follow existing code patterns in the repository.
- Use `from __future__ import annotations` at the top of every module.
- Pydantic models go in `core/models.py` or `db/models.py`.
- New detection strategies implement the `DetectionStrategy` protocol defined in `core/recognition/strategies/base.py`.
- Keep functions focused and single-purpose.
- Prefer explicit error handling with descriptive messages over silent failures.

### Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/unit/test_catalog.py
```

Tests live in `tests/unit/` for unit tests and `tests/` for integration tests. Test fixtures go in `tests/fixtures/`.

---

## Reporting Issues

When reporting a bug or detection error:

1. Open a [GitHub Issue](../../issues/new).
2. Include:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce
   - For detection errors: the video file or a link, the expected trick, and the actual output
   - Python version and OS

---

## Code of Conduct

Be respectful and constructive. This project is used by athletes, coaches, judges, and developers from different countries and backgrounds. Contributions in English or French are welcome.

---

## Questions?

Open a GitHub issue with the `question` label. We are happy to help.
