# StructEP

StructEP provides checkpoint-compatible inference for a structure-aware PoseMIL ensemble that estimates cardiac ion-channel potency. The model combines ligand fingerprints with protein–ligand pose geometry, pools evidence across docked poses and receptor states, and reports prediction uncertainty.

The model registry contains ten ensemble members for each supported channel:

| Channel | Registry key | Members | Primary output |
| --- | --- | ---: | --- |
| hERG | `herg` | 10 | pIC50 and uncertainty |
| NaV1.5 | `nav1d5` | 10 | pIC50 and uncertainty |
| CaV1.2 | `cav1d2` | 10 | pIC50 and uncertainty |

## Highlights

- Exact checkpoint compatibility with strict key, shape, dtype, tensor-count, and parameter-count validation.
- Single-member or ten-member channel-ensemble inference.
- Aleatoric, epistemic, and total predictive uncertainty summaries.
- Safe NumPy input loading with pickle deserialization disabled.
- Complete tensor-shape, dtype, index-range, and hierarchy validation.
- CPU, CUDA, and Apple Metal execution through a Python API and command-line interface.
- Compact channel configurations and a thirty-member registry bundled with the package.

## Installation

StructEP supports Python 3.10, 3.11, and 3.12.

```bash
git clone https://github.com/cardiac-modeling/StructEP.git
cd StructEP
python -m pip install -e .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
python -m pytest -ra
```

## Command-line usage

List all registered models or filter by channel:

```bash
structep list-models
structep list-models --channel herg
```

Strict-load a complete channel ensemble and execute a deterministic forward check:

```bash
structep verify \
  --weights-dir /path/to/posemil-assets \
  --channel herg \
  --forward-smoke \
  --output verification.json
```

Verify one exact member:

```bash
structep verify \
  --weights-dir /path/to/posemil-assets/model_weights \
  --model-id posemil_nav1d5_member03 \
  --forward-smoke
```

Create a structurally valid input archive for deployment checks:

```bash
structep make-smoke-input \
  --channel cav1d2 \
  --sample-id example_compound \
  --output smoke_input.npz
```

Run a ten-member channel ensemble:

```bash
structep predict \
  --weights-dir /path/to/posemil-assets \
  --channel herg \
  --input prepared_herg_batch.npz \
  --output herg_predictions.json
```

Run selected members and retain each member's result:

```bash
structep predict \
  --weights-dir /path/to/posemil-assets \
  --model-id posemil_herg_member00 \
  --model-id posemil_herg_member04 \
  --input prepared_herg_batch.npz \
  --include-members
```

The weight directory may contain the `.safetensors` files directly or expose them through a `model_weights/` child directory.

## Python API

Run a complete channel ensemble from a prepared archive:

```python
from structep import predict_npz

report = predict_npz(
    "prepared_herg_batch.npz",
    weights_directory="/path/to/posemil-assets",
    channel="herg",
    device="cpu",
)

for prediction in report["predictions"]:
    print(
        prediction["sample_id"],
        prediction["mean_pic50"],
        prediction["total_std_pic50"],
    )
```

Load one checkpoint directly:

```python
import torch

from structep import load_npz_batch, load_registered_model

model, config, spec = load_registered_model(
    "posemil_herg_member00",
    weights_directory="/path/to/posemil-assets",
    device="cuda",
)
batch, sample_ids = load_npz_batch(
    "prepared_herg_batch.npz",
    config,
    device="cuda",
)

with torch.inference_mode():
    output = model(batch)

print(spec.model_id)
print(output["mu_pic50"])
print(torch.exp(0.5 * output["log_var_pic50"]))
print(torch.sigmoid(output["blocker_logit"]))
```

## Input batch format

StructEP accepts a compressed `.npz` archive containing model-ready tensors. The archive is opened with `allow_pickle=False`.

Let:

- `B` be the number of molecule-channel bags;
- `I` be the total number of docked pose instances;
- `S` be the total number of receptor states;
- `R` be the padded residue count; and
- `A` be the padded ligand-atom count.

| Array | Shape | Runtime dtype | Description |
| --- | --- | --- | --- |
| `x_2d` | `[B, 2048]` | `float32` | Morgan fingerprint features |
| `protein_aa` | `[I, R]` | `int64` | Protein residue-type indices |
| `protein_xyz` | `[I, R, 3]` | `float32` | Protein coordinates |
| `protein_mask` | `[I, R]` | `bool` | Valid protein positions |
| `ligand_atom` | `[I, A]` | `int64` | Ligand atom-type indices |
| `ligand_xyz` | `[I, A, 3]` | `float32` | Ligand coordinates |
| `ligand_mask` | `[I, A]` | `bool` | Valid ligand positions |
| `pose_quality` | `[I, 5]` | `float32` | Pose-quality features |
| `bag_index` | `[I]` | `int64` | Pose-to-bag mapping |
| `state_index` | `[I]` | `int64` | Pose-to-state mapping |
| `state_to_bag` | `[S]` | `int64` | State-to-bag mapping |
| `state_type_idx` | `[S]` | `int64` | State type: `0` unknown, `1` open, `2` inactivated, `3` other |
| `state_role` | `[S]` | `float32` | Primary or auxiliary state indicator |
| `state_features` | `[S, 0]` | `float32` | Zero-width checkpoint-compatibility array |
| `channel_idx` | `[B]` | `int64` | Zero for a channel-specific checkpoint |
| `num_bags` | scalar | integer | Number of bags |
| `num_states` | scalar | integer | Number of receptor states |

The optional `sample_ids` array must be one-dimensional, contain unique strings, and have one entry per bag. The optional scalar `schema_version` must equal `1`.

Validation also requires:

- every bag and state to contain at least one pose;
- every pose to contain at least one valid protein residue and ligand atom;
- all index arrays to stay within range;
- `bag_index` to agree with `state_to_bag[state_index]`; and
- every floating-point input to be finite.

Production inputs should use the same atom typing, receptor-state representation, coordinate conventions, molecular fingerprints, and pose-quality definitions as the registered checkpoints.

## Prediction output

The JSON report contains one record per input bag. Core fields are:

| Field | Meaning |
| --- | --- |
| `mean_pic50` | Ensemble mean pIC50 |
| `aleatoric_std_pic50` | Mean modeled observation uncertainty |
| `epistemic_std_pic50` | Ensemble-member disagreement |
| `total_std_pic50` | Combined predictive standard deviation |
| `pic50_interval_95` | Mean ± 1.96 total standard deviations |
| `blocker_probability` | Mean sigmoid blocker probability |
| `ordinal_exceedance_probability` | Mean cumulative probability above each configured pIC50 threshold |

Use `--include-members` or `include_members=True` to include individual member predictions.

## Model and checkpoint contract

The package registry is stored at `src/structep/configs/model_registry.csv`. Each channel has one compact inference configuration:

```text
src/structep/configs/
├── model_registry.csv
├── herg.yaml
├── nav1d5.yaml
└── cav1d2.yaml
```

Every registered checkpoint is validated as a float32 `safetensors` state dictionary with:

- 223 tensors;
- 5,180,904 tensor elements; and
- 5,180,840 trainable model parameters.

Loading is always strict. Checkpoints with missing, additional, renamed, reshaped, or non-float32 tensors are rejected before inference.

## Architecture

The frozen inference path consists of:

1. a 2048-bit ligand fingerprint multilayer perceptron;
2. Transformer encoders for protein residues and ligand atoms;
3. distance-aware bidirectional protein–ligand cross-attention;
4. attention pooling over poses within each receptor state;
5. attention pooling across receptor states;
6. gated 2D and 3D fusion; and
7. pIC50 mean, log-variance, blocker, and ordinal prediction heads.

The implementation preserves the checkpoint state-key layout while exposing a focused inference API.

## Repository layout

```text
StructEP/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── src/structep/
│   ├── batch.py
│   ├── cli.py
│   ├── errors.py
│   ├── inference.py
│   ├── registry.py
│   ├── configs/
│   ├── model/
│   └── _vendor/stageb/
└── tests/
```

## Research use

StructEP is research software for computational modeling. Its outputs are not clinical recommendations and must not be used as the sole basis for patient-level or safety-critical decisions.

## License

StructEP is released under the MIT License. See [LICENSE](LICENSE).
