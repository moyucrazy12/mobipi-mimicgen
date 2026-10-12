# RBY1 — MimicGen Commands

This README records the current working pipeline for generating RBY1 `CloseSingleDoor` demonstrations with MimicGen. It can be updated to any other task, but these instructions are for the CloseSingleDoor as reference, also check how you want to divide your subtasks.

## 0. Key files

- **Source demonstrations:** `/home/$USER$/mobipi_EP/mobipi/rby1_seed_demos/close_door_surface_mg_source.hdf5`
- **Manifest:** `/home/$USER$//mobipi_EP/mobipi/external/robocasa/robocasa/scripts/rby1_seed_demos_10/manifest.csv`

Useful in case you do not know exactly the path:
```bash
 find ~/mobipi_EP/mobipi -name "manifest.csv" -print
```

- **MimicGen config:** `~/mobipi_EP/mobipi/configs/mimicgen/CloseSingleDoor_RBY1_surface_layout1_115mg.json`
- **Custom interface:** `MG_RBY1_CloseSingleDoor`

The source demos use the recorded RBY1 31-D WholeBodyIK controller configuration. MimicGen retargets the task-relevant left TCP / left-arm WBIK trajectory and reconstructs the full environment action through the custom interface.

---

## 1. Prepare the source dataset

**Input:** source HDF5 + manifest

```bash
cd ~/mobipi_EP/mobipi/external/mimicgen/mimicgen/scripts

python prepare_src_dataset_multistyle.py \
  --dataset /home/$USER$/mobipi_EP/mobipi/rby1_seed_demos/close_door_surface_mg_source.hdf5 \
  --manifest /home/$USER$/mobipi_EP/mobipi/external/robocasa/robocasa/scripts/rby1_seed_demos_10/manifest.csv \
  --env-interface MG_RBY1_CloseSingleDoor
```

**Output:** the same source HDF5, now containing `datagen_info`.

**Context:** the script restores each recorded RoboCasa model, uses the controller metadata stored in the HDF5, extracts MimicGen information, detects the first physical robot–microwave-door contact, and automatically writes the `stage_contact_door` subtask signal.

The resulting label is latched:

```text
0 0 0 0 0 1 1 1 1 ...
          ^
     first contact
```

This gives approximately:

```text
Subtask 0: approach the microwave door
Subtask 1: push / close the microwave door
```

---

## 2. Verify the automatic subtask labels

**Input:** prepared source HDF5

```bash
python - <<'PY'
import h5py
import numpy as np

p = "/home/$USER$/mobipi_EP/mobipi/rby1_seed_demos/close_door_surface_mg_source.hdf5"

with h5py.File(p, "r") as f:
    demos = sorted(
        f["data"],
        key=lambda x: int(x.split("_")[-1]),
    )

    for demo in demos:
        sig = f[
            f"data/{demo}/datagen_info/"
            "subtask_term_signals/stage_contact_door"
        ][:]

        transitions = np.where(
            (sig[:-1] == 0) & (sig[1:] == 1)
        )[0] + 1

        valid = (
            len(transitions) == 1
            and sig[0] == 0
            and sig[-1] == 1
            and int(sig.sum()) == len(sig) - transitions[0]
        )

        print(
            f"{demo:8s}",
            f"len={len(sig):4d}",
            f"boundary={transitions.tolist()}",
            f"push_len={int(sig.sum()):4d}",
            "OK" if valid else "BAD",
        )
PY
```

**Expected output:** exactly one `0 -> 1` boundary and `OK` for every source demo.

Example:

```text
demo_1   len=1391 boundary=[858] push_len=533 OK
```

---

## 3. Generate the MimicGen dataset

**Input:** prepared source HDF5 referenced by the MimicGen JSON config.

```bash
cd ~/mobipi_EP/mobipi/external/mimicgen

python mimicgen/scripts/generate_dataset.py \
  --config ~/mobipi_EP/mobipi/configs/mimicgen/CloseSingleDoor_RBY1_surface_layout1_115mg.json
```

**Output:** generated MimicGen demonstrations under the configured generation directory.

With experiment name:

```text
CloseSingleDoor_RBY1_surface_layout1_115mg
```

the successful dataset is expected at:

```text
<generation.path>/CloseSingleDoor_RBY1_surface_layout1_115mg/demo.hdf5
```

**Context:** MimicGen selects source subtask trajectories, transforms them into the target scene, reconnects/interpolates the subtasks, executes them through the custom RBY1 interface, checks task success, and stores the resulting demonstrations.

Do **not** use `--auto-remove-exp` unless you intentionally want to remove an existing experiment output directory.

---

## 4. Changing `generation.path`

Current style:

```json
"generation": {
    "path": "/home/$USER$/robocasa/datasets/v0.2/single_stage/kitchen_doors/CloseSingleDoor/mg_nav"
}
```

This is the **root output directory** for generated MimicGen experiments.

For example, to keep generated data inside the project:

```json
"generation": {
    "path": "/home/$USER$/mobipi_EP/mobipi/generated_datasets/mimicgen"
}
```

Then the current experiment would be written to:

```text
/home/smarc/mobipi_EP/mobipi/generated_datasets/mimicgen/
└── CloseSingleDoor_RBY1_surface_layout1_115mg/
    └── demo.hdf5
```

Changing `generation.path` does **not** change the source-demo HDF5 path. The source dataset is configured separately.

Using an absolute path is recommended because it is explicit and avoids ambiguity around `~` expansion.

---

## 5. Current pipeline

```text
10 scripted RBY1 source demos
        |
        v
prepare_src_dataset_multistyle.py
        |
        +-- restore per-demo RoboCasa XML
        +-- use recorded WholeBodyIK controller metadata
        +-- extract MimicGen datagen_info
        +-- detect first robot <-> microwave-door contact
        +-- automatically label stage_contact_door
        |
        v
prepared source HDF5
        |
        v
generate_dataset.py
        |
        +-- subtask selection
        +-- object-relative trajectory transformation
        +-- interpolation / connection
        +-- full RBY1 action reconstruction
        +-- execution + success checking
        |
        v
generated demo.hdf5
```