# Dataset Summary — Cross-Factory Safety Benchmark

> Reference document for all datasets used in the cross-factory TTA benchmark.
> Canonical class mapping and integration status are defined here.

---

## Canonical Class Mapping

The benchmark unifies all datasets under **3 classes**:

| Canonical Index | Canonical Name | Rationale |
|---:|---|---|
| 0 | `hard_hat` | Primary safety-critical detection target |
| 1 | `no_hard_hat` | Violation detection — person without helmet |
| 2 | `person` | General person detection across all environments |

### Per-Dataset Mapping

| Canonical Class | SH17 | SHWD | Pictor-PPE | CHV |
|---|---|---|---|---|
| `hard_hat` (0) | Hard Hat (4) | hat/helmet (4) | helmet (0) | blue/red/white/yellow helmet (2,3,4,5) |
| `no_hard_hat` (1) | No Hard Hat (9) | — not present — | head (1) | — not present — |
| `person` (2) | person (16) | person (XML, class 2) | person (2) | person (0) |

**Dropped classes:** All other SH17 PPE classes (coverall, gloves, vest, etc.), CHV vest (1) — not present across datasets, out of scope for safety-hat benchmark.

---

## Dataset 1: SH17

| Property | Value |
|---|---|
| **Role** | Source domain (training) |
| **Total images** | 8,099 |
| **Total labels** | 8,099 |
| **Annotation format** | YOLO `.txt` (normalized cx cy w h) |
| **Classes** | 17 classes, indices 0–16 |
| **Split** | Train: 6,479 / Val: 1,620 |
| **Location** | `data/sh17/` |
| **Config** | `data/sh17/sh17.yaml` |
| **Integration status** | ✅ Fully integrated, YOLO-ready |

### SH17 Full Class List

```
0  Coverall          8   No Goggles
1  Face Shield       9   No Hard Hat
2  Gloves           10   No Safety Boot
3  Goggles          11   No Safety Vest
4  Hard Hat         12   No Vest
5  No Coverall      13   Safety Boot
6  No Face Shield   14   Safety Vest
7  No Gloves        15   Vest
                    16   person
```

### Canonical Remapping (SH17 → benchmark)
```
4  → 0  (Hard Hat     → hard_hat)
9  → 1  (No Hard Hat  → no_hard_hat)
16 → 2  (person       → person)
all others → DROP
```

---

## Dataset 2: SHWD (from VOC2028)

| Property | Value |
|---|---|
| **Role** | Target domain (evaluation) |
| **Total images** | 1,517 (test split only) |
| **Total labels** | 663 (images with ≥1 hat/helmet box) |
| **Annotation format** | YOLO `.txt` (converted from VOC XML) |
| **Raw source** | `data/raw/shwd/` — Pascal VOC XML, 7,581 images total |
| **Raw classes** | `hat`, `person` (VOC XML `<name>` strings) |
| **Location** | `data/shwd/` (converted output) |
| **Config** | `data/shwd/shwd.yaml` |
| **Integration status** | ✅ Fully integrated — `hat`/`helmet` → class 0, `person` → class 2 |

### Raw Structure
```
data/raw/shwd/
  Annotations/     7,581 XML files
  JPEGImages/      7,581 images
  ImageSets/Main/
    train.txt      5,457 stems
    val.txt          607 stems
    trainval.txt   6,064 stems
    test.txt       1,517 stems
```

### Current Conversion Behavior (`convert_shwd.py`)
- Keeps `hat` / `helmet` XML objects → writes as YOLO class id `0` (hard_hat)
- Keeps `person` XML objects → writes as YOLO class id `2` (person)
- Only processes `test.txt` split (1,517 images → label files for images with ≥1 annotation)

### Canonical Remapping (SHWD → benchmark)
```
hat/helmet → 0  (hard_hat)
person     → 2  (person)
```

---

## Dataset 3: Pictor-PPE

| Property | Value |
|---|---|
| **Role** | Target domain (evaluation) |
| **Total images** | 152 |
| **Total labels** | 152 |
| **Annotation format** | YOLO `.txt` (converted from custom TSV source) |
| **Raw source format** | Tab-separated file with VOC-style pixel bboxes |
| **Raw source path** | `Labels/pictor_ppe_crowdsourced_approach-01_test.txt` |
| **Location** | `data/pictor_ppe/` |
| **Config** | `data/pictor_ppe/pictor.yaml` |
| **Metadata** | `data/pictor_ppe/dataset-metadata.json` (Kaggle: `zyanahmed/pictor-ppe-yolo`) |
| **Integration status** | ✅ Converted and YOLO-ready |

### Pictor-PPE Class List
```
0  helmet
1  head
2  person
```

### Canonical Remapping (Pictor-PPE → benchmark)
```
0 → 0  (helmet → hard_hat)
1 → 1  (head   → no_hard_hat)
2 → 2  (person → person)
```

> Clean 1:1 mapping. No drops needed.

---

## Dataset 4: CHV

| Property | Value |
|---|---|
| **Role** | Target domain (evaluation) |
| **Total images** | 1,330 |
| **Total labels** | 1,330 |
| **Annotation format** | YOLO-like `.txt` (normalized cx cy w h, same format as YOLO) |
| **Location** | `data/chv/` |
| **Split** | Train: 1,064 / Valid: 133 / Test: 133 |
| **Split lists** | `data/chv/data split/train.txt` etc. |
| **Integration status** | ❌ Not integrated — no entry in prepare_data.py or setup_data.py |

### CHV Full Class List
```
0  person
1  vest
2  blue helmet
3  red helmet
4  white helmet
5  yellow helmet
```

### Canonical Remapping (CHV → benchmark)
```
2 → 0  (blue helmet   → hard_hat)
3 → 0  (red helmet    → hard_hat)
4 → 0  (white helmet  → hard_hat)
5 → 0  (yellow helmet → hard_hat)
0 → 2  (person        → person)
1 → DROP (vest — not present in other datasets)
```

> Note: CHV contributes no `no_hard_hat` annotations. This class will only appear in SH17 and Pictor-PPE evaluation.

---

## Benchmark Summary

| Dataset | Role | hard_hat | no_hard_hat | person | Total Images |
|---|---|---|---|---|---|
| SH17 | Source (train) | ✅ | ✅ | ✅ | 8,099 |
| SHWD | Target (eval) | ✅ | ❌ | ✅ | 1,517 |
| Pictor-PPE | Target (eval) | ✅ | ✅ | ✅ | 152 |
| CHV | Target (eval) | ✅ | ❌ | ✅ | 1,330 |

### Domain Shift Pairs (source → target)
```
SH17 → SHWD         severe shift (web-mined Chinese construction sites)
SH17 → Pictor-PPE   moderate shift (crowdsourced diverse environments)
SH17 → CHV          moderate shift (color-annotated helmet dataset)
```

---

## File Locations Reference

```
data/
  sh17/
    images/{train,val}/
    labels/{train,val}/
    sh17.yaml
  shwd/
    images/test/
    labels/test/
    shwd.yaml
  raw/
    shwd/                     ← raw source for SHWD (VOC XML)
      Annotations/
      JPEGImages/
      ImageSets/Main/
  pictor_ppe/
    images/test/
    labels/test/
    pictor.yaml
    dataset-metadata.json
  chv/
    images/
    annotations/
    data split/
      train.txt
      valid.txt
      test.txt
```

---

*Last updated: April 2026*
*Research: Cross-Factory Domain Generalization for Industrial Worker Safety Perception*