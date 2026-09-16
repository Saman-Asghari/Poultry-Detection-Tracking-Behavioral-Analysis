# Poultry Detection, Tracking & Behavioral Analysis

A computer vision system for detecting, counting, and tracking broiler chickens
in industrial and prototype poultry houses, and extracting quantitative
behavioral parameters from their movement across the growth cycle.

Built on the public **PIO dataset**, using **YOLOv10** for detection and
**BoT-SORT + OSNet** for multi-object tracking and re-identification.

---

## Overview

Manual monitoring of poultry houses doesn't scale: pens hold thousands of
birds, visual density is extremely high, and human observation can't
continuously track population, movement, or social structure. This project
builds an automated pipeline that turns raw camera footage into structured,
per-bird behavioral data — a foundation for anomaly detection, welfare
monitoring, and flock-level social analysis.

**Goals:**
1. Detect and count chickens accurately in dense scenes (up to ~300 birds/frame).
2. Track individual chickens across raw video to extract movement trajectories.
3. Compute a rich set of kinematic and social parameters per chicken.
4. Build reference behavioral distributions, as groundwork for detecting
   abnormal patterns.
5. Compare behavior across different weeks of the growth cycle.

---

## System Architecture

The pipeline is composed of three stages, each independently swappable:

```
┌──────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
│   1. DETECTION    │     │    2. TRACKING         │     │  3. BEHAVIORAL ANALYSIS │
│                   │     │                        │     │                         │
│   YOLOv10m        │────▶│   BoT-SORT             │────▶│  Per-track kinematic &  │
│   (object         │     │   ├─ Kalman filter +   │     │  social parameter       │
│   detector,       │     │   │   IoU motion model  │     │  extraction             │
│   trained on PIO) │     │   ├─ Camera motion     │     │  + track stitching to   │
│                   │     │   │   compensation (GMC)│     │  repair fragmented IDs  │
│                   │     │   └─ OSNet Re-ID        │     │                         │
│                   │     │       (appearance       │     │                         │
│                   │     │        embedding)       │     │                         │
└──────────────────┘     └───────────────────────┘     └────────────────────────┘
        │                            │                              │
        ▼                            ▼                              ▼
  Bounding boxes            Per-frame track_id +           chicken_movement_
  per frame                 (x, y, w, h) trajectory        params_full.csv
```

**Stage 1 — Detection (YOLOv10m).** A single-stage detector trained on the PIO
dataset to localize chickens frame-by-frame, even at extreme density.

**Stage 2 — Tracking (BoT-SORT + OSNet).** Detections are linked across frames
into per-chicken trajectories. BoT-SORT combines a Kalman-filter motion model,
IoU-based association (inherited from ByteTrack), and camera-motion
compensation — but the critical addition for this environment is **OSNet**, a
lightweight re-identification CNN that generates an appearance embedding for
every detected chicken. When a bird is briefly hidden behind another (which
happens constantly at this density), OSNet lets the tracker re-recognize it
by what it *looks like*, not just where it was predicted to be — substantially
reducing identity loss compared to motion-only tracking.

**Stage 3 — Behavioral analysis.** Raw trajectories are post-processed
(track stitching to merge fragmented IDs caused by occlusion, then filtered
for minimum length) and converted into a structured table of kinematic and
social parameters per chicken, per video/week.

---

## Dataset — PIO

- **Source:** [Zenodo record](https://zenodo.org/records/16686320)
- **Labeled images:** 1,487 images, 327,289 annotated chicken instances
- **Format:** `.jpg` images (1280×720) + YOLO-format `.txt` labels
- **Class:** single class (chicken)
- **Raw video:** 5 raw `.mp4` files, 30 fps, ~70–80 min each, spanning the
  growth cycle from hatch to market weight (~5–7 weeks)


> Videos are currently mapped to weeks 1–5 by filename order — a working
> assumption, not yet verified against the dataset's timestamp metadata.

---

## Detection Model — YOLOv10m

| Metric | Result |
|---|---|
| Precision | 0.953 |
| Recall | 0.877 |
| mAP50 | 0.897 |
| mAP50-95 | 0.684 |
| Inference speed | ~9–13 ms/image |

---

## Tracking — BoT-SORT + OSNet Re-ID

- **Motion + association:** Kalman filter, IoU matching, sparse-optical-flow
  camera motion compensation.
- **Appearance Re-ID:** **OSNet** (Omni-Scale Network) generates a compact
  appearance embedding per detection, fused with motion cues during
  association — the key mechanism that lets the tracker recover a chicken's
  identity after occlusion instead of spawning a new ID.
- **Extended track memory:** `track_buffer` increased from the default so a
  temporarily lost bird stays "recoverable" longer.
- **Frame sampling:** raw video (30 fps, ~130k frames/video) processed at a
  reduced effective rate via `vid_stride` to keep runtime tractable.
- **Outputs per video:** a trajectories CSV (`frame, track_id, x, y, w, h`)
  and an annotated `.mp4` with boxes + track IDs drawn per frame.

### Identity fragmentation & mitigation

Extreme density causes frequent occlusion, which fragments a single bird's
trajectory into many short-lived IDs. Early runs showed this clearly:

Mitigated via OSNet Re-ID + extended track buffer + a custom **track-stitching**
post-process (velocity-based candidate matching, greedy nearest-cost merging
via union-find, implemented with a sparse candidate-edge list for memory
safety at this scale) + a minimum track-length filter on the result.

---

## Behavioral & Social Parameters

Computed per qualifying track segment:

**Kinematics** — total distance, average/std/max speed, acceleration,
movement-vs-rest ratio.

**Movement structure** — movement/rest bout counts and durations, spatial and
temporal gaps between rest periods.

**Direction & turning** — dominant direction, direction consistency, sharp-turn
frequency.

**Social/spatial context** — nearest-neighbor distance, local group size,
fraction of time alone, distance to nearest group when isolated.


---

## Research Potential

This pipeline — dense-scene detection, occlusion-robust re-identification
tracking, and structured per-bird behavioral extraction, applied across a
full growth cycle — goes beyond a course project scope. With a verified week
mapping, calibrated units, and a defined anomaly-detection method on top of
the extracted parameters, this could be developed into a standalone research
paper on automated behavioral phenotyping in precision poultry farming.
