# Architecture Notes

## Core principle

The selected target and semantic obstacles are separate perception streams:

```text
Qwen target grounding -> category-agnostic tracker -> target guidance
YOLO obstacle detection ---------------------------> safety arbiter
2D LiDAR provider ---------------------------------> safety arbiter
```

YOLO is not allowed to replace the selected target simply because another object
has the same category or is currently farther right. The selected target owns a
stable logical ID until the user resets it or re-grounding explicitly succeeds.

## Replacement points

- Replace `CSRTTargetTracker` with OSTrack or SAM 2 without changing Qwen grounding.
- Replace `NullLidarProvider` with the vendor SDK adapter.
- Replace virtual `MotionGuidance` output with a chassis transport only after the
  safety layer has been validated.
