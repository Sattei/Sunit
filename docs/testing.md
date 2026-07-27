# Sunit testing record

## Automated checks

The normal test suite uses fake DSINE and BiRefNet adapters. It does not load
model weights or allocate CUDA memory.

Covered behavior:

- `RelightSettings` defaults;
- missing and unsupported inputs;
- virtual-environment interpreter-path preservation;
- repeated job-ID rejection and explicit overwrite;
- safe API-precreated upload directories;
- stable progress stages;
- processing resize;
- no-subject detection;
- invalid light vectors;
- disguised non-image uploads;
- output URL sanitization;
- clean queued-job API contract.

Run:

```bash
make test
```

## Verified real-pipeline cases

| Input | Parameters | Result | Background change | Foreground change | Notes |
| --- | --- | --- | ---: | ---: | --- |
| `docs/assets/original-portrait.png` | Natural preset, light `(0.55, -0.20, 0.80)` | Pass | `0.00000000` | `0.01684618` | Supplied 477×1065 portrait |

The output matched the processed input dimensions and contained only finite
pixel values.

## Remaining coverage matrix

These cases are intentionally not marked as passed until suitable licensed
fixtures are added and processed:

| Case | Status |
| --- | --- |
| Side-lit portrait | Not yet run |
| Dark clothing | Not yet run |
| Light clothing | Not yet run |
| Outdoor background | Not yet run |
| Bright background | Not yet run |
| Low-light portrait | Not yet run |
| Full-body image | Not yet run |
| No-person image through real BiRefNet | Not yet run |
| Multiple-person image | Experimental / not yet run |
| Non-portrait object | Experimental / not yet run |

Sunit should not be described as universally validated from the current
portrait case.
