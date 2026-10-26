# 4K stream framing verification

Exact source JPEGs and SHA256 hashes are retained in `stream-4k-verification/` and the JSON. Before:1280x720; after:3840x2160, exactly3x each axis. The native4K frame was visually inspected; the comparison displays it downscaled3x. Yellow marks show matched static rail features.

## Quantitative registration and board check

```json
{
  "registration": {
    "method": "SIFT perimeter rails only; ratio0.7; RANSAC1.5px",
    "matches": 216,
    "inliers": 204,
    "rms_residual_720px": 0.29242125153541565,
    "H_before_to_after_downscaled": [
      [
        1.0001522745778821,
        0.00013557461874352868,
        -0.09129686878232536
      ],
      [
        4.733012002963306e-06,
        1.0001474516027795,
        0.01776868118896688
      ],
      [
        8.468635293239876e-08,
        2.2524903044549132e-07,
        1.0
      ]
    ],
    "identity_frame_corner_shifts_720px": [
      0.09300991495884878,
      0.04239209302269089,
      0.1584040964834711,
      0.009595564770500215
    ]
  },
  "board": {
    "before": {
      "success": true,
      "count": 54,
      "median_pitch_px": 12.151420447761973,
      "minimum_pitch_px": 11.043036115788436
    },
    "after": {
      "success": true,
      "count": 54,
      "method": "native ROI SB exhaustive",
      "roi_origin_px": [
        172,
        920
      ],
      "full_frame_SB_initial_attempt": "failed",
      "median_pitch_px": 36.18434721492615,
      "minimum_pitch_px": 33.26717644845563,
      "frozen_zero_K_scaled3_parity_validation": {
        "rms_native_px": 1.3093793494259545,
        "rms_720_equivalent_px": 0.4364597831419848,
        "intrinsics_or_lens_refit": false,
        "only_pose_fitted": true
      }
    },
    "before_after_displacement_720px": {
      "rms": 0.42398094838363687,
      "median": 0.3782973221589005,
      "max": 0.7680001623689663,
      "limitation": "Board may move; localization differs at two resolutions."
    }
  }
}
```

## Interpretation

Static perimeter features test global framing while excluding movable cloth-area objects. Small residuals and near-identity frame-corner displacements support similar framing, but JPEG/localization/illumination changes affect these numbers. They do not prove identical optical intrinsics. Board-corner displacement may reflect board movement, not camera crop.

Frozen zero-radial K was scaled3 directly. No lens/intrinsics refit or activation occurred. Only checkerboard pose was fitted on one parity and tested on the other. These residuals are optical projection diagnostics at this placement, not verified physical XY accuracy.

The source is genuinely4K, but production Rust capture may still resize to720p through configured caps. More pixels do not establish3x physical accuracy. Original nine ball measurements belong only to their original associated source snapshot, not these changed-arrangement frames.
