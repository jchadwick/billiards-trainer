# Prospective frozen-model checkerboard validation

view-009

Frozen full-training-set intrinsics and lens parameters; held-out view pose estimated from one corner parity and evaluated on the other, then swapped.

| Model | Frozen validation RMS px |
|---|---:|
| zero | 0.2279 |
| k1 | 0.2307 |
| k1k2 | 0.2299 |

One new image probes projection consistency at one placement. It does not prove table XY/ball-center accuracy, establish physical uncertainty, or justify selecting a more complex model from this result alone.

Coefficients were frozen before this view was captured. No calibration activated.

## Reviewed conclusion

The frozen-baseline hash and exact model coefficients match the eight-view
baseline. Training IDs are view-001 through view-008; view-009 is excluded.
Each score contains 54 corner residuals: estimate pose on one alternating
checkerboard parity and score the other, then swap. Every corner is scored
once while excluded from its pose-fitting subset. Neighboring corner errors
may still be correlated.

Neither eight-view whole-view cross-validation nor this prospective ninth
view demonstrates a benefit from adding the tested radial terms. The small
differences between these models do not establish zero lens distortion.
The small target covers a local patch in each independently estimated pose;
pose flexibility, image localization and model uncertainty limit what these
residuals can identify. They cannot determine the cause of elongated ball
silhouettes. Checkerboard reprojection error is not measured table X/Y error
or millimeter ball-center accuracy.

Pause further board captures as planned. A subsequent software investigation
can inspect actual ball outlines in existing source images and compare known
straight cushion segments with their observed image curves, accounting for
shadows, occlusion and reference ambiguity. These complementary diagnostics
can examine contour bias and optical-model adequacy without more tedious
manual round-ball measurements. They are proposed work, not implemented or
validated corrections.
