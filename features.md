# Localization features

Status: **done** = implemented and checked on simulated data. **planned** = not built yet. Nothing has been tested on a real ROV or a real Water Linked unit.

## Output
- **done** ROV pose in the ship frame, per image: position (x, y, z), orientation (roll, pitch, yaw, quaternion) → `rov_ship.csv`, `poses_ship.txt`
- **done** 1-sigma position uncertainty per image, including the uncertainty of the map-to-ship alignment
- **done** the map-to-ship transform, saved in `localization_info.json`

## Sensors
| Sensor | Role | Status |
|---|---|---|
| ZED visual-inertial pose | smooth relative motion and rotation, trusted less when tracking quality drops | done |
| DVL velocity | metric scale, keeps moving when vision fails | done |
| ROV IMU (roll, pitch) | gravity direction, fixes tilt | done |
| Pressure sensor | absolute depth, bounds vertical drift | done |
| Water Linked Underwater GPS | absolute position, no drift | done |
| Compass | not used (steel hull) | by design |
| ZED RGB / depth against the COLMAP map | map-based fixes and depth-to-hull constraint | planned |

## Underwater GPS handling
- **done** one position factor per acoustic fix, interpolated between the two neighbouring images
- **done** robust loss, so multipath outliers are down-weighted
- **done** uses the unit's reported `std` when available, with `sigma_ugps` as the floor and fallback
- **done** locator lever arm (`t_loc_b`): the locator position on the ROV body is accounted for
- **done** dropouts: gaps with no fixes are bridged by dead reckoning
- **done** fixes outside the image time span are ignored
- **done** outlier count (fixes beyond 3 sigma) reported

## Frames
- **done** automatic alignment of the VIO map frame to the receiver frame (yaw about gravity + translation), solved together with the poses
- **done** ship frame defined by the receiver mounting on the ship (`R_ship_rx`, `t_ship_rx`)
- **done** warning when the trajectory has no horizontal extent, so heading cannot be observed

## Tools
- **done** `--gt-ship` prints position error against ground truth
- **done** `--no-ugps` runs dead reckoning only, for comparison
- **done** simulated UGPS generator (`synth/make_synth_uwgps.py`) with a rotated and offset ship frame, noise, outliers and a dropout

## Not built yet
- **planned** online, real-time operation (sliding-window or incremental solver; currently one offline batch solve)
- **planned** map-based absolute fix: match images to the COLMAP model and solve PnP
- **planned** depth-to-hull constraint for featureless paint (standoff and tilt)
- **planned** loop closures against the map
- **planned** receiver attitude input, to handle a listing or trimmed ship (currently gravity is assumed to be "down" in the receiver frame)
- **planned** logger for the Water Linked unit's API output
- **planned** automatic time-offset estimation between the UGPS log and the other sensors
