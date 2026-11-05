import numpy as np

NUM_RAYS = 36

print("\nSITES:\n")

for i in range(NUM_RAYS):

    angle = 2.0 * np.pi * i / NUM_RAYS

    print(
        f'<site name="lidar_{i:02d}" '
        f'pos="0 0 0" '
        f'euler="0 1.570796 {angle:.6f}"/>'
    )


print("\nSENSORS:\n")

for i in range(NUM_RAYS):

    print(
        f'<rangefinder '
        f'name="lidar_{i:02d}_sensor" '
        f'site="lidar_{i:02d}"/>'
    )