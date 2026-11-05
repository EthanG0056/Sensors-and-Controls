import numpy as np

NUM_RAYS = 36

print("\nSITES:\n")

for i in range(NUM_RAYS):

    angle = 2.0 * np.pi * i / NUM_RAYS

    # Rotate MuJoCo site's local +Z axis into the horizontal XY plane.
    #
    # Desired ray direction:
    # [cos(angle), sin(angle), 0]
    #
    # Quaternion format in MuJoCo:
    # [w, x, y, z]
    half = np.sqrt(0.5)

    qw = half
    qx = -np.sin(angle) * half
    qy = np.cos(angle) * half
    qz = 0.0

    print(
        f'<site '
        f'name="lidar_{i:02d}" '
        f'type="sphere" '
        f'pos="0 0 0" '
        f'size="0.001" '
        f'quat="{qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}" '
        f'rgba="0 0 0 0"/>'
    )


print("\nSENSORS:\n")

for i in range(NUM_RAYS):

    print(
        f'<rangefinder '
        f'name="lidar_{i:02d}_sensor" '
        f'site="lidar_{i:02d}"/>'
    )