# Terminal pose results — IDEAL-STATE DIAGNOSTIC ONLY

No estimated-state integration, obstacle avoidance or ArUco docking is tested. All failures are retained.

| kv | dt (s) | Strategy | Failures / 10 | Mean final position (m) | Mean absolute final heading (rad) | Mean settling among successes (s) | Max position-plane overshoot (m) | Saturated intervals (%) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.002 | stationary_stop | 9 / 10 | 0.087988 | 0.866350 | 0.000 | 0.000000 | 0.000 |
| 0.1 | 0.002 | hybrid | 9 / 10 | 0.049953 | 0.782890 | 0.000 | 0.000000 | 0.000 |
| 0.1 | 0.001 | stationary_stop | 9 / 10 | 0.088024 | 0.866770 | 0.000 | 0.000000 | 0.000 |
| 0.1 | 0.001 | hybrid | 9 / 10 | 0.049963 | 0.783559 | 0.000 | 0.000000 | 0.000 |
| 1 | 0.002 | stationary_stop | 7 / 10 | 0.082218 | 0.864754 | 0.067 | 0.000000 | 0.000 |
| 1 | 0.002 | hybrid | 3 / 10 | 0.016502 | 0.383474 | 4.579 | 0.001183 | 0.000 |
| 1 | 0.001 | stationary_stop | 7 / 10 | 0.082367 | 0.864947 | 0.067 | 0.000000 | 0.000 |
| 1 | 0.001 | hybrid | 3 / 10 | 0.016723 | 0.402738 | 4.514 | 0.001264 | 0.000 |

Settling requires position ≤0.025 m, absolute heading ≤0.07 rad, observed speed ≤0.008 m/s and yaw rate ≤0.04 rad/s continuously to the horizon, for at least 2 s. Unsettled trials fail. Means include all ten scenarios; settling means exclude failures.

| Trial | Final position (m) | Final heading (rad) | Settling (s) | Failed | Mode switches | Final mode |
|---|---:|---:|---:|---|---:|---|
| ahead_kv0.1_dt0.002_stationary_stop | 0.200067 | 0.000646 | — | True | 0 | stationary_stop |
| ahead_kv0.1_dt0.002_hybrid | 0.039394 | 0.000193 | — | True | 1 | fault |
| behind_kv0.1_dt0.002_stationary_stop | 0.149933 | 0.000646 | — | True | 0 | stationary_stop |
| behind_kv0.1_dt0.002_hybrid | 0.117849 | 2.020811 | — | True | 1 | fault |
| lateral_kv0.1_dt0.002_stationary_stop | 0.150046 | 1.000646 | — | True | 0 | stationary_stop |
| lateral_kv0.1_dt0.002_hybrid | 0.046891 | 1.530394 | — | True | 2 | fault |
| opposed_kv0.1_dt0.002_stationary_stop | 0.180211 | -3.140947 | — | True | 0 | stationary_stop |
| opposed_kv0.1_dt0.002_hybrid | 0.139975 | -1.120781 | — | True | 1 | fault |
| heading_only_kv0.1_dt0.002_stationary_stop | 0.000070 | -1.199354 | — | True | 0 | stationary_stop |
| heading_only_kv0.1_dt0.002_hybrid | 0.023715 | -0.161369 | — | True | 1 | fault |
| wrap_kv0.1_dt0.002_stationary_stop | 0.084885 | 3.122239 | — | True | 0 | stationary_stop |
| wrap_kv0.1_dt0.002_hybrid | 0.045634 | -2.504512 | — | True | 2 | fault |
| near_kv0.1_dt0.002_stationary_stop | 0.015743 | 0.150646 | — | True | 0 | stationary_stop |
| near_kv0.1_dt0.002_hybrid | 0.016067 | 0.164703 | — | True | 2 | fault |
| at_goal_kv0.1_dt0.002_stationary_stop | 0.000070 | 0.000646 | 0.00 | False | 0 | stationary_stop |
| at_goal_kv0.1_dt0.002_hybrid | 0.000070 | 0.000646 | 0.00 | False | 1 | stop |
| handoff_straight_kv0.1_dt0.002_stationary_stop | 0.048742 | 0.032003 | — | True | 0 | stationary_stop |
| handoff_straight_kv0.1_dt0.002_hybrid | 0.035746 | 0.193149 | — | True | 2 | fault |
| handoff_waypoint_kv0.1_dt0.002_stationary_stop | 0.050117 | -0.015723 | — | True | 0 | stationary_stop |
| handoff_waypoint_kv0.1_dt0.002_hybrid | 0.034194 | -0.132342 | — | True | 2 | fault |
| ahead_kv0.1_dt0.001_stationary_stop | 0.200065 | 0.000646 | — | True | 0 | stationary_stop |
| ahead_kv0.1_dt0.001_hybrid | 0.039391 | 0.000340 | — | True | 1 | fault |
| behind_kv0.1_dt0.001_stationary_stop | 0.149935 | 0.000646 | — | True | 0 | stationary_stop |
| behind_kv0.1_dt0.001_hybrid | 0.117853 | 2.018832 | — | True | 1 | fault |
| lateral_kv0.1_dt0.001_stationary_stop | 0.150044 | 1.000646 | — | True | 0 | stationary_stop |
| lateral_kv0.1_dt0.001_hybrid | 0.046885 | 1.530860 | — | True | 2 | fault |
| opposed_kv0.1_dt0.001_stationary_stop | 0.180213 | -3.140947 | — | True | 0 | stationary_stop |
| opposed_kv0.1_dt0.001_hybrid | 0.139943 | -1.122761 | — | True | 1 | fault |
| heading_only_kv0.1_dt0.001_stationary_stop | 0.000068 | -1.199354 | — | True | 0 | stationary_stop |
| heading_only_kv0.1_dt0.001_hybrid | 0.023728 | -0.161370 | — | True | 1 | fault |
| wrap_kv0.1_dt0.001_stationary_stop | 0.084883 | 3.122239 | — | True | 0 | stationary_stop |
| wrap_kv0.1_dt0.001_hybrid | 0.045634 | -2.504692 | — | True | 2 | fault |
| near_kv0.1_dt0.001_stationary_stop | 0.015745 | 0.150646 | — | True | 0 | stationary_stop |
| near_kv0.1_dt0.001_hybrid | 0.016058 | 0.165177 | — | True | 2 | fault |
| at_goal_kv0.1_dt0.001_stationary_stop | 0.000068 | 0.000646 | 0.00 | False | 0 | stationary_stop |
| at_goal_kv0.1_dt0.001_hybrid | 0.000068 | 0.000646 | 0.00 | False | 1 | stop |
| handoff_straight_kv0.1_dt0.001_stationary_stop | 0.049350 | 0.030084 | — | True | 0 | stationary_stop |
| handoff_straight_kv0.1_dt0.001_hybrid | 0.036320 | 0.211905 | — | True | 2 | fault |
| handoff_waypoint_kv0.1_dt0.001_stationary_stop | 0.049875 | -0.021846 | — | True | 0 | stationary_stop |
| handoff_waypoint_kv0.1_dt0.001_hybrid | 0.033748 | -0.119009 | — | True | 1 | fault |
| ahead_kv1_dt0.002_stationary_stop | 0.200062 | 0.000577 | — | True | 0 | stationary_stop |
| ahead_kv1_dt0.002_hybrid | 0.011787 | -0.001137 | 4.85 | False | 2 | stop |
| behind_kv1_dt0.002_stationary_stop | 0.149938 | 0.000577 | — | True | 0 | stationary_stop |
| behind_kv1_dt0.002_hybrid | 0.025093 | -1.769474 | — | True | 9 | fault |
| lateral_kv1_dt0.002_stationary_stop | 0.150042 | 1.000577 | — | True | 0 | stationary_stop |
| lateral_kv1_dt0.002_hybrid | 0.025320 | 0.294880 | — | True | 9 | fault |
| opposed_kv1_dt0.002_stationary_stop | 0.180216 | -3.141016 | — | True | 0 | stationary_stop |
| opposed_kv1_dt0.002_hybrid | 0.018488 | -0.038573 | 12.75 | False | 4 | stop |
| heading_only_kv1_dt0.002_stationary_stop | 0.000064 | -1.199423 | — | True | 0 | stationary_stop |
| heading_only_kv1_dt0.002_hybrid | 0.016161 | -0.038934 | 3.55 | False | 2 | stop |
| wrap_kv1_dt0.002_stationary_stop | 0.084883 | 3.122169 | — | True | 0 | stationary_stop |
| wrap_kv1_dt0.002_hybrid | 0.025313 | -1.573878 | — | True | 9 | fault |
| near_kv1_dt0.002_stationary_stop | 0.015748 | 0.150577 | — | True | 0 | stationary_stop |
| near_kv1_dt0.002_hybrid | 0.017449 | 0.038506 | 6.55 | False | 3 | stop |
| at_goal_kv1_dt0.002_stationary_stop | 0.000064 | 0.000577 | 0.00 | False | 0 | stationary_stop |
| at_goal_kv1_dt0.002_hybrid | 0.000064 | 0.000577 | 0.00 | False | 1 | stop |
| handoff_straight_kv1_dt0.002_stationary_stop | 0.021494 | 0.024048 | 0.10 | False | 0 | stationary_stop |
| handoff_straight_kv1_dt0.002_hybrid | 0.013471 | 0.039137 | 2.90 | False | 4 | stop |
| handoff_waypoint_kv1_dt0.002_stationary_stop | 0.019672 | -0.008002 | 0.10 | False | 0 | stationary_stop |
| handoff_waypoint_kv1_dt0.002_hybrid | 0.011873 | -0.039643 | 1.45 | False | 3 | stop |
| ahead_kv1_dt0.001_stationary_stop | 0.200060 | 0.000577 | — | True | 0 | stationary_stop |
| ahead_kv1_dt0.001_hybrid | 0.011897 | -0.001687 | 4.85 | False | 2 | stop |
| behind_kv1_dt0.001_stationary_stop | 0.149940 | 0.000577 | — | True | 0 | stationary_stop |
| behind_kv1_dt0.001_hybrid | 0.025308 | -1.810868 | — | True | 9 | fault |
| lateral_kv1_dt0.001_stationary_stop | 0.150040 | 1.000577 | — | True | 0 | stationary_stop |
| lateral_kv1_dt0.001_hybrid | 0.025123 | 0.418066 | — | True | 9 | fault |
| opposed_kv1_dt0.001_stationary_stop | 0.180218 | -3.141016 | — | True | 0 | stationary_stop |
| opposed_kv1_dt0.001_hybrid | 0.018237 | -0.039215 | 12.65 | False | 4 | stop |
| heading_only_kv1_dt0.001_stationary_stop | 0.000062 | -1.199423 | — | True | 0 | stationary_stop |
| heading_only_kv1_dt0.001_hybrid | 0.017842 | -0.038743 | 3.60 | False | 2 | stop |
| wrap_kv1_dt0.001_stationary_stop | 0.084881 | 3.122170 | — | True | 0 | stationary_stop |
| wrap_kv1_dt0.001_hybrid | 0.025393 | -1.600338 | — | True | 9 | fault |
| near_kv1_dt0.001_stationary_stop | 0.015750 | 0.150577 | — | True | 0 | stationary_stop |
| near_kv1_dt0.001_hybrid | 0.017195 | 0.038590 | 7.05 | False | 3 | stop |
| at_goal_kv1_dt0.001_stationary_stop | 0.000062 | 0.000577 | 0.00 | False | 0 | stationary_stop |
| at_goal_kv1_dt0.001_hybrid | 0.000062 | 0.000577 | 0.00 | False | 1 | stop |
| handoff_straight_kv1_dt0.001_stationary_stop | 0.022825 | 0.028775 | 0.10 | False | 0 | stationary_stop |
| handoff_straight_kv1_dt0.001_hybrid | 0.014267 | 0.039835 | 3.10 | False | 4 | stop |
| handoff_waypoint_kv1_dt0.001_stationary_stop | 0.019830 | -0.005204 | 0.10 | False | 0 | stationary_stop |
| handoff_waypoint_kv1_dt0.001_hybrid | 0.011906 | -0.039458 | 0.35 | False | 2 | stop |

![Terminal errors](errors.png)

![Handoff and actuator response](handoff.png)
