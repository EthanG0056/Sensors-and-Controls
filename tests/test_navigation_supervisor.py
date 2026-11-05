"""Synthetic navigation regression fixtures, not autonomous validation."""
import unittest
import numpy as np
from integration.navigation_supervisor import NavigationSupervisor


class Normal:
    end_time = 12.
    def __init__(self): self.calls = []
    def command(self, pose, ref, time):
        self.calls.append(time)
        return np.zeros(2), np.zeros(2), 'tracking_lqr', False


class NavigationSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.normal = Normal()
        self.nav = NavigationSupervisor(self.normal, [2.624, .4])

    def tick(self, k, obstacle=None, pose=(0., 0., 0.)):
        self.nav.update(pose, obstacle, k*.05)
        return self.nav.command(np.asarray(pose), self.nav.reference(0), k*.05)

    def test_exclusive_phases_and_reference_resynchronisation(self):
        self.tick(0)
        self.assertEqual(len(self.normal.calls), 1)
        for k in range(1, 201):
            result = self.tick(k, (.1, .2), (.8, .1, .2))
            self.assertEqual(result[2], 'tracking_avoidance')
            self.assertLessEqual(max(abs(result[1])), 6.67)
            self.assertEqual(len(self.normal.calls), 1)
            expected = 'turn_out' if k < 41 else 'drive_past' if k < 141 else 'rejoin'
            self.assertEqual(self.nav.phase, expected)
        self.tick(201, None, (.8, .1, .2))
        self.assertIsNone(self.nav.phase)
        self.assertEqual(len(self.normal.calls), 2)
        self.assertAlmostEqual(self.nav.navigation_time, .8/.08+.05)
        self.assertAlmostEqual(self.nav.reference(0).pose[0], .804)

    def test_turn_signs_and_inverse_kinematics(self):
        for angle, sign in [(.1,-1),(-.1,1),(0.,1)]:
            nav=NavigationSupervisor(Normal(), [3.,0.])
            nav.update([0.,0.,0.],(.1,angle),0.)
            body,wheels,_,_=nav.command(np.zeros(3),nav.reference(0),0.)
            self.assertAlmostEqual(body[1],sign*.35)
            np.testing.assert_allclose(body,[.033*sum(wheels)/2,.033*(wheels[1]-wheels[0])/.160])

    def test_invalid_input_never_invokes_normal_controller(self):
        for pose, obstacle, time in [([np.nan,0,0],None,0.),([0,0],None,0.),
                ([0,0,0],(np.nan,0),0.),([0,0,0],(-1,0),0.),
                ([0,0,0],(1,np.inf),0.),([0,0,0],None,np.nan)]:
            with self.assertRaises(ValueError): self.nav.update(pose,obstacle,time)
        self.assertEqual(self.normal.calls,[])

    def test_duplicate_skipped_and_out_of_order_intervals(self):
        for next_time in [0.,-.05,.1]:
            nav=NavigationSupervisor(Normal(),[3.,0.]);nav.update([0,0,0],None,0.)
            with self.assertRaises(ValueError): nav.update([0,0,0],None,next_time)

    def test_docking_trigger_is_latched_stop_not_terminal(self):
        self.nav.update([1.9,.4,0],None,0.)
        self.assertEqual(self.nav.stop_reason,'terminal_clearance_unavailable')
        with self.assertRaises(ValueError): self.nav.command(np.array([1.9,.4,0]),self.nav.reference(0),0.)
        self.assertEqual(self.normal.calls,[])

    def test_one_command_per_measurement_and_no_scheduled_terminal(self):
        self.nav.navigation_time=20.
        self.tick(0)
        self.assertLess(self.normal.calls[-1],12.)
        with self.assertRaises(ValueError): self.nav.command(np.zeros(3),self.nav.reference(0),0.)

    def test_invalid_normal_wheel_output_rejected(self):
        self.normal.command=lambda *args:(np.zeros(2),np.array([7.,0.]),'tracking_lqr',False)
        with self.assertRaises(ValueError): self.tick(0)


if __name__ == '__main__': unittest.main()
