import unittest
import numpy as np
from control.terminal_controller import TerminalController,TrajectoryTerminalController,TerminalParameters
from control.trajectory import Reference
from control.lqr_controller import wheel_commands,wrap


class TerminalTests(unittest.TestCase):
    def test_origin_heading_and_wrap(self):
        c=TerminalController([0,0,-np.pi+.02])
        u,mode=c.command([0,0,np.pi-.1])
        self.assertEqual(mode,'heading');self.assertEqual(u[0],0);self.assertGreater(u[1],0)

    def test_behind_rotates_without_reversing(self):
        c=TerminalController([-1,0,0]);u,m=c.command([0,0,0])
        self.assertEqual(m,'bearing');self.assertEqual(u[0],0);self.assertGreater(abs(u[1]),0)

    def test_hysteresis(self):
        c=TerminalController([1,0,0])
        c.command([0,0,0]);self.assertEqual(c.mode,'approach')
        c.command([0,0,.3]);self.assertEqual(c.mode,'approach')
        c.command([0,0,.5]);self.assertEqual(c.mode,'bearing')
        c.command([0,0,.3]);self.assertEqual(c.mode,'bearing')
        c.command([0,0,.1]);self.assertEqual(c.mode,'approach')

    def test_stop_dwell_latch_and_fault(self):
        c=TerminalController([0,0,0])
        for _ in range(10): c.command([0,0,0])
        self.assertNotEqual(c.mode,'stop')
        for _ in range(20):c.command([0,0,0])
        self.assertEqual(c.mode,'stop')
        np.testing.assert_array_equal(c.command([1,1,1])[0],[0,0])
        c=TerminalController([1,0,0]);c.command([np.nan,0,0]);self.assertEqual(c.mode,'fault')
        c=TerminalController([1,0,0],TerminalParameters(timeout=1.))
        for _ in range(30):c.command([0,0,0])
        self.assertEqual(c.mode,'fault')

    def test_coasting_drift_recovery_and_retry_budget(self):
        c=TerminalController([0,0,0])
        for _ in range(4):
            c.command([0,0,.2]);c.command([.03,0,.2])
        self.assertEqual(c.mode,'fault')
        c=TerminalController([0,0,0]);c.command([0,0,0]);c.command([0,0,.08])
        self.assertEqual(c.mode,'heading')

    def test_one_way_handoff(self):
        ref=Reference(np.zeros(3),np.array([.08,0.]))
        c=TrajectoryTerminalController([0,0,0],1.)
        self.assertTrue(c.command([0,0,0],ref,0)[2].startswith('tracking'))
        self.assertFalse(c.command([0,0,0],ref,1.)[2].startswith('tracking'))
        with self.assertRaises(ValueError):c.command([0,0,0],ref,.9)
        self.assertTrue(c.handed_off)

    def test_limits_and_speed_reduction(self):
        for distance in (.02,.04,.2,1.):
            c=TerminalController([distance,0,0]);u,_=c.command([0,0,0])
            self.assertAlmostEqual(u[0],min(.08,.8*distance))
            wheels,_,sat=wheel_commands(*u);self.assertFalse(sat);self.assertLessEqual(max(abs(wheels)),6.67)

    def test_ideal_capture_multiple_poses(self):
        for initial in ([.2,.1,2.],[-.1,.1,-2.],[0,0,3.],[.003,0,.01]):
            pose=np.array(initial,dtype=float);c=TerminalController([0,0,.3])
            for _ in range(900):
                u,_=c.command(pose)
                pose+=.05*np.array([u[0]*np.cos(pose[2]),u[0]*np.sin(pose[2]),u[1]])
            self.assertEqual(c.mode,'stop')
            self.assertLess(np.linalg.norm(pose[:2]),.025)
            self.assertLess(abs(wrap(pose[2]-.3)),.04)


if __name__=='__main__':unittest.main()
