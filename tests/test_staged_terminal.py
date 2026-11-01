import unittest
import numpy as np
from control.staged_terminal_controller import StagedTerminalController
from control.lqr_controller import wrap


class StagedTests(unittest.TestCase):
    def test_stage_selection_and_one_way_completion(self):
        c=StagedTerminalController([0,0,0])
        _,mode=c.command([.2,0,0]);self.assertTrue(mode.startswith('stage_'))
        c.command([-.12,0,np.pi]);self.assertIsNone(c.stage)
        c.command([-.11,0,0]);self.assertIsNone(c.stage)
        c=StagedTerminalController([0,0,0]);_,mode=c.command([-.2,0,0])
        self.assertFalse(mode.startswith('stage_'))

    def test_invalid_and_total_timeout(self):
        c=StagedTerminalController([0,0,0]);c.command([np.inf,0,0]);self.assertEqual(c.mode,'fault')
        c=StagedTerminalController([0,0,0])
        for _ in range(910):c.command([.2,0,0])
        self.assertEqual(c.mode,'fault')

    def test_ideal_staged_capture(self):
        for initial in ([.15,0,0],[0,.15,-1],[.06,-.06,-np.pi+.02]):
            c=StagedTerminalController([0,0,0]);pose=np.array(initial,dtype=float)
            for _ in range(1000):
                u,mode=c.command(pose)
                pose+=.05*np.array([u[0]*np.cos(pose[2]),u[0]*np.sin(pose[2]),u[1]])
            self.assertEqual(mode,'stop');self.assertLess(np.linalg.norm(pose[:2]),.025)
            self.assertLess(abs(wrap(pose[2])),.07)


if __name__=='__main__':unittest.main()
