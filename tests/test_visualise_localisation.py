"""No GUI or motion needed: bounded detached-display contract tests."""
import inspect
import queue
import unittest
from unittest.mock import Mock,patch
import numpy as np
from integration.mujoco_owner import Measurement
from integration.visualise_localisation import latest_frame,display_payload,FrameDisplay,display_worker


class VisualisationTests(unittest.TestCase):
    def packet(self,rgb=True):
        return Measurement([0,0],[np.inf],np.zeros((4,4,3),np.uint8) if rgb else None,.05,10.,0,'fixture')
    def test_missing_stale_and_future_frame(self):
        self.assertIsNone(display_payload(self.packet(False),{},10.,1.))
        self.assertIsNone(display_payload(self.packet(),{},11.,1.))
        self.assertIsNone(display_payload(self.packet(),{},9.,1.))
    def test_detached_image_and_diagnostic_metadata(self):
        packet=self.packet();value=display_payload(packet,{'observed_ids':[1]},10.,1.)
        value['rgb'][:]=255
        self.assertEqual(packet.rgb.max(),0)
        self.assertEqual(value['sequence'],0);self.assertEqual(value['epoch'],'fixture')
    def test_bounded_latest_frame_nonblocking(self):
        channel=queue.Queue(1)
        self.assertTrue(latest_frame(channel,'old'))
        self.assertTrue(latest_frame(channel,'new'))
        self.assertEqual(channel.qsize(),1);self.assertEqual(channel.get_nowait(),'new')
        full=Mock();full.put_nowait.side_effect=queue.Full;full.get_nowait.side_effect=queue.Empty
        self.assertFalse(latest_frame(full,'frame'))
    def test_shutdown_idempotent_and_unresponsive_gui_terminated(self):
        display=FrameDisplay.__new__(FrameDisplay)
        display.closed=False;display.stop=Mock();display.worker=Mock();display.channel=Mock()
        display.worker.is_alive.return_value=True
        display.close();display.close()
        display.stop.set.assert_called_once();display.worker.terminate.assert_called_once()
        display.channel.close.assert_called_once()
    def test_worker_exits_without_owner_or_actuator_access(self):
        stop=Mock();stop.is_set.return_value=True
        with patch('cv2.destroyAllWindows') as close:display_worker(queue.Queue(1),stop,1.)
        close.assert_called_once()
        source=inspect.getsource(display_worker)
        for token in ('OwnerSession','client','MjData','mujoco','submit(','Evaluation'):
            self.assertNotIn(token,source)


if __name__=='__main__':unittest.main()
