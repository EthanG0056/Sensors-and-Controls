"""Stationary camera display: python -B -m integration.visualise_localisation.

Only run_plan orchestrates the existing owner and ZERO leases. The display worker
receives detached RGB frames, never owner/client/model/data/Evaluation. No viewer,
control callback or interactive arming is provided. GUI stalls drop frames.
"""
import argparse
import multiprocessing as mp
import queue
import time
import numpy as np
from integration.tests.validate_localisation import PLANS, run_plan


def latest_frame(channel, item):
    """Best effort nonblocking bounded delivery; never wait for a GUI consumer."""
    try: channel.put_nowait(item);return True
    except queue.Full:
        try:channel.get_nowait()
        except queue.Empty:pass
        try:channel.put_nowait(item);return True
        except queue.Full:return False


def display_payload(packet, info, now, max_age):
    if packet.rgb is None:return None
    if (not np.isfinite(now) or not np.isfinite(packet.sample_monotonic)
            or packet.sample_monotonic>now or now-packet.sample_monotonic>=max_age):return None
    return dict(rgb=packet.rgb.copy(),acquired=packet.sample_monotonic,
                sequence=packet.sequence,epoch=packet.epoch,info=dict(info))


def display_worker(channel, stop, max_age):
    import cv2
    title='Stationary localisation — ZERO commands; not navigation'
    last_acquired=None
    try:
        while not stop.is_set():
            try:item=channel.get(timeout=.05)
            except queue.Empty:
                if last_acquired is not None and time.monotonic()-last_acquired>=max_age:
                    cv2.destroyAllWindows();last_acquired=None
                if cv2.waitKey(1)&255 in (27,ord('q')):stop.set()
                continue
            if time.monotonic()-item['acquired']>=max_age:
                cv2.destroyAllWindows() # do not leave a stale image looking live
                continue
            last_acquired=item['acquired']
            image=cv2.cvtColor(item['rgb'],cv2.COLOR_RGB2BGR)
            label=f"seq {item['sequence']} IDs {item['info']['observed_ids']} stationary; quality NOT authorised"
            cv2.putText(image,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            pose=item['info'].get('estimate')
            if pose is not None:
                text='EKF x %.3f y %.3f yaw %.3f rad' % tuple(pose)
                cv2.putText(image,text,(8,40),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
            cv2.imshow(title,image)
            if cv2.waitKey(1)&255 in (27,ord('q')):stop.set()
    finally:cv2.destroyAllWindows()


class FrameDisplay:
    def __init__(self, max_age=1.):
        if not np.isfinite(max_age) or max_age<=0:raise ValueError('Positive display-only age required')
        self.max_age=max_age;ctx=mp.get_context('spawn')
        self.channel=ctx.Queue(1);self.stop=ctx.Event()
        self.worker=ctx.Process(target=display_worker,args=(self.channel,self.stop,max_age))
        self.closed=False;self.dropped=0
    def start(self):self.worker.start();return self
    def publish(self,packet,info):
        if self.closed or self.stop.is_set() or not self.worker.is_alive():
            raise InterruptedError('Display closed')
        item=display_payload(packet,info,time.monotonic(),self.max_age)
        if item is None or not latest_frame(self.channel,item):self.dropped+=1
    def close(self):
        if self.closed:return
        self.closed=True;self.stop.set();self.worker.join(timeout=1.)
        if self.worker.is_alive():self.worker.terminate();self.worker.join(timeout=1.)
        self.channel.cancel_join_thread();self.channel.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args();display=None
    try:
        if not args.headless:display=FrameDisplay().start()
        summary,_=run_plan(PLANS['stationary'],None if display is None else display.publish)
        import json
        print(json.dumps(summary,sort_keys=True))
        return int(not summary['completed'] or not summary['stop_confirmed'] or not summary['all_recorded_writes_zero'])
    finally:
        if display is not None:display.close()


if __name__=='__main__':raise SystemExit(main())
