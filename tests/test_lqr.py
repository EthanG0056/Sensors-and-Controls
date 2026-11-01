import unittest
import numpy as np
from scipy.integrate import solve_ivp
from control.lqr_controller import (tracking_error,error_dynamics,linearise,discretise,
    controllability,lqr_gain,LQRTracker,CONFIGS,wheel_commands)
from control.trajectory import reference,DURATIONS,Reference


class Mathematics(unittest.TestCase):
    def test_jacobians(self):
        h=1e-6
        for v,w in [(.08,0),(.08,.3),(0,.4),(0,0),(-.08,-.3)]:
            a,b=linearise(v,w)
            aj=np.column_stack([(error_dynamics(np.eye(3)[j]*h,np.zeros(2),(v,w))-error_dynamics(-np.eye(3)[j]*h,np.zeros(2),(v,w)))/(2*h) for j in range(3)])
            bj=np.column_stack([(error_dynamics(np.zeros(3),np.eye(2)[j]*h,(v,w))-error_dynamics(np.zeros(3),-np.eye(2)[j]*h,(v,w)))/(2*h) for j in range(2)])
            np.testing.assert_allclose(a,aj,atol=1e-10); np.testing.assert_allclose(b,bj,atol=1e-10)

    def test_body_error_sign_and_nonlinear_derivative(self):
        pose=np.array([.2,-.1,.4]); target=np.array([.4,.3,.7]); u=np.array([.1,.2]); ur=np.array([.08,.3]); h=1e-7
        derivative=lambda p,c:np.array([c[0]*np.cos(p[2]),c[0]*np.sin(p[2]),c[1]])
        numerical=(tracking_error(pose+h*derivative(pose,u),target+h*derivative(target,ur))-tracking_error(pose,target))/h
        np.testing.assert_allclose(numerical,error_dynamics(tracking_error(pose,target),u-ur,ur),atol=1e-7)
        np.testing.assert_allclose(tracking_error([0,0,np.pi/2],[0,1,np.pi/2]),[1,0,0],atol=1e-12)

    def test_controllability(self):
        for v,w,rank in [(.08,0,3),(.08,.3,3),(0,.4,3),(0,0,2)]:
            a,b=linearise(v,w); self.assertEqual(controllability(a,b)[0],rank)
            self.assertEqual(controllability(*discretise(a,b,.05))[0],rank)

    def test_discretisation(self):
        a,b=linearise(.1,.3); ad,bd=discretise(a,b,.05)
        x=np.array([.1,-.1,.2]); u=np.array([.02,-.03])
        numerical=solve_ivp(lambda t,y:a@y+b@u,[0,.05],x,rtol=1e-11,atol=1e-13).y[:,-1]
        np.testing.assert_allclose(ad@x+bd@u,numerical,atol=1e-11)
        a,b=linearise(0,0); ad,bd=discretise(a,b,.05)
        np.testing.assert_allclose(ad,np.eye(3)); np.testing.assert_allclose(bd,.05*b)

    def test_riccati_and_stability(self):
        for config in CONFIGS.values():
            for v,w in [(.08,0),(.08,.3),(0,.4),(.001,0)]:
                k,p,a,b=lqr_gain(v,w,.05,config.q,config.r)
                residual=a.T@p@a-p-a.T@p@b@k+np.diag(config.q)
                self.assertLess(np.linalg.norm(residual),1e-6)
                self.assertLess(max(abs(np.linalg.eigvals(a-b@k))),1)
        with self.assertRaises(ValueError): lqr_gain(0,0,.05,(1,1,1),(1,1))

    def test_saturation(self):
        for v,w in [(.5,2),(.5,-2),(-.5,2),(0,5)]:
            wheels,body,sat=wheel_commands(v,w)
            self.assertTrue(sat); self.assertLessEqual(max(abs(wheels)),6.67+1e-12)
            np.testing.assert_allclose([.033*sum(wheels)/2,.033*(wheels[1]-wheels[0])/.16],body)
            self.assertEqual(np.sign(body[1]),np.sign(w))
            self.assertAlmostEqual(body[0]*w,body[1]*v)

    def test_references_feasible(self):
        for name,duration in DURATIONS.items():
            for t in np.linspace(.01,duration-.01,41):
                r=reference(name,t); h=1e-5
                before,after=reference(name,t-h),reference(name,t+h)
                diff=(after.pose-before.pose)/(2*h)
                # angular wrapping is absent away from spline tangent discontinuities here.
                np.testing.assert_allclose(diff,[r.input[0]*np.cos(r.pose[2]),r.input[0]*np.sin(r.pose[2]),r.input[1]],atol=1e-7)
                self.assertFalse(wheel_commands(*r.input)[2])

    def test_ideal_kinematic_recovery(self):
        tracker=LQRTracker(); pose=np.array([-.05,.04,.15])
        initial=np.linalg.norm(tracking_error(pose,reference('straight',0).pose))
        for t in np.arange(0,40,.05):
            command,_=tracker.command(pose,Reference(np.array([.08*t,0.,0.]),np.array([.08,0.]))); _,u,_=wheel_commands(*command)
            pose=solve_ivp(lambda _,p:[u[0]*np.cos(p[2]),u[0]*np.sin(p[2]),u[1]],[0,.05],pose,rtol=1e-9,atol=1e-11).y[:,-1]
        self.assertLess(np.linalg.norm(tracking_error(pose,np.array([3.2,0.,0.]))),.01*initial)
        command,mode=tracker.command([.1,.1,.1],reference('stationary',0))
        np.testing.assert_array_equal(command,[0,0]); self.assertEqual(mode,'stationary_stop')


if __name__=='__main__': unittest.main()
