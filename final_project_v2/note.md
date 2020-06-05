## Random factors:

Box size: [0.01, 0.02]  (1cm 2cm)
Box location: x: [-0.1, 0.1] (-10cm 10cm) y: [-0.2, 0.2] (-20cm 20cm)  

Spade size: x: [1.4cm 1.6cm], y/z: [1cm 13cm] (not identical)  
(Maybe should directly skip when space is too small)  

Bin location: x: [-1.1m -1.3m] y: [-0.1m 0.1m] z: 0.6m orientation [0 pi]  
Bin size: x: [15cm 25cm] y: [25cm 35cm] z: [35cm 45cm] thickness 1cm 2cm

### Orientation of the spade:
From the top camera / rendering view, we have  
X-axis: image plane heading up  
Y-axis: image plane pointing right  
Z-axis: pointing outside of the image plane  
And the euler angle is defined by extrinsic rotations. So the rotation axis doesn't change with the pose change.  
If we want to set the pose from 3D euler angle, the strategy should be first fixing the angle of X,Y direction. The X should always be either 0 or np.pi. When X=0, the spade is heading up and when X=np.pi the spade is heading down. The Y decide the tilting angle which is important to have object holding inside.  
One specific setting in the baseline is having (X=np.pi, Y=np.pi/180*120, Z=?).  
Notice the origin 



### Running record
running for 163.51 seconds, success on 10/10 boxes  
success_rate=100.00%, efficiency=3.67/minute  
running for 363.51 seconds, success on 15/20 boxes  
success_rate=75.00%, efficiency=2.48/minute  
running for 563.51 seconds, success on 17/30 boxes  
success_rate=56.67%, efficiency=1.81/minute  
running for 763.51 seconds, success on 25/40 boxes  
success_rate=62.50%, efficiency=1.96/minute  
running for 963.51 seconds, success on 32/50 boxes  
success_rate=64.00%, efficiency=1.99/minute  
running for 1163.51 seconds, success on 40/60 boxes  
success_rate=66.67%, efficiency=2.06/minute  
running for 1324.94 seconds, success on 49/70 boxes  
success_rate=70.00%, efficiency=2.22/minute  
running for 1524.94 seconds, success on 55/80 boxes  
success_rate=68.75%, efficiency=2.16/minute  
running for 1724.94 seconds, success on 61/90 boxes  
success_rate=67.78%, efficiency=2.12/minute  
running for 1835.36 seconds, success on 71/100 boxes  
success_rate=71.00%, efficiency=2.32/minute  
running for 2000.00 seconds, success on 76/110 boxes  
success_rate=69.09%, efficiency=2.28/minute  
running for 2000.00 seconds, success on 76/110 boxes  
success_rate=69.09%, efficiency=2.28/minute  

With server evaluation:  
success_rate=70.83333333333334	efficiency=2.549999878881505





