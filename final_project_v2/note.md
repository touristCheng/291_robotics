### Random factors:

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
One specific setting in the baseline is having (X=np.pi, Y=np.pi/180*120, Z=?). This pose can hold the boxes in the spade.  




### Running record
commit 4e28113  
np.random.seed(0)  
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



commit 22809f3  
np.random.seed(0)  
running for 145.47 seconds, success on 10/10 boxes  
success_rate=100.00%, efficiency=4.12/minute  
running for 345.47 seconds, success on 17/20 boxes  
success_rate=85.00%, efficiency=2.95/minute  
running for 545.47 seconds, success on 25/30 boxes  
success_rate=83.33%, efficiency=2.75/minute  
running for 691.27 seconds, success on 27/40 boxes  
success_rate=67.50%, efficiency=2.34/minute  
running for 891.27 seconds, success on 32/50 boxes  
success_rate=64.00%, efficiency=2.15/minute  
running for 1084.70 seconds, success on 41/60 boxes  
success_rate=68.33%, efficiency=2.27/minute  
running for 1284.70 seconds, success on 45/70 boxes  
success_rate=64.29%, efficiency=2.10/minute  
running for 1484.70 seconds, success on 54/80 boxes  
success_rate=67.50%, efficiency=2.18/minute  
running for 1629.05 seconds, success on 60/90 boxes  
success_rate=66.67%, efficiency=2.21/minute  
running for 1829.05 seconds, success on 64/100 boxes  
success_rate=64.00%, efficiency=2.10/minute  
running for 1976.66 seconds, success on 74/110 boxes  
success_rate=67.27%, efficiency=2.25/minute  
running for 2000.00 seconds, success on 74/120 boxes  
success_rate=61.67%, efficiency=2.22/minute  
running for 2000.00 seconds, success on 74/120 boxes  
success_rate=61.67%, efficiency=2.22/minute  

With server evaluation: 
success_rate=75.45454545454545	efficiency=2.489999881731352



commit d2f2a01
np.random.seed(0)  
running for 200.00 seconds, success on 1/10 boxes  
success_rate=10.00%, efficiency=0.30/minute  
running for 400.00 seconds, success on 7/20 boxes  
success_rate=35.00%, efficiency=1.05/minute  
running for 592.80 seconds, success on 16/30 boxes  
success_rate=53.33%, efficiency=1.62/minute  
running for 737.17 seconds, success on 25/40 boxes  
success_rate=62.50%, efficiency=2.03/minute  
running for 937.17 seconds, success on 30/50 boxes  
success_rate=60.00%, efficiency=1.92/minute  
running for 1137.17 seconds, success on 39/60 boxes  
success_rate=65.00%, efficiency=2.06/minute  
running for 1337.17 seconds, success on 45/70 boxes  
success_rate=64.29%, efficiency=2.02/minute  
running for 1537.17 seconds, success on 46/80 boxes  
success_rate=57.50%, efficiency=1.80/minute  
running for 1680.92 seconds, success on 56/90 boxes  
success_rate=62.22%, efficiency=2.00/minute  
running for 1880.92 seconds, success on 62/100 boxes  
success_rate=62.00%, efficiency=1.98/minute  
running for 2000.00 seconds, success on 70/110 boxes  
success_rate=63.64%, efficiency=2.10/minute  
running for 2000.00 seconds, success on 70/110 boxes  
success_rate=63.64%, efficiency=2.10/minute  

With server evaluation: 
success_rate=61.66666666666667	efficiency=2.2199998945556634
	


commit 
np.random.seed(0)  
running for 141.25 seconds, success on 9/10 boxes  
success_rate=90.00%, efficiency=3.82/minute  
running for 341.25 seconds, success on 14/20 boxes  
success_rate=70.00%, efficiency=2.46/minute  
running for 484.16 seconds, success on 21/30 boxes  
success_rate=70.00%, efficiency=2.60/minute  
running for 684.16 seconds, success on 26/40 boxes  
success_rate=65.00%, efficiency=2.28/minute  
running for 884.16 seconds, success on 26/50 boxes  
success_rate=52.00%, efficiency=1.76/minute  
running for 1084.16 seconds, success on 35/60 boxes  
success_rate=58.33%, efficiency=1.94/minute  
running for 1284.16 seconds, success on 41/70 boxes  
success_rate=58.57%, efficiency=1.92/minute  
running for 1484.16 seconds, success on 49/80 boxes  
success_rate=61.25%, efficiency=1.98/minute  
running for 1684.16 seconds, success on 56/90 boxes  
success_rate=62.22%, efficiency=2.00/minute  
running for 1884.16 seconds, success on 63/100 boxes  
success_rate=63.00%, efficiency=2.01/minute  
running for 2000.00 seconds, success on 65/110 boxes  
success_rate=59.09%, efficiency=1.95/minute  
running for 2000.00 seconds, success on 65/110 boxes  
success_rate=59.09%, efficiency=1.95/minute  

With server evaluation: 
success_rate=89.23076923076924	efficiency=3.4799998347088774