## Random factors:

Box size: [0.01, 0.02]  (1cm~2cm)
Box location: x: [-0.1, 0.1] (-10cm~10cm) y: [-0.2, 0.2] (-20cm~20cm)  

Spade size: x: [1.4cm~1.6cm], y/z: [1cm~13cm] (not identical)  
(Maybe should directly skip when space is too small)  

Bin location: x: [-1.1m~-1.3m] y: [-0.1m~0.1m] z: 0.6m orientation [0~pi]  
Bin size: x: [15cm~25cm] y: [25cm~35cm] z: [35cm~45cm] thickness 1cm~2cm

### Orientation of the spade:
From the top camera / rendering view, we have  
X-axis: image plane heading up  
Y-axis: image plane pointing right  
Z-axis: pointing outside of the image plane  
And the euler angle is defined by extrinsic rotations. So the rotation axis doesn't change with the pose change.  
If we want to set the pose from 3D euler angle, the strategy should be first fixing the angle of X,Y direction. The X should always be either 0 or np.pi. When X=0, the spade is heading up and when X=np.pi the spade is heading down. The Y decide the tilting angle which is important to have object holding inside.  
One specific setting in the baseline is having (X=np.pi, Y=np.pi/180*120, Z=?).  
Notice the origin 






After picking up:
left
[ 1.5254265   0.61441237 -0.05420243 -2.2665877  -1.5219445   2.1323466  -2.6600673 ]
Pose([-0.0462755, 0.15704, 0.668106], [-0.349973, -0.619102, 0.604926, -0.358186])
right
[-1.433983   0.6611327  0.0517656 -2.1967015  1.6250713  1.959965  -2.031935 ]
Pose([-0.037062, -0.0918611, 0.672602], [-0.420247, 0.578114, 0.571023, 0.403868])