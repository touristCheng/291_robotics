# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from .stacking_env import StackingEnv
import sapien.core as sapien
import numpy as np
from typing import List, Tuple, Sequence

#######################################
# === Some useful functions below === #
#######################################
import cv2

def Rp2T(R,p):
    """
    Combine rotation and translation to SE(3)
    Args:
        R: SO(3) matrix
        t: 3d vector
    Returns:
        T: SE(3) matrix
    """ 
    T = np.eye(4)    
    T[:3,:3] = R 
    T[:3,3] = np.squeeze(p)
    return T

def SE3inverse(T):
    """
    The inverse matrix for a SE(3) matrix
    Args:
        T: SE(3) matrix
    Returns:
        T_inv: SE(3) matrix
    """     
    R = T[:3,:3]
    p = T[:3,3]
    R_inv = R.transpose()
    p_inv = -np.matmul(R_inv,p)
    T_inv = np.eye(4)
    T_inv[:3,:3] = R_inv 
    T_inv[:3,3] = np.squeeze(p_inv)
    return T_inv

def SkewSymmetric(theta):
    """
    The skew-symmetric matrix for a 3D vector
    Args:
        theta: 3d vector
    Returns:
        R: 3x3 matrix
    """
    theta = np.squeeze(theta)        
    R = np.array([[0,-theta[2],theta[1]],
                [theta[2],0,-theta[0]],
                [-theta[1],theta[0],0]])        
    return R

def SkewSymmetricinv(R):
    """
    The inverse skew-symmetric from a matrix to a 3D vector
    Args:
        theta: 3d vector
    Returns:
        R: 3x3 matrix
    """        
    theta = np.array([R[2,1],R[0,2],R[1,0]])        
    return theta

def dist_SO3(rot1,rot2): 
    """
    The distance between two rotation matrices in SO(3)
    Args:
        rot1,rot2: 3x3 matrix in SO(3)
    Returns:
        dist: non-negative scalar
    """ 
    R_diff = np.matmul(rot1.transpose(),rot2)
    dist = np.arccos(max(-1,min(1,(np.trace(R_diff)-1)/2)))
    return dist

def SO32rotvec(rot):
    """
    The logarithm map for SO(3)
    Args:
        rot: 3x3 matrix in SO(3)
    Returns:
        theta: 3d rotation vector
    """     
    norm = np.arccos((np.trace(rot)-1)/2)
    direction = 1/(2*np.sin(norm))*np.array([rot[2,1]-rot[1,2],rot[0,2]-rot[2,0],rot[1,0]-rot[0,1]])
    theta = direction*norm
    theta_test = SkewSymmetricinv(norm/(2*np.sin(norm))*(rot-rot.transpose()))
    return theta

def J_L_inv(theta):
    """
    The inverse left Jacobian of a 3D rotation vector
    Args:
        theta: 3d vector
    Returns:
        R: 3x3 matrix
    """   
    R = np.eye(3) - 0.5*SkewSymmetric(theta) + ((1+np.cos(np.linalg.norm(theta)))/(np.linalg.norm(theta)- 1/(2*np.linalg.norm(theta)*np.sin(np.linalg.norm(theta))))**2)*np.matmul(SkewSymmetric(theta),SkewSymmetric(theta))
    return R

def rot2quatMinimal(R):
    """
    Convert SO(3) rotation matrix into a modified Rodrigues formula,
    which is the image part of the associated quaternion
    q = 2 * sin(theta/2) * v
    Args:
        R: 3x3 rotation matrix
    Returns:
        q: 3d vector
    """
    assert R.shape[0]==3 and R.shape[1]==3
    m00 = R[0,0]
    m01 = R[0,1]
    m02 = R[0,2]
    m10 = R[1,0]
    m11 = R[1,1]
    m12 = R[1,2]
    m20 = R[2,0]
    m21 = R[2,1]
    m22 = R[2,2]
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif (m11 > m22):
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = np.array([qx,qy,qz])
    q *= 2
    return q

def quatMinimal2rot(q):
    """
    Convert a modified Rodrigues formula back to SO(3) rotation matrix
    Inverse of rot2quatMinimal()
    Args:
        q: 3d vector
    Returns:
        R: 3x3 rotation matrix
    """   
    q = np.squeeze(q)        
    p = np.linalg.norm(q)**2
    #w = np.sqrt(1 - p)
    w = np.sqrt(4 - p)
    diag_p = p*np.eye(3)
    q = np.expand_dims(q,axis=1)
    #R = np.eye(3) - 2*diag_p + 2*(np.matmul(q,q.transpose()) + w*SkewSymmetric(q))
    R = np.eye(3) - 0.5*diag_p + 0.5*(np.matmul(q,q.transpose()) + w*SkewSymmetric(q))
    return R
#######################################
# === Some useful functions above === #
#######################################

class HW1Env(StackingEnv):
    def __init__(self, timestep: float):
        """Class for homework1

        Args:
            timestep: timestep to balance the precision and the speed of physical simulation
        """
        StackingEnv.__init__(self, timestep)

        self.near, self.far = 0.1, 100
        self.camera_link = self.scene.create_actor_builder().build(is_kinematic=True)
        self.gl_pose = sapien.Pose([0, 0, 0], [0.5, -0.5, 0.5, -0.5])
        self.camera_link.set_pose(sapien.Pose([1.2, 0.0, 0.8], [0, -0.258819, 0, 0.9659258]))
        self.camera = self.scene.add_mounted_camera('fixed_camera', self.camera_link,
                                                    sapien.Pose(), 1920, 1080,
                                                    np.deg2rad(50), np.deg2rad(50), self.near, self.far)

    def cam2base_gt(self) -> np.ndarray:
        """Get ground truth transformation of camera to base transformation

        Returns:
            Ground truth transformation from camera to robot base

        """
        camera_pose = self.camera_link.get_pose() * self.gl_pose
        base_pose = self.robot.get_root_pose()
        return self.pose2mat(base_pose.inv() * camera_pose)

    def board2cam_gt(self) -> np.ndarray:
        """This function is only used to debug your code.

        Note that the output of board2cam_gt() and get_current_marker_pose() will have slight difference,
        e.g. no more than 5cm, due to different definition of checker board frame
        So you can use this function to verify that whether you get the correct marker pose.

        Returns:
            Ground truth transformation from board base to camera

        """
        cam2base = self.cam2base_gt()
        board2base = self.pose2mat(self.robot.get_links()[-5].get_pose())
        return np.linalg.inv(cam2base) @ board2base

    def get_current_ee_pose(self) -> np.ndarray:
        """Get current end effector pose for calibration calculation

        Returns:
            Transformation from end effector to robot base

        """
        return self.pose2mat(self.end_effector.get_pose())

    def get_object_point_cloud(self, seg_id: int) -> np.ndarray:
        """Fetch the object point cloud given segmentation id

        For example, you can use this function to directly get the point cloud of a colored box and use it for further
        calculation.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()

        Returns:
            (3, n) dimension point cloud in the camera frame with(x, y, z) order

        """
        self.scene.update_render()
        self.camera.take_picture()
        camera_matrix = self.camera.get_camera_matrix()[:3, :3]
        gl_depth = self.camera.get_depth()
        y, x = np.where(gl_depth < 1)
        z = self.near * self.far / (self.far + gl_depth * (self.near - self.far))

        point_cloud = (np.dot(np.linalg.inv(camera_matrix),
                              np.stack([x, y, np.ones_like(x)] * z[y, x], 0)))

        seg_mask = self.camera.get_segmentation()[y, x]
        selected_index = np.nonzero(seg_mask == seg_id)[0]
        return point_cloud[:, selected_index]

    def is_grasped(self, obj: sapien.ActorBase) -> bool:
        """Check whether the robot gripper is holding the object firmly

        Args:
            obj: A object in SAPIEN simulation, e.g. a colored box

        Returns:
            Bool indicator whether the grasp is success

        """
        self.scene.set_timestep(1e-14)
        for i in range(2):
            self.step()
        contacts = self.scene.get_contacts()
        grasped = True
        for q in self.gripper_joints:
            touched = any({obj, q.get_child_link()} == {c.actor1, c.actor2} for c in contacts if c.separation < 2.5e-2)
            grasped = grasped and touched

        return grasped

    def close_gripper(self):
        """Set close gripper command. It will make effect after simulation step"""
        qpos = self.robot.get_qpos()
        qpos[-2:] = [0.02]
        self.robot.set_qpos(qpos)

    def grasp(self, qpos: Sequence[float]):
        """Grasp an object given joint pose. In robotics simulation, q is generalized coordinate which is often joint"""
        self.robot.set_qpos(qpos)
        for i in range(100):
            if self._windows:
                self.render()

        self.close_gripper()
        for i in range(100):
            if self._windows:
                self.render()

    def release_grasp(self):
        """As above, release the gripper"""
        qpos = self.robot.get_qpos()
        qpos[-2:] = [0.04]
        self.robot.set_qpos(qpos)

    @staticmethod
    def compute_ik(ee2base: np.ndarray) -> List[List[float]]:
        """Compute the inverse kinematics of franka panda robot.

        This function is provided to help do the inverse kinematics calculation.
        The output of this function is deterministic.
        It will return a list of solutions for the given cartesian pose.
        In practice, some solutions are not physically-plausible due to self collision.
        So in this homework, you may need to choose the free_joint_value and which solution to use by yourself.

        References:
            ikfast_pybind:
            ikfast_pybind is a python binding generation library for the analytic kinematics engine IKfast.
            Please visit here for more information: https://pypi.org/project/ikfast-pybind/

            ikfast:
            ikfast is a powerful inverse kinematics solver provided within
            Rosen Diankov’s OpenRAVE motion planning software.
            Diankov, R. (2010). Automated construction of robotic manipulation programs.

        Args:
            ee2base: transformation from end effector to base

        Returns:
            A list of possible IK solutions when the last joint value is set as free_joint_value

        """
        try:
            import ikfast_franka_panda as panda
        except ImportError:
            print("Please install ikfast_pybind before using this function")
            print("Install: pip3 install ikfast-pybind")
            raise ImportError

        link72ee = np.array([[0.7071068, -0.7071068, 0, 0], [0.7071068, 0.7071068, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        link7_pose = ee2base @ link72ee
        pos = link7_pose[:3, 3]
        rot = link7_pose[:3, :3]
        return panda.get_ik(pos, rot, [0.785])

    ####################################################################################################################
    # ============================== You will need to implement all the functions below ================================
    ####################################################################################################################
    @staticmethod
    def pose2mat(pose: sapien.Pose) -> np.ndarray:
        """You need to implement this function

        You will need to implement this function first before any other funcions.
        In this function, you need to convert a (position: pose.p, quaternion: pose.q) into a SE(3) matrix

        You can not directly use external library to transform quaternion into rotation matrix.
        Only numpy can be used here.
        Args:
            pose: sapien Pose object, where Pose.p and Pose.q are position and quaternion respectively

        Hint: the convention of quaternion

        Returns:
            (4, 4) transformation matrix represent the same pose

        """
        # QJ
        q_w = pose.q[0]
        q_xyz = np.expand_dims(pose.q[1:],axis=1)
        q_xyz_hat = SkewSymmetric(q_xyz)
        Eq = np.hstack((-q_xyz,q_w*np.eye(3)+q_xyz_hat))
        Gq = np.hstack((-q_xyz,q_w*np.eye(3)-q_xyz_hat))
        T = np.eye(4)
        T[:3,:3] = np.matmul(Eq,Gq.transpose())
        T[:3,3] = pose.p
        return T

    def get_current_marker_pose(self) -> np.ndarray:
        """You need to implement this function

        Using the visual information from an checkerboard to calculate the transformation.
        Some useful fact: checkerboard size: (3, 4), square size: 0.012

        hint: you can use standard library to hep you find the chessboard corner, e.g. OpenCV
        https://docs.opencv.org/4.3.0/dc/dbb/tutorial_py_calibration.html
        pip3 install opencv-python

        Returns:
            Transformation matrix from checkerboard frame to camera frame
        """
        self.scene.update_render()
        self.camera.take_picture()
        image = self.camera.get_color_rgba()[:, :, :3]

        pattern_size = (3, 4)
        square_size = 0.012

        # QJ
        # Convert the image to be 8-bit
        image = np.array(image*255, dtype=np.uint8)
        # Get gray-scale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Define the points in world frame (3D) with square_size
        objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        objp *= square_size
        # Detect corners in image coordiante (2D)
        ret, corners = cv2.findChessboardCorners(gray,pattern_size,flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        # Get known camera intrinsic parameters
        camera_matrix = self.camera.get_camera_matrix()[:3, :3]
        camera_distortion = np.zeros(4)
        # Solve a PnP problem with given intrinsic parameters
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, camera_matrix, camera_distortion)
        
        # optional visualization scripts
        '''
        def draw(img, corners, imgpts):
            corner = tuple(corners[0].ravel())
            img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
            img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
            return img
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        axis *= square_size
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, camera_distortion)
        img = draw(image,corners,imgpts)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        
        # Define 4x4 SE(3) matrix
        # Get the rotation (rvecs) and the translation (tvecs)
        T = np.eye(4)
        T[:3,:3],_ = cv2.Rodrigues(rvecs)
        T[:3,3] = np.squeeze(tvecs)
        return T

    def capture_calibration_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """You need to implement this function

        Capture transformation matrices that can be used for perform hand eye calibration.
        You can decide how to choose the pose and how many set of data to be captured by modifying qpos_list

        Returns:
            Transformation matrices used for calibration. The length of both list will be equal.
        """
        qpos_list = [list(self.robot.get_qpos()),
                     [0.1, 0.1, 0.1, -1.5, 0, 1.5, 0.7, 0.4, 0.4],
                     [2.6, -1.4, 0.3, -0.5, -2.9, 1.3, 0.4, 0.4, 0.4],
                     [2.6, -1.5, -0.2, 0, -1.6, 1.2, 0.8, 0.4, 0.4],
                     [0.1, 0, 0.2, -1.4, 0, 1.3, 0.6, 0.4, 0.4],
                     [0, 0.2, -0.1, -1.5, -0.3, 1.4, 0.9, 0.4, 0.4],
                     [0.1, 0.3, 0, -1.4, -0.2, 1.7, 0.85, 0.4, 0.4]]
        
        # QJ
        marker2cam_poses = []
        ee2base_poses = []
        for qpos in qpos_list[0:7]:
            self.robot.set_qpos(qpos)
            self.step()
            T_marker2cam = self.get_current_marker_pose()
            T_ee2base = self.get_current_ee_pose()
            marker2cam_poses.append(T_marker2cam)
            ee2base_poses.append(T_ee2base)
        return marker2cam_poses, ee2base_poses

    @staticmethod
    def compute_cam2base(marker2cam_poses: List[np.ndarray], ee2base_poses: List[np.ndarray]) -> np.ndarray:
        """You need to implement this function

        Main calculation function for hand eye calibration.
        The input of this function is exactly the output of capture_calibration_data
        Implement an algorithm to solve the AX=XB equation, e.g. Tsai or any other methods

        The reference below list a traditional algorithm to solve this problem, you can also use any other method.

        References:
            Tsai, R. Y., & Lenz, R. K. (1989).
            A new technique for fully autonomous and efficient 3 D robotics hand/eye calibration.
            IEEE Transactions on robotics and automation, 5(3), 345-358.

        Args:
            marker2cam_poses: a list of SE(3) matrices represent poses of marker to camera
            ee2base_poses: a list of SE(3) matrices represent poses of robot hand to robot base (static frame)

        Returns:
            Transformation matrix from camera to base

        """
        # QJ
        # Paper ref: http://kmlee.gatech.edu/me6406/handeye.pdf
        # Code ref: https://github.com/opencv/opencv/blob/master/modules/calib3d/src/calibration_handeye.cpp#L268
        R_base2ee = []
        p_base2ee = []
        T_base2ee = []
        R_marker2cam = []
        p_marker2cam = []
        T_marker2cam = []
        for idx in range(len(marker2cam_poses)):
            base2ee = SE3inverse(ee2base_poses[idx])
            marker2cam = marker2cam_poses[idx]
            R_base2ee.append(base2ee[:3,:3])
            p_base2ee.append(base2ee[:3,3])
            T_base2ee.append(Rp2T(R_base2ee[-1],p_base2ee[-1]))
            R_marker2cam.append(marker2cam[:3,:3])
            p_marker2cam.append(marker2cam[:3,3])
            T_marker2cam.append(Rp2T(R_marker2cam[-1],p_marker2cam[-1]))
        n_pairs = len(R_base2ee)
        A1,B1,A2,B2 = [],[],[],[] 
        # R_cg Step 1 eq (12)
        for i in range(n_pairs):
            for j in range(i,n_pairs):
                # Build P_gij eq (6)
                H_gij = np.matmul(SE3inverse(T_base2ee[j]),T_base2ee[i])
                R_gij = H_gij[:3,:3]
                P_gij = rot2quatMinimal(R_gij)
                # Build P_cij eq (7)
                H_cij = np.matmul(T_marker2cam[j],SE3inverse(T_marker2cam[i]))
                R_cij = H_cij[:3,:3]
                P_cij = rot2quatMinimal(R_cij)
                # A1: left side. B1: right side
                A1.append(SkewSymmetric(P_gij+P_cij))
                B1.append(np.expand_dims(P_cij-P_gij,axis=1))
        A1 = np.vstack(A1)
        B1 = np.vstack(B1)
        # pseudo-inverse solution
        #P_cg_prime = np.linalg.inv((A1.transpose() @ A1)) @ A1.transpose() @ B1
        # numpy least-squares
        P_cg_prime = np.linalg.lstsq(A1,B1,rcond=None)[0]
        # R_cg Step 3 eq (14)
        P_cg = 2*P_cg_prime/np.sqrt(1+P_cg_prime.transpose()@P_cg_prime)
        R_cg = quatMinimal2rot(P_cg)
        # T_cg eq (15)
        for i in range(n_pairs):
            for j in range(i,n_pairs):
                H_gij = np.matmul(SE3inverse(T_base2ee[j]),T_base2ee[i])
                R_gij = H_gij[:3,:3]
                T_gij = np.expand_dims(H_gij[:3,3],axis=1)
                H_cij = np.matmul(T_marker2cam[j],SE3inverse(T_marker2cam[i]))
                T_cij = np.expand_dims(H_cij[:3,3],axis=1)
                # A2: left side. B2: right side
                A2.append(R_gij-np.eye(3))
                B2.append(R_cg@T_cij-T_gij)
        A2 = np.vstack(A2)
        B2 = np.vstack(B2)
        # numpy least-squares
        T_cg = np.linalg.lstsq(A2,B2,rcond=None)[0]
        # Build SE(3) matrix
        T = np.eye(4)
        T[:3,:3] = R_cg
        T[:3,3] = np.squeeze(T_cg)
        
        # OpenCV implementation
        '''
        R_cam2base,t_cam2base =	cv2.calibrateHandEye(R_base2ee,p_base2ee,R_marker2cam,p_marker2cam,method=cv2.CALIB_HAND_EYE_TSAI)
        T = np.eye(4)
        T[:3,:3] = R_cam2base
        T[:3,3] = np.squeeze(t_cam2base)
        '''

        return T

    @staticmethod
    def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
        """You need to implement this function

        A distance function in SE(3) space
        Args:
            pose1: transformation matrix
            pose2: transformation matrix

        Returns:
            Distance scalar

        """
        # QJ
        T_diff = np.matmul(np.linalg.inv(pose1),pose2)
        R_diff = T_diff[:3,:3]
        p_diff = T_diff[:3,3]
        # edge cases: R_diff approx identity
        if np.abs(np.trace(R_diff)-3) < 1e-1:
            theta = np.zeros(3)
            rho = p_diff
        else:
            theta = SO32rotvec(R_diff)
            rho = np.matmul(J_L_inv(theta),p_diff)
        dist = np.linalg.norm(np.hstack((theta,rho)))
        return dist 

    def compute_grasp_qpos(self, cam2base: np.ndarray, seg_id: int) -> Sequence[float]:
        """You need to implement this function

        Compute the joint values for robot in order to grasp the object with given segmentation id.
        The feasible grasp pose can be extracted from the point cloud, you need to figure out a way (maybe hard-code)
        You can use the provided IK function here but you can not use the ground-truth cam2base transformation

        Note that the end-effector is defined as robot hand, not the finger tip.
        Also the input of the compute_ik function is the pose of hand.
        Thus you may want to estimate the transformation from finger tip to hand (e.g. trail and error)
        in order to make problem easier.


        Args:
            seg_id: segmentation id of the object
            cam2base: transformation from camera to base, calculated from hand-eye calibration

        Returns:
            Joint position of the robot, can be list, one-dimensional numpy array, tuple
            Note that you need to consider the joint position of gripper when return the qpos.
            e.g. len(qpos) == robot.dof == 9

        hint: the inverse kinematics function will calculate the joint angle of robot arm in order to achieve the
            desired SE(3) pose of end-effector(robot hand). Thus it only consider the first 7 dof of the robot. For the
            joint angle of last 2 dof, which are two fingers, will not be calculated by IK. For this panda robot, if the
            gripper is closed, the joint angles (qpos) of the last 2 dof are both 0. Similarly, if joint value is
            0.04, it means that the gripper is fully open. You can play with the GUI and see what will happen if you
            change the value of any joint.


        """
        # QJ
        self.robot.set_qpos([-0.144,0.263,0.176,-2.641,0.048,2.774,0.624,0.04,0.04])
        self.step()
        #print(SE3inverse(cam2base)@self.get_current_ee_pose())

        # Get object point cloud
        point_cloud = self.get_object_point_cloud(seg_id)
        # Simply get the mean of the point cloud
        obj_p = np.mean(point_cloud,axis=1)
        obj2cam = np.eye(4)
        obj2cam[:3,3] = obj_p
        # manually tested with different R and p offsets
        #obj2cam[:3,:3] = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        #obj2cam[:3,3] -= np.array([0,0,0.095])
        obj2cam[:3,:3] = np.array([[ 0.13959368, -0.98842393,  0.05943362],
                                [ 0.59127277,  0.13135001,  0.79570365],
                                [-0.79429889, -0.07593369,  0.6027636]])
        obj2cam[:3,3] -= np.array([0,0.08,0.05])
        gripper_pose = cam2base@obj2cam
        qpos = self.compute_ik(gripper_pose)[0]
        qpos = qpos + [0.04, 0.04]
        return qpos