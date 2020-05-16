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
from simple_pid import PID

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
    #theta_test = SkewSymmetricinv(norm/(2*np.sin(norm))*(rot-rot.transpose()))
    return theta

def rotvec2SO3(theta):
    """
    The exponential map for rotation vector
    Args:
        theta: 3d rotation vector
    Returns:
        R: 3x3 matrix in SO(3)
    """
    theta_norm = np.linalg.norm(theta)
    R = np.eye(3) + np.sin(theta_norm)/theta_norm*SkewSymmetric(theta) + (1-np.cos(theta_norm))/(theta_norm**2)*SkewSymmetric(theta)@SkewSymmetric(theta)
    return R      

def G_inv(theta):
    """
    The inverse matrix used to solve translation part of a twist from SE(3)
    Args:
        theta: 3d vector
    Returns:
        R: 3x3 matrix
    """
    # From MLS book page 43-44
    theta_norm = np.linalg.norm(theta)
    unit_theta = theta / theta_norm
    R = (np.eye(3)-rotvec2SO3(theta))@SkewSymmetric(unit_theta) + theta_norm*np.reshape(unit_theta,(3,1))@np.reshape(unit_theta,(1,3))
    R = np.linalg.inv(R)
    return R

def dist_SE3(pose1,pose2): 
    """
    The distance between two SE(3) matrices
    Args:
        rot1,rot2: 4x4 matrix in SE(3)
    Returns:
        dist: non-negative scalar
    """ 
    T_diff = np.matmul(np.linalg.inv(pose1),pose2)
    R = T_diff[:3,:3]
    p = T_diff[:3,3]
    if np.abs(np.trace(R)-3) < 1e-1:
        theta = np.zeros(3)
        rho = p
        Theta = np.linalg.norm(rho)
        Unit_twist = np.hstack((theta,rho/Theta))
    else:
        theta = SO32rotvec(R)
        rho = G_inv(theta)@p
        Theta = np.linalg.norm(theta)
        Unit_twist = np.hstack((theta/Theta,rho))
    dist = np.linalg.norm(Unit_twist)*Theta
    return dist

def exp_twist(twist,theta): 
    """
    Exponential map from twist to SE(3)
    Args:
        twist: 6D vector
        theta: scalar
    Returns:
        T: 4x4 matrix in SE(3)
    """ 
    xi_skew = np.zeros((4,4))
    twist = twist*theta
    xi_skew[:3,:3] = SkewSymmetric(twist[:3])
    xi_skew[:3,3] = twist[3:]
    T = np.eye(4) + xi_skew + (1-np.cos(theta))/(theta**2)*xi_skew@xi_skew + (theta-np.sin(theta))/(theta**3)*xi_skew@xi_skew@xi_skew
    return T
#######################################
# === Some useful functions above === #
#######################################

class HW2Env(StackingEnv):
    def __init__(self, timestep: float):
        """Class for homework2

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

        self.arm_joints = [joint for joint in self.robot.get_joints() if
                           joint.get_dof() > 0 and not joint.get_name().startswith("panda_finger")]
        self.set_joint_group_property(self.arm_joints, 1000, 400)
        assert len(self.arm_joints) == self.robot.dof - 2
        self.set_joint_group_property(self.gripper_joints, 200, 60)

        self.step()
        self.robot.set_drive_target(self.robot.get_qpos())

    def cam2base_gt(self) -> np.ndarray:
        """Get ground truth transformation of camera to base transformation

        Returns:
            Ground truth transformation from camera to robot base

        """
        camera_pose = self.camera_link.get_pose() * self.gl_pose
        base_pose = self.robot.get_root_pose()
        return self.pose2mat(base_pose.inv() * camera_pose)

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

    def close_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.001)

    def open_gripper(self):
        for joint in self.gripper_joints:
            joint.set_drive_target(0.04)

    def clear_velocity_command(self):
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(0)

    def wait_n_steps(self, n: int):
        self.clear_velocity_command()
        for i in range(n):
            passive_force = self.robot.compute_passive_force()
            self.robot.set_qf(passive_force)
            self.step()
            self.render()
        self.robot.set_qf([0] * self.robot.dof)

    def internal_controller(self, qvel: np.ndarray) -> None:
        """Control the robot dynamically to execute the given twist for one time step

        This method will try to execute the joint velocity using the internal dynamics function in SAPIEN.

        Note that this function is only used for one time step, so you may need to call it multiple times in your code
        Also this controller is not perfect, it will still have some small movement even after you have finishing using
        it. Thus try to wait for some steps using self.wait_n_steps(n) like in the hw2.py after you call it multiple
        time to allow it to reach the target position

        Args:
            qvel: (7,) vector to represent the joint velocity

        """
        assert qvel.size == len(self.arm_joints)
        target_qpos = qvel * self.scene.get_timestep() + self.robot.get_drive_target()[:-2]
        for i, joint in enumerate(self.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = self.robot.compute_passive_force()
        self.robot.set_qf(passive_force)

    def evaluate_first_two_box(self) -> bool:
        """Evaluate whether you stack the first two boxes successfully"""
        position, size = self.target
        rbox, gbox, _ = self.boxes
        contacts = self.scene.get_contacts()

        red_target_position = np.array([position[0], position[1], size])
        green_target_position = np.array([position[0], position[1], 3 * size])
        print("red target",red_target_position,"red reality",rbox.get_pose().p)
        print("green target",green_target_position,"green reality",gbox.get_pose().p)
        red_in_place = np.linalg.norm(rbox.get_pose().p - red_target_position) < 0.01
        green_in_place = np.linalg.norm(gbox.get_pose().p - green_target_position) < 0.01
        return green_in_place and red_in_place

    def evaluate_final_result(self) -> bool:
        """Evaluate whether you stack the all three boxes successfully"""
        position, size = self.target
        rbox, gbox, bbox = self.boxes
        static = (rbox.velocity @ rbox.velocity) < 1e-5 and \
                 (gbox.velocity @ gbox.velocity) < 1e-5 and \
                 (bbox.velocity @ bbox.velocity) < 1e-5
        if not static:
            return False

        first_two_box_success = self.evaluate_first_two_box()
        blue_target_position = np.array([position[0], position[1], 5 * size])
        blue_in_place = np.linalg.norm(bbox.get_pose().p - blue_target_position) < 0.01
        return blue_in_place and first_two_box_success

    ####################################################################################################################
    # ============================== You will need to implement all the functions below ================================
    ####################################################################################################################
    @staticmethod
    def pose2mat(pose: sapien.Pose) -> np.ndarray:
        """You need to implement this function

        You will need to implement this function first before any other functions.
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
        T[:3,:3] = Eq@(Gq.transpose())
        T[:3,3] = pose.p
        return T

    def pose2exp_coordinate(self, pose: np.ndarray) -> Tuple[np.ndarray, float]:
        """You may need to implement this function

        Compute the exponential coordinate corresponding to the given SE(3) matrix
        Note: unit twist is not a unit vector

        Args:
            pose: (4, 4) transformation matrix

        Returns:
            Unit twist: (6, ) vector represent the unit twist
            Theta: scalar represent the quantity of exponential coordinate
        """
        # QJ
        # Decouple to get rotation and translation part
        R = pose[:3,:3]
        p = pose[:3,3]
        # Special case when R == I
        if np.abs(np.trace(R)-3) < 1e-1:
            theta = np.zeros(3)
            rho = p
            Theta = np.linalg.norm(rho)
            Unit_twist = np.hstack((theta,rho/Theta))
        else:
            theta = SO32rotvec(R)
            rho = G_inv(theta)@p
            Theta = np.linalg.norm(theta)
            Unit_twist = np.hstack((theta/Theta,rho))
        return Unit_twist, Theta


    def compute_joint_velocity_from_twist(self, twist: np.ndarray) -> np.ndarray:
        """You need to implement this function

        This function is a kinematic-level calculation which do not consider dynamics.
        Pay attention to the frame of twist, is it spatial twist or body twist

        Jacobian is provided for your, so no need to compute the velocity kinematics
        ee_jacobian is the geometric Jacobian on account of only the joint of robot arm, not gripper
        Jacobian in SAPIEN is defined as the derivative of spatial twist with respect to joint velocity

        Args:
            twist: (6,) vector to represent the twist

        Returns:
            (7, ) vector for the velocity of arm joints (not include gripper)

        """
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        dense_jacobian = self.robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        ee_jacobian = np.zeros([6, self.robot.dof - 2])  # (6, 7)
        ee_jacobian[:3, :] = dense_jacobian[self.end_effector_index * 6 - 3:self.end_effector_index * 6, :7]
        ee_jacobian[3:6, :] = dense_jacobian[(self.end_effector_index - 1) * 6:self.end_effector_index * 6 - 3, :7]
        # QJ 
        # Derive the pseudo-inverse of Jacobian
        ee_jacobian_inv = np.linalg.pinv(ee_jacobian)
        # Convert the twist to the joint velocity
        joint_velocity = ee_jacobian_inv@twist
        return joint_velocity
        

    def move_to_target_pose_with_internal_controller(self, target_ee_pose: np.ndarray, num_steps: int) -> None:
        """You need to implement this function

        Move the robot hand dynamically to a given target pose
        You may need to call self.internal_controller and your self.compute_joint_velocity_from_twist in this function

        To make command (e.g. internal controller) take effect and simulate all the physical effects, you need to step
        the simulation world for one step and render the new scene for visualization by something like:
            for i in range(num_step):
                # Do something
                self.internal_controller(target_joint_velocity)
                self.step()
                self.render()

        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps: how much steps to reach to target pose, each step correspond to self.scene.get_timestep() seconds
                in physical simulation

        """
        # QJ
        for i in range(num_steps):
            # Get current ee pose
            current_ee_pose = self.get_current_ee_pose()
            current_ee_R = current_ee_pose[:3,:3]
            current_ee_p = current_ee_pose[:3,3]
            # Compare with the target ee pose and get the relative transformation
            delta_ee_pose = np.linalg.inv(current_ee_pose)@target_ee_pose
            # Compute the exponential coordinate of the relative transformation
            delta_ee_twist,delta_ee_theta = self.pose2exp_coordinate(delta_ee_pose)
            # Given the time left to approach the target, compute the average twist
            time_to_target = (num_steps - i)*self.scene.get_timestep()            
            delta_ee_twist_body = delta_ee_twist * (delta_ee_theta / time_to_target)
            # Convert the body twist to the spatial twist
            # by timing the adjoint matrix
            adj = np.zeros((6,6))
            adj[0:3,0:3] = current_ee_R
            adj[3:6,3:6] = current_ee_R
            adj[3:6,0:3] = SkewSymmetric(current_ee_p)@current_ee_R
            delta_ee_twist_spatial = adj@delta_ee_twist_body
            # Compute the joint velocities qvel from the spatial twist
            target_joint_velocity = self.compute_joint_velocity_from_twist(delta_ee_twist_spatial)
            self.internal_controller(target_joint_velocity)
            self.step()
            self.render()
        executed_time = num_steps * self.scene.get_timestep()
        return executed_time

    def pick_up_object_with_internal_controller(self, seg_id: int, height: float) -> None:
        """You need to implement this function

        Pick up a specific box to a target height using the given internal controller


        You can use the following function to get the segmented point cloud:
            point_cloud = self.get_object_point_cloud(seg_id)

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            height: target height of the box

        """
        # QJ
        # Get object point cloud
        point_cloud = self.get_object_point_cloud(seg_id)
        # Simply get the mean of the point cloud
        obj_p = np.mean(point_cloud,axis=1)
        obj2cam = np.eye(4)
        obj2cam[:3,3] = obj_p
        # (?) trick to refine the pose approaching the object
        # manually tested with different R and p offsets
        obj2cam[:3,:3] = np.array([[ 0.13959368, -0.98842393,  0.05943362],
                                [ 0.59127277,  0.13135001,  0.79570365],
                                [-0.79429889, -0.07593369,  0.6027636]])
        obj2cam[:3,3] -= np.array([0,0.08,0.05])
        # convert to the pose in spatial coordinate
        target_pose = self.cam2base_gt()@obj2cam
        current_pose = self.get_current_ee_pose()
        keep_height = current_pose[2,3] - target_pose[2,3]
        target_pose[2,3] += keep_height
        # move to target pose with internal controller
        # with two steps
        self.move_to_target_pose_with_internal_controller(target_pose, 100)
        self.wait_n_steps(100)
        target_pose[2,3] -= keep_height-0.01
        self.move_to_target_pose_with_internal_controller(target_pose, 100)
        self.wait_n_steps(50)
        # close the gripper
        self.close_gripper()
        self.wait_n_steps(200)
        # lift the object for certain height
        target_lift_pose = target_pose
        target_lift_pose[2,3] += height
        self.move_to_target_pose_with_internal_controller(target_lift_pose, 100)
        self.wait_n_steps(50)
        
        return 0

    def place_object_with_internal_controller(self, seg_id: int, target_object_position: np.ndarray) -> None:
        """You need to implement this function

        Place a specific box to a target position
        This function do not consider rotation, so you can just assume a fixed rotation or any orientation you want

        After place the box, you also need to move the gripper to some safe place. Thus when you pick up next box, it
        will not jeopardise your previous result.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            target_object_position: target position of the box

        """
        # QJ
        # move to target pose with internal controller
        current_object_pose = self.get_current_ee_pose()
        target_object_pose = np.copy(current_object_pose)
        # slightly tuning the target_object_position     
        target_object_position += np.array([0.01,0.00,0.11])
        target_object_pose[:3,3] = target_object_position 
        # move to target pose with internal controller
        # with two steps
        target_object_pose[2,3] += 0.03
        self.move_to_target_pose_with_internal_controller(target_object_pose, 100)
        self.wait_n_steps(50)
        target_object_pose[2,3] -= 0.02
        self.move_to_target_pose_with_internal_controller(target_object_pose, 100)
        self.wait_n_steps(50)
        # open the gripper to place the object
        self.open_gripper()
        self.wait_n_steps(200)
        # move the gripper to some safe place
        save_distance_yz = 0.1
        target_save_pose = self.get_current_ee_pose()
        #target_save_pose[2,2] += save_distance_yz
        target_save_pose[2,3] += save_distance_yz
        self.move_to_target_pose_with_internal_controller(target_save_pose, 100)
        self.wait_n_steps(50)
        
        return 0



    def move_to_target_pose_with_user_controller(self, target_ee_pose: np.ndarray, num_steps: int) -> None:
        """You need to implement this function

        Similar to self.move_to_target_pose_with_internal_controller. However, this time you need to implement your own
        controller instead of the SAPIEN internal controller.

        You can use anything you want to perform dynamically execution of the robot, e.g. PID, compute torque control
        You can write additional class or function to help implement this function.
        You can also use the inverse kinematics to calculate target joint position.
        You do not need to follow the given timestep exactly if you do not know how to do that.

        However, you are not allow to magically set robot's joint position and velocity using set_qpos() and set_qvel()
        in this function. You need to control the robot by applying appropriate force on the robot like real-world.

        There are two function you may need to use (optional):
            gravity_compensation = self.robot.compute_passive_force(gravity=False, coriolis_and_centrifugal=True,
                                                                    external=False)
            coriolis_and_centrifugal_compensation = self.robot.compute_passive_force(gravity=False,
                                                                                    coriolis_and_centrifugal=True,
                                                                                    external=False)

        The first function calculate how much torque each joint need to apply in order to balance the gravity
        Similarly, the second function calculate how much torque to balance the coriolis and centrifugal force

        To controller your robot actuator dynamically (actuator is mounted on each joint), you can use
        self.robot.set_qf(joint_torque)
        Note that joint_torque is a (9, ) vector which also includes the joint torque of two gripper

        Args:
            target_ee_pose: (4, 4) transformation of robot hand in robot base frame (ee2base)
            num_steps: how much steps to reach to target pose, each step correspond to self.scene.get_timestep() seconds

        """
        timestep = self.scene.get_timestep()
        executed_time = num_steps * timestep
        # QJ
        # (?) TODO
        # Try a PID controller
        # ref: https://sapien.ucsd.edu/docs/tutorial/robotics/pid.html
        
        # Define a PID class
        class SimplePID:
            def __init__(self, kp=0.0, ki=0.0, kd=0.0):
                self.p = kp
                self.i = ki
                self.d = kd
                self._cp = 0
                self._ci = 0
                self._cd = 0
                self._last_error = 0
            # compute the control signal based on error
            def compute(self, current_error, dt):
                d_error = current_error - self._last_error
                self._cp = current_error
                self._ci += current_error * dt
                if abs(self._last_error) > 0.01:
                    self._cd = d_error / dt
                self._last_error = current_error
                signal = (self.p * self._cp) + (self.i * self._ci) + (self.d * self._cd)
                return signal

        # compute control signal for a list of joints
        def pid_forward(pids: list, target_pos: np.ndarray, current_pos: np.ndarray, dt: float) -> np.ndarray:
            qf = np.zeros(len(pids))
            errors = target_pos - current_pos
            #print(errors)
            for i in range(len(pids)):
                qf[i] = pids[i].compute(errors[i], dt)
            return qf, errors

        # PID parameters
        pids = []
        #pid_parameters = [(1000, 0, 0), (100,3,10), (100,0,5), (100,3,3), 
        #                (100,0,0), (100,3,0), (100,0,0), 
        #                (0,0,0),(0,0,0)]
        pid_parameters = [(500,0,3),(100,30,1),(100,0,3),(100,0,3), 
                        (10,10,1),(50,0,1),(50,0,1), 
                        (0,0,0),(0,0,0)]
        for i in range(9):
            pids.append(SimplePID(pid_parameters[i][0], pid_parameters[i][1], pid_parameters[i][2]))

        for i in range(num_steps):
            current_ee_pose = self.get_current_ee_pose()
            current_ee_R = current_ee_pose[:3,:3]
            current_ee_p = current_ee_pose[:3,3]
            # Compare with the target ee pose and get the relative transformation
            delta_ee_pose = np.linalg.inv(current_ee_pose)@target_ee_pose
            # Compute the exponential coordinate of the relative transformation
            delta_ee_twist,delta_ee_theta = self.pose2exp_coordinate(delta_ee_pose)
            # Given the time left to approach the target, compute the average twist
            time_to_target = (num_steps-i)*self.scene.get_timestep()        
            delta_ee_twist_body = delta_ee_twist * (delta_ee_theta / time_to_target)
            # Convert the body twist to the spatial twist
            # by timing the adjoint matrix
            adj = np.zeros((6,6))
            adj[0:3,0:3] = current_ee_R
            adj[3:6,3:6] = current_ee_R
            adj[3:6,0:3] = SkewSymmetric(current_ee_p)@current_ee_R
            delta_ee_twist_spatial = adj@delta_ee_twist_body
            # Compute the joint velocities qvel from the spatial twist
            target_joint_velocity = self.compute_joint_velocity_from_twist(delta_ee_twist_spatial)
            # Compute target joint pose
            target_qpos = target_joint_velocity*self.scene.get_timestep() + self.robot.get_drive_target()[:-2]
            # keep the location of the gripper
            target_qpos = np.hstack((target_qpos,self.robot.get_qpos()[-2:]))
            
            
            ### visualization ###
            import matplotlib.pyplot as plt
            x = np.linspace(0, 100, 100)
            y = np.zeros((7,x.shape[0]))
            # You probably won't need this if you're embedding things in a tkinter plot...
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line0, = ax.plot(x,y[0,:],'b--',label='zero')
            line1, = ax.plot(x,y[0,:],'-',label='j1')
            line2, = ax.plot(x,y[0,:],'-',label='j2')
            line3, = ax.plot(x,y[0,:],'-',label='j3')
            line4, = ax.plot(x,y[0,:],'-',label='j4')
            line5, = ax.plot(x,y[0,:],'-',label='j5')
            line6, = ax.plot(x,y[0,:],'-',label='j6')
            line7, = ax.plot(x,y[0,:],'-',label='j7')
            ax.legend()
            ### visualization ###

            pid_list = []
            pid_list.append(PID(1000, 0, 100, setpoint=target_qpos[0]))
            pid_list.append(PID(50, 0, 10, setpoint=target_qpos[1]))
            pid_list.append(PID(100, 0, 10, setpoint=target_qpos[2]))
            pid_list.append(PID(50, 0, 10, setpoint=target_qpos[3]))
            pid_list.append(PID(100, 0, 10, setpoint=target_qpos[4]))
            pid_list.append(PID(100, 0, 10, setpoint=target_qpos[5]))
            pid_list.append(PID(100, 0, 10, setpoint=target_qpos[6]))
            

            while 1:
                gravity_compensation = self.robot.compute_passive_force(gravity=True, 
                                                                                    coriolis_and_centrifugal=False,
                                                                                    external=False)
                coriolis_and_centrifugal_compensation = self.robot.compute_passive_force(gravity=False,
                                                                                                    coriolis_and_centrifugal=True,
                                                                                                    external=False)
                # control signal from PID
                #pid_qf,q_error = pid_forward(pids,target_qpos,self.robot.get_qpos(),timestep) 
                pid_qf = []
                current_qpos = self.robot.get_qpos()
                for i in range(7):
                    pid_qf.append(pid_list[i](current_qpos[i]))
                pid_qf.append(0)
                pid_qf.append(0)
                pid_qf = np.array(pid_qf)
                q_error = target_qpos-current_qpos
                joint_torque = pid_qf+gravity_compensation+coriolis_and_centrifugal_compensation
                print(q_error,np.linalg.norm(q_error))
                if np.linalg.norm(q_error)<1e-3:
                    break
                self.robot.set_qf(joint_torque)
                self.step()
                self.render()         
                break

                ### visualization ###
                for idx in range(7):
                    y[idx,0:-1] = y[idx,1:]
                    y[idx,-1] = q_error[idx]
                line1.set_ydata(y[0,:])
                line2.set_ydata(y[1,:])
                line3.set_ydata(y[2,:])
                line4.set_ydata(y[3,:])
                line5.set_ydata(y[4,:])
                line6.set_ydata(y[5,:])
                line7.set_ydata(y[6,:])
                fig.canvas.draw()
                fig.canvas.flush_events()
                ### visualization ###            

            ### visualization ###
            plt.close('all')
            ### visualization ### 
            
            for j, joint in enumerate(self.arm_joints):
                joint.set_drive_velocity_target(target_joint_velocity[j])
                joint.set_drive_target(target_qpos[j])
            
        print("target_ee_pose",target_ee_pose)
        print("real_ee_pose",self.get_current_ee_pose())
        print("error",dist_SE3(target_ee_pose,self.get_current_ee_pose())) 

        return 0

    def pick_up_object_with_user_controller(self, seg_id: int, height: float) -> None:
        """You need to implement this function

        Pick up a specific box to a target height using your own controller

        To achieve a pick up action, you can either call your self.move_to_target_pose_with_user_controller function to
        move a sequence of pose designed manually, or using a motion planning algorithm like RRT to get a trajectory and
        execute the trajectory.

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            height: target height of the box

        """
        # QJ
        # TODO: replace internal_controller with user_controller
        def user_wait_n_steps(n):
            for i in range(n):
                passive_force = self.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
                self.robot.set_qf(passive_force)
                self.step()
                self.render()

        def user_close_gripper(n):
            for i in range(n):
                passive_force = self.robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True)
                gripper_force = passive_force
                gripper_force[-2:] = -10
                print("gripper_force",gripper_force)
                self.robot.set_qf(gripper_force)
                self.step()
                self.render()  

        _, _, bbox = self.boxes
        # Get object point cloud
        point_cloud = self.get_object_point_cloud(seg_id)
        # Simply get the mean of the point cloud
        obj_p = np.mean(point_cloud,axis=1)
        obj2cam = np.eye(4)
        obj2cam[:3,3] = obj_p
        # manually tested with different R and p offsets
        # Even the blue box need slight change on these
        obj2cam[:3,:3] = np.array([[ 0.13959368, -0.98842393,  0.05943362],
                                [ 0.59127277,  0.13135001,  0.79570365],
                                [-0.79429889, -0.07593369,  0.6027636]])
        obj2cam[:3,3] -= np.array([0,0.10,0.05])
        # convert to the pose in spatial coordinate
        target_pose = self.cam2base_gt()@obj2cam
        # manually refine the contact point
        target_pose[:3,:3] = -np.eye(3)
        target_pose[0,0] = 1
        current_pose = self.get_current_ee_pose()
        keep_height = current_pose[2,3] - target_pose[2,3]
        target_pose[2,3] += keep_height
        # move to target pose with internal controller
        # with two steps
        print("move to the top")
        self.move_to_target_pose_with_internal_controller(target_pose, 1000)
        target_pose[2,3] -= keep_height+0.02
        print("move to the box")
        self.move_to_target_pose_with_internal_controller(target_pose, 2000)
        print("keep move to the box")
        self.move_to_target_pose_with_internal_controller(target_pose, 500)
        # close the gripper
        print("pick the box")
        self.close_gripper()
        self.move_to_target_pose_with_internal_controller(target_pose, 500)
        # lift the object for certain height
        target_lift_pose = target_pose
        # trick
        target_lift_pose[2,3] += height
        print("lift the box")
        self.move_to_target_pose_with_internal_controller(target_lift_pose, 1000)
        
        return 0

    def place_object_with_user_controller(self, seg_id: int, target_object_position: np.ndarray) -> None:
        """You need to implement this function

        Similar to the last function, place the box to the given position with your own controller
        If you have already implemented the pick_up_object_with_user_controller, this function is not hard for you

        Args:
            seg_id: segmentation id, you can get it by e.g. box.get_id()
            target_object_position: target position of the box

        """
        # QJ
        # TODO: replace internal_controller with user_controller
        _, _, bbox = self.boxes
        current_object_pose = self.get_current_ee_pose()
        target_object_pose = np.copy(current_object_pose)
        target_object_pose[:3,:3] = -np.eye(3)
        target_object_pose[0,0] = 1
        # slightly tuning the target_object_position  
        target_object_position[0] += 0.0408   
        print(target_object_position)
        target_object_pose[:3,3] = target_object_position 
        # move to target pose with user controller
        # with multiple steps
        hold_height = current_object_pose[2,3]-target_object_position[2]+0.2
        target_object_pose[2,3] += hold_height
        print("horizontally move the box")
        self.move_to_target_pose_with_internal_controller(target_object_pose, 3000)
        target_object_pose[2,3] -= hold_height-0.1
        print("lower place the box")
        self.move_to_target_pose_with_internal_controller(target_object_pose, 3000)
        # open the gripper to place the object
        print("stable the box")
        self.move_to_target_pose_with_internal_controller(target_object_pose, 1000)
        print("release the box")
        self.open_gripper()
        self.move_to_target_pose_with_internal_controller(target_object_pose, 1000)
        print("blue box",bbox.get_pose().q,bbox.get_pose().p)
        print(self.get_current_ee_pose())
        # move the gripper to some safe place
        save_distance_yz = 0.1
        target_save_pose = self.get_current_ee_pose()
        target_save_pose[2,2] += save_distance_yz
        target_save_pose[2,3] += save_distance_yz
        self.move_to_target_pose_with_internal_controller(target_save_pose, 1000)
        
        return 0
