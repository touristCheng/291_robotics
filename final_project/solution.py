# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC San Diego.
# Created by Yuzhe Qin, Fanbo Xiang

from final_env import FinalEnv, SolutionBase
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2axangle, qmult, qinverse
import cv2
import matplotlib.pyplot as plt

def skew(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])


def adjoint_matrix(pose):
    adjoint = np.zeros([6, 6])
    adjoint[:3, :3] = pose[:3, :3]
    adjoint[3:6, 3:6] = pose[:3, :3]
    adjoint[3:6, 0:3] = skew(pose[:3, 3]) @ pose[:3, :3]
    return adjoint

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
    return direction, norm

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

def so32rot(rotation: np.ndarray):
    assert rotation.shape == (3, 3)
    if np.isclose(rotation.trace(), 3):
        return np.zeros(3), 1
    if np.isclose(rotation.trace(), -1):
        # omega, theta = mat2axangle(rotation)
        theta = np.arccos((np.max([-1+1e-7,rotation.trace()]) - 1) / 2)
        omega = 1 / 2 / np.sin(theta) * np.array(
                [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
        return omega, theta 
    theta = np.arccos((rotation.trace() - 1) / 2)
    omega = 1 / 2 / np.sin(theta) * np.array(
                [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]).T
    return omega, theta
'''
# use transforms3d
def so32rot(rotation):
    omega, theta = mat2axangle(rotation)
    return omega, theta 
'''

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

def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """You need to implement this function

    A distance function in SE(3) space
    Args:
        pose1: transformation matrix
        pose2: transformation matrix

    Returns:
        Distance scalar

    """
    relative_rotation = pose1[:3, :3].T @ pose2[:3, :3]
    rotation_term = np.arccos((np.trace(relative_rotation) - 1) / 2)
    translation_term = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
    print("Rotation term is: {}\nTranslation Term is: {}".format(rotation_term, translation_term))
    return rotation_term + translation_term

class Solution(SolutionBase):
    """
    This is a very bad baseline solution
    It operates in the following ways:
    1. roughly align the 2 spades
    2. move the spades towards the center
    3. lift 1 spade and move the other away
    4. somehow transport the lifted spade to the bin
    5. pour into the bin
    6. go back to 1
    """

    def init(self, env: FinalEnv):

        self.env = env
        self.phase = 0
        self.drive = 0
        meta = env.get_metadata()
        self.box_ids = meta['box_ids']
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        self.ps = [1000, 800, 600, 600, 200, 200, 100]
        #self.ps = [0, 0, 0, 0, 0, 0, 0]
        self.ds = [1000, 800, 600, 600, 200, 200, 100]
        r1.configure_controllers(self.ps, self.ds)
        r2.configure_controllers(self.ps, self.ds)

        # measure the bin
        self.bin_id = meta['bin_id']
        self.basic_info = {}
        self.locate_bin_bbox(c4)
        print(self.basic_info)

        self.measured = False
        self.marks = set()
        
        # get the box location from the overhead camera
        self.box_location_update_flag = False
        self.box_state = np.zeros(10)
        self.box_location = np.zeros((10,3))
        self.update_box_state()
        print(self.box_location)

    def act(self, env: FinalEnv, current_timestep: int):
        
        
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        pf_left = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = r2.get_compute_functions()['passive_force'](True, True, False)

        # goal: gather the box into the same location        

        # initialize to some place
        if self.phase == 0:

            # Directly set joint pose
            t1 = [2, 1, 0, -1.5, -1, 1, -2]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]

            r1.set_action(t1, [0] * 7, pf_left)
            r2.set_action(t2, [0] * 7, pf_right)

            if np.allclose(r1.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    r2.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 1
                self.counter = 0
                self.selected_x = None

        # pick some place to aim for boxes
        if self.phase == 1:

            self.counter += 1
            if (self.counter == 1):

                selected, done = self.pick_one(c4)
                if done:
                    return False
                else:
                    self.selected_x = selected

                # selected = self.pick_box(c4)
                # self.selected_x = selected[0]
                # self.selected_y = selected[1]
                # if self.selected_x is None:
                #     return False

            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 3, -np.pi / 2))

            self.diff_drive(r1, 9, target_pose_left)
            self.target_pose_left = target_pose_left

            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 3, np.pi / 2))

            self.diff_drive(r2, 9, target_pose_right)
            self.target_pose_right = target_pose_right

            # set target to close two spades
            if self.counter == 2000 / 5:
                self.phase = 2
                self.counter = 0

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = -0.07
                self.pose_left = Pose(p, q)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.07
                self.pose_right = Pose(p, q)


        # close two spades
        if self.phase == 2:
        
            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)

            # set target to lift one spade
            if self.counter == 2000 / 5:
                self.phase = 3

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.2
                q = euler2quat(np.pi, -np.pi / 0.75, quat2euler(q)[2])
                self.pose_right = Pose(p, q)

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.5
                q = euler2quat(np.pi, -np.pi / 2, -np.pi / 2)
                self.pose_left = Pose(p, q)
                self.pose_left = self.target_pose_left
                self.counter = 0



        # lift one spade and move the other away
        if self.phase == 3:

            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)
            if self.counter == 200 / 5:
                self.phase = 4
                self.counter = 0

                if not self.measured:
                    self.measure_spade(c4, r2)

        # move to the bin
        if self.phase == 4:
            
            '''
            # used to fix at specific pose
            self.ps = [0, 0, 0, 0, 0, 0, 0]
            self.ds = [1000, 800, 600, 600, 200, 200, 100]
            r1.configure_controllers(self.ps, self.ds)
            r2.configure_controllers(self.ps, self.ds)
            pf_left = r1.get_compute_functions()['passive_force'](True, True, False)
            pf_right = r2.get_compute_functions()['passive_force'](True, True, False)

            r1.set_action(r1.get_observation()[0], self.ps, pf_left)
            r2.set_action(r2.get_observation()[0], self.ps, pf_right)
            '''
            

            self.counter += 1
            # middle point 1
            if (self.counter < 3000 / 5):
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.5
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.jacobian_drive(env, r2, 9, Pose(p, q))
            # target bin location
            elif (self.counter < 9000 / 5):
                p = [-1, -0.15, 1.18] #a milestone to control the trajectory for avoiding collision
                q = euler2quat(0, -np.pi / 3, 0)
                self.jacobian_drive(env, r2, 9, Pose(p, q), speed=0.3)
            # target bin location

            elif (self.counter < 15000 / 5):
                p = [-1, -0.1, 1.2]

                cent = self.basic_info['bin_center']
                length = self.basic_info['spade_length']
                p = cent.copy()
                p[2] += 0.15
                p[0] += length * 2.

                q = euler2quat(0, -np.pi / 1.2, 0)
                self.jacobian_drive(env, r2, 9, Pose(p, q), speed=0.2)
            else:
                self.phase = 0
                # return False

            '''
            self.counter += 1
            # middle point 1
            if (self.counter < 3000 / 5):
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 0, -1, 0], [0, 1, 2, 3, 4])
            # middle point 2
            elif (self.counter < 6000 / 5):
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 1, -1, 0], [0, 1, 2, 3, 4])
            # target bin location
            elif (self.counter < 9000 / 5):
                p = [-1, 0, 1.2]
                q = euler2quat(0, -np.pi / 1.5, 0)
                self.diff_drive(r2, 9, Pose(p, q))
            else:
                self.phase = 0
                # return False
            '''

    def diff_drive(self, robot, index, target_pose):
        """
        this diff drive is very hacky
        it tries to transport the target pose to match an end pose
        by computing the pose difference between current pose and target pose
        then it estimates a cartesian velocity for the end effector to follow.
        It uses differential IK to compute the required joint velocity, and set
        the joint velocity as current step target velocity.
        This technique makes the trajectory very unstable but it still works some times.
        """
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        max_v = 0.1
        max_w = np.pi
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        delta_p = target_pose.p - current_pose.p
        delta_q = qmult(target_pose.q, qinverse(current_pose.q))

        axis, theta = quat2axangle(delta_q)
        if (theta > np.pi):
            theta -= np.pi * 2

        t1 = np.linalg.norm(delta_p) / max_v
        t2 = theta / max_w
        t = max(np.abs(t1), np.abs(t2), 0.001)
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        v = delta_p / t
        w = theta / t * axis
        target_qvel = robot.get_compute_functions()['cartesian_diff_ik'](np.concatenate((v, w)), 9)
        robot.set_action(qpos, target_qvel, pf)

    def pose2mat(self, pose):
        # ref solution

        mat44 = np.eye(4)
        mat44[:3, 3] = pose.p

        quat = np.array(pose.q).reshape([4, 1])
        if np.linalg.norm(quat) < np.finfo(np.float).eps:
            return mat44
        quat /= np.linalg.norm(quat, axis=0, keepdims=False)
        img = quat[1:, :]
        w = quat[0, 0]

        Eq = np.concatenate([-img, w * np.eye(3) + skew(img)], axis=1)  # (3, 4)
        Gq = np.concatenate([-img, w * np.eye(3) - skew(img)], axis=1)  # (3, 4)

        mat44[:3, :3] = np.dot(Eq, Gq.T)
        return mat44

    def measure_spade(self, c, r):
        '''
        use hough line transform to detect long spade side, and determine the size of spade

        :param c:
        :return:
        '''
        color, depth, segmentation = c.get_observation()

        r_meta = r.get_metadata()
        spade_id = r_meta['link_ids'][-1]

        h_s, w_s = np.where(segmentation == spade_id)
        points = np.stack([w_s, h_s], axis=1)
        rbox = cv2.minAreaRect(points)
        center, size, rot = rbox
        center = np.array(center, np.int)

        # corners = cv2.boxPoints(rbox)
        # plt.figure()
        # plt.imshow(color)
        # for p in corners:
        #     plt.plot(p[0], p[1], 'ro')
        # plt.plot(center[0], center[1], 'o', c=(0., 1., 0.))
        # plt.show()

        spade_cent_world = self.get_global_position_from_camera(c, depth, center[0], center[1])

        cur_pose = r.get_observation()[2][9] #get spade position
        spade_root_world = cur_pose.p

        spade_length = 2 * (np.sum((spade_cent_world[:3] - spade_root_world)**2)**0.5)

        self.basic_info['spade_length'] = spade_length
        self.measured = True

    def pose2exp_coordinate(self, pose: np.ndarray):
        # ref solution
        # omega, theta = SO32rotvec(pose[:3, :3])
        omega, theta = so32rot(pose[:3, :3])
        ss = skew(omega)
        inv_left_jacobian = np.eye(3, dtype=np.float) / theta - 0.5 * ss + (
                1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
        v = inv_left_jacobian @ pose[:3, 3]
        return np.concatenate([omega, v]), theta

    def internal_controller(self, env, robot, qvel: np.ndarray):
        target_qpos = qvel * env.get_metadata["timestep"] + robot.get_drive_target()[:-2]
        for i, joint in enumerate(robot.arm_joints):
            joint.set_drive_velocity_target(qvel[i])
            joint.set_drive_target(target_qpos[i])
        passive_force = robot.compute_passive_force(True, True, False)
        robot.set_qf(passive_force)

    def compute_joint_velocity_from_twist(self, robot, twist: np.ndarray) -> np.ndarray:
        assert twist.size == 6
        # Jacobian define in SAPIEN use twist (v, \omega) which is different from the definition in the slides
        # So we perform the matrix block operation below
        # dense_jacobian = robot.compute_spatial_twist_jacobian()  # (num_link * 6, dof())
        dense_jacobian = robot.get_compute_functions()['spatial_twist_jacobian']()
        ee_jacobian = np.zeros([6, robot.dof])  # (6, 7)
        end_effector_index = 9
        ee_jacobian[:3, :] = dense_jacobian[end_effector_index * 6 - 3:end_effector_index * 6, :7]
        ee_jacobian[3:6, :] = dense_jacobian[(end_effector_index - 1) * 6:end_effector_index * 6 - 3, :7]
        # QJ 
        # Derive the pseudo-inverse of Jacobian

        ee_jacobian_inv = np.linalg.pinv(ee_jacobian)
        # Convert the twist to the joint velocity
        joint_velocity = ee_jacobian_inv@twist
        return joint_velocity

    def jacobian_drive(self, env, robot, index, target_pose, speed=0.3):
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        current_pose = self.pose2mat(current_pose)
        current_R = current_pose[:3,:3]
        current_p = current_pose[:3,3]
        # Compare with the target ee pose and get the relative transformation
        target_pose = self.pose2mat(target_pose)
        delta_pose = np.linalg.inv(current_pose)@target_pose
        # Compute the exponential coordinate of the relative transformation
        delta_twist,delta_theta = self.pose2exp_coordinate(delta_pose)
        # Given the time left to approach the target, compute the average twist
        delta_twist_body = delta_twist * speed
        # Convert the body twist to the spatial twist
        # by timing the adjoint matrix
        adj = np.zeros((6,6))
        adj[0:3,0:3] = current_R
        adj[3:6,3:6] = current_R
        adj[3:6,0:3] = SkewSymmetric(current_p)@current_R
        delta_ee_twist_spatial = adj@delta_twist_body
        # Compute the joint velocities qvel from the spatial twist
        target_qvel = self.compute_joint_velocity_from_twist(robot, delta_ee_twist_spatial)
            
        robot.set_action(qpos, target_qvel, pf)


    def diff_drive2(self, robot, index, target_pose, js1, joint_target, js2):
        """
        This is a hackier version of the diff_drive
        It uses specified joints to achieve the target pose of the end effector
        while asking some other specified joints to match a global pose
        """
        pf = robot.get_compute_functions()['passive_force'](True, True, False)
        max_v = 0.1
        max_w = np.pi
        qpos, qvel, poses = robot.get_observation()
        current_pose: Pose = poses[index]
        delta_p = target_pose.p - current_pose.p
        delta_q = qmult(target_pose.q, qinverse(current_pose.q))

        axis, theta = quat2axangle(delta_q)
        if (theta > np.pi):
            theta -= np.pi * 2

        t1 = np.linalg.norm(delta_p) / max_v
        t2 = theta / max_w
        t = max(np.abs(t1), np.abs(t2), 0.001)
        thres = 0.1
        if t < thres:
            k = (np.exp(thres) - 1) / thres
            t = np.log(k * t + 1)
        v = delta_p / t
        w = theta / t * axis
        target_qvel = robot.get_compute_functions()['cartesian_diff_ik'](np.concatenate((v, w)), 9)
        for j, target in zip(js2, joint_target):
            qpos[j] = target
        robot.set_action(qpos, target_qvel, pf)

    def get_global_position_from_camera(self, camera, depth, x, y):
        """
        camera: an camera agent
        depth: the depth obsrevation
        x, y: the horizontal, vertical index for a pixel, you would access the images by image[y, x]
        """
        cm = camera.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        # get 0 to 1 coordinate for (x, y) coordinates
        xf, yf = (x + 0.5) / w, 1 - (y + 0.5) / h

        # get 0 to 1 depth value at (x,y)
        zf = depth[int(y), int(x)]

        # get the -1 to 1 (x,y,z) coordinates
        ndc = np.array([xf, yf, zf, 1]) * 2 - 1

        # transform from image space to view space
        v = np.linalg.inv(proj) @ ndc
        v /= v[3]

        # transform from view space to world space
        v = model @ v

        return v

    def pick_box(self, c):
        color, depth, segmentation = c.get_observation()

        np.random.shuffle(self.box_ids)
        for i in self.box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for y, x in zip(m[0], m[1]):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                return self.get_global_position_from_camera(c, depth, x, y)

        return False

    def pick_one(self, c):
        right_bound = -0.4
        left_bound = 0.4
        bottom_bound = -0.3
        top_bound = 0.3

        spade_width = 0.1
        discrete = 100

        delta = (top_bound - bottom_bound)*1. /discrete
        half_grids = int((spade_width / 2) / delta)

        color, depth, segmentation = c.get_observation()
        cnt_map = np.zeros((discrete, ))

        done = True

        for i in self.box_ids:
            m = np.where(segmentation == i)
            if len(m[0]):
                min_x, max_x = np.min(m[1]), np.max(m[1])
                min_y, max_y = np.min(m[0]), np.max(m[0])
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                p_in_world = self.get_global_position_from_camera(c, depth, x, y)[:3]
                x_in_w, y_in_w, z_in_w = p_in_world

                print('id: {} p: {}'.format(i, p_in_world))
                if x_in_w > bottom_bound and x_in_w < top_bound and y_in_w < left_bound and y_in_w > right_bound:

                    done = False
                    grid_id = int((x_in_w - bottom_bound)/ delta)

                    for k in range(grid_id - half_grids, grid_id + half_grids+1):
                        if k > 0 and k < discrete:
                            cnt_map[k] += 1

        # print('debug cnter', cnt_map)

        if done:
            return None, True
        else:
            res = []
            for k in range(discrete):
                if k not in self.marks:
                    res.append([k, cnt_map[k]])

            if not res:
                return None, True

            res = sorted(res, key=lambda x: x[1], reverse=True)
            assert res[0][1] > 0, 'must some error with the counter map.'
            pick_id = res[0][0]
            self.marks.add(pick_id)
            selected_x = bottom_bound+pick_id*delta+delta*0.5

            print('selected x: ', selected_x)
            return selected_x, False

    # Use top camera to get the box location
    # return False when cannot find it
    def observe_box(self, box_idx):
        r1, r2, c1, c2, c3, c4 = self.env.get_agents()
        # convert the box_idx in [0,9] to the id in the sim env
        box_id = self.box_ids[box_idx]
        for c in [c4]:
            # read the observation from the camera
            color, depth, segmentation = c.get_observation()
            # get the segmentation mask
            m = np.where(segmentation == box_id)
            if len(m[0]):
                # get the x-y range of the mask
                min_x = 10000
                max_x = -1
                min_y = 10000
                max_y = -1
                for y, x in zip(m[0], m[1]):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                # get the middle pixel from the range
                x, y = round((min_x + max_x) / 2), round((min_y + max_y) / 2)
                position = self.get_global_position_from_camera(c, depth, x, y)
                return position
            else:
                return False 

    # TODO: add inside bin judgement 

    def update_box_state(self):
        self.box_location_update_flag = False
        for box_idx in range(10):
            box_idx_location = self.observe_box(box_idx)
            if box_idx_location is not False:
                box_idx_location = box_idx_location[:3]
            else:
                continue
            # if cannot find the box
            if box_idx_location[0] == None:
                self.box_location[box_idx,:3] = np.one(3)*100 
                self.box_state[box_idx] = -1         
            else:
                self.box_location[box_idx,:3] = box_idx_location
                self.box_state[box_idx] = 0

    def locate_bin_bbox(self, c):
        '''
        :param c:
        :return:
        '''
        color, depth, segmentation = c.get_observation()
        mask = segmentation == self.bin_id

        cm = c.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        xf = (np.arange(w) + 0.5) / w
        yf = 1 - (np.arange(h) + 0.5) / h

        gx, gy = np.meshgrid(xf, yf, )
        # get 0 to 1 coordinate for (x, y) coordinates

        ndc = np.stack([gx, gy, depth, np.ones(depth.shape)], axis=2) * 2 - 1
        # get the -1 to 1 (x,y,z) coordinates
        ndc = np.expand_dims(ndc, axis=3)

        # transform from image space to view space
        unproj = np.linalg.inv(proj)
        unproj = np.reshape(unproj, (1, 1, 4, 4))

        v = np.matmul(unproj, ndc)
        v = v / v[:, :, 3:4, 0:1]

        # transform from view space to world space
        model = np.reshape(model, (1, 1, 4, 4))
        points = np.matmul(model, v)[..., :3, 0]
        #points in the world coordinate

        # _, axs = plt.subplots(1, 3)
        # for i in range(3):
        #     axs[i].imshow(points[..., i])
        # plt.show()

        z_coord_masked = points[..., 2] * mask.astype(np.float32)
        max_height = z_coord_masked.max()
        #find the top regions

        canvas = np.zeros(depth.shape)
        canvas[np.abs(z_coord_masked-max_height)<0.0001] = 255

        h_s, w_s = np.where(canvas == 255)
        bin_top_area = np.stack([w_s, h_s], axis=1)
        rbox = cv2.minAreaRect(bin_top_area)
        center, size, rot = rbox
        # (w_in_im, h_in_im), (), (height, width)
        corners = cv2.boxPoints(rbox)
        # (x, y), x = w_in_im, y = h_in_im


        center = np.array(center, np.int)
        h_in_im, w_in_im = center[1], center[0]

        # plt.figure()
        # plt.imshow(color)
        # for p in corners:
        #     plt.plot(p[0], p[1], 'ro')
        # plt.plot(w_in_im, h_in_im, 'o', c=(0., 1., 0.))
        # plt.show()

        bin_top_center = np.array([points[h_in_im, w_in_im, 0],
                                   points[h_in_im, w_in_im, 1],
                                   max_height])

        self.basic_info['bin_center'] = bin_top_center
        self.basic_info['bin_orientation'] = rot
        self.basic_info['bin_corner'] = corners
 

    

if __name__ == '__main__':
    np.random.seed(4)
    env = FinalEnv()
    env.run(Solution(), render=True, render_interval=25, debug=True)
    # env.run(Solution())
    env.close()
