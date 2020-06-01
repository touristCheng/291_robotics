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
        self.phase = 0
        self.drive = 0
        meta = env.get_metadata()

        self.box_ids = meta['box_ids']
        self.bin_id = meta['bin_id']

        r1, r2, c1, c2, c3, c4 = env.get_agents()

        r1_meta = r1.get_metadata()
        self.left_spade_id = r1_meta['link_ids'][-1]

        r2_meta = r2.get_metadata()
        self.right_spade_id = r2_meta['link_ids'][-1]

        self.basic_info = {}
        self.locate_bin_bbox(c4)
        self.measured = False


        self.ps = [1000, 800, 600, 600, 200, 200, 100]
        self.ds = [1000, 800, 600, 600, 200, 200, 100]
        r1.configure_controllers(self.ps, self.ds)
        r2.configure_controllers(self.ps, self.ds)

    def act(self, env: FinalEnv, current_timestep: int):
        r1, r2, c1, c2, c3, c4 = env.get_agents()

        pf_left = f = r1.get_compute_functions()['passive_force'](True, True, False)
        pf_right = f = r2.get_compute_functions()['passive_force'](True, True, False)

        if self.phase == 0:
            t1 = [2, 1, 0, -1.5, -1, 1, -2]
            t2 = [-2, 1, 0, -1.5, 1, 1, -2]

            r1.set_action(t1, [0] * 7, pf_left)
            r2.set_action(t2, [0] * 7, pf_right)

            if np.allclose(r1.get_observation()[0], t1, 0.05, 0.05) and np.allclose(
                    r2.get_observation()[0], t2, 0.05, 0.05):
                self.phase = 1
                self.counter = 0
                self.selected_x = None


        if self.phase == 1:
            self.counter += 1

            if (self.counter == 1):
                selected = self.pick_box(c4)
                self.selected_x = selected[0]
                if self.selected_x is None:
                    return False

            target_pose_left = Pose([self.selected_x, 0.5, 0.67], euler2quat(np.pi, -np.pi / 3, -np.pi / 2))
            self.diff_drive(r1, 9, target_pose_left)

            target_pose_right = Pose([self.selected_x, -0.5, 0.6], euler2quat(np.pi, -np.pi / 3, np.pi / 2))
            self.diff_drive(r2, 9, target_pose_right)

            if self.counter == 2000 / 5:
                self.phase = 2
                self.counter = 0

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.07
                self.pose_left = Pose(p, q)

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = -0.07
                self.pose_right = Pose(p, q)

        if self.phase == 2:
            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)
            if self.counter == 2000 / 5:
                self.phase = 3

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[2] += 0.2
                self.pose_right = Pose(p, q)

                pose = r1.get_observation()[2][9]
                p, q = pose.p, pose.q
                p[1] = 0.5
                q = euler2quat(np.pi, -np.pi / 2, -np.pi / 2)
                self.pose_left = Pose(p, q)

                self.counter = 0

        if self.phase == 3:
            self.counter += 1
            self.diff_drive(r1, 9, self.pose_left)
            self.diff_drive(r2, 9, self.pose_right)
            if self.counter == 200 / 5:
                self.phase = 4
                self.counter = 0

        if self.phase == 4:
            self.counter += 1
            if (self.counter < 3000 / 5):
                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 0, -1, 0], [0, 1, 2, 3, 4])

            elif (self.counter < 6000 / 5):

                pose = r2.get_observation()[2][9]
                p, q = pose.p, pose.q
                q = euler2quat(np.pi, -np.pi / 1.5, quat2euler(q)[2])
                self.diff_drive2(r2, 9, Pose(p, q), [4, 5, 6], [0, 0, 1, -1, 0], [0, 1, 2, 3, 4])

                if self.counter > 5000 / 5 and not self.measured:
                    print('Measuring spade size ...')
                    self.measure_spade(c4, r2)
                    self.calc_target_pose(c4)

            elif (self.counter < 9000 / 5):
                self.freeze(c4, 'phase 4, last')

                p = [-1, 0, 1.5]
                print(p)
                p = self.basic_info['bin_top_center']
                print(p)
                q = euler2quat(0, -np.pi / 1.5, 0)


                self.diff_drive(r2, 9, Pose(p, q))
            else:
                self.phase = 0
                # return False

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

    def project_global_position_into_camera(self, camera, p):
        '''
        use this function to check the position of joint
        #todo debug
        :param camera:
        :param p:
        :return:
        '''

        cm = camera.get_metadata()
        proj, model = cm['projection_matrix'], cm['model_matrix']
        w, h = cm['width'], cm['height']

        p_cam = np.linalg.inv(model) @ p
        p_cam /= p_cam[3]

        p_im = proj @ p_cam

        p_im = (p_im + 1.) / 2.
        p_im = p_im[:2, 0]

        x = p_im[1] * w - 0.5
        y = (1 - p_im[0]) * h - 0.5

        return np.array([x, y])

    def freeze(self, c, info = None):
        if info:
            print('info: ', info)
        color, depth, segmentation = c.get_observation()
        plt.figure()
        plt.imshow(color)
        plt.show()

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

    def calc_target_pose(self, c):
        # todo calc the final target pose

        pass

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


if __name__ == '__main__':
    np.random.seed(2)
    env = FinalEnv()
    # env.run(Solution(), render=True, render_interval=5, debug=True)
    env.run(Solution(), render=True, render_interval=5)
    env.close()
