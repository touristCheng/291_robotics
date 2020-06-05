import sapien.core as sapien


def robot_basic_control_demo(fix_robot_root, balance_passive_force, add_joint_damping):
    sim = sapien.Engine()
    renderer = sapien.OptifuserRenderer()
    renderer.enable_global_axes(False)
    sim.set_renderer(renderer)
    renderer_controller = sapien.OptifuserController(renderer)
    renderer_controller.set_camera_position(-3, 0, 3)
    renderer_controller.set_camera_rotation(0, -0.5)

    scene0 = sim.create_scene(gravity=[0, 0, -9.81])
    renderer_controller.set_current_scene(scene0)
    scene0.add_ground(altitude=0)
    scene0.set_timestep(1 / 240)
    scene0.set_ambient_light([0.5, 0.5, 0.5])
    scene0.set_shadow_light([0, 1, -1], [0.5, 0.5, 0.5])

    loader = scene0.create_urdf_loader()
    loader.fix_root_link = fix_robot_root
    robot = loader.load("assets/robot/panda_spade.urdf")
    # robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    robot.set_root_pose(sapien.Pose([-0.5, -0.25, 0.6], [0.707107, 0, 6.98492e-09, 0.707107]))

    arm_init_qpos = [4.71, 2.84, 0, 0.75, 4.62, 4.48, 4.88]
    arm_init_qpos = [-1.3641514,   0.8450767,  -0.05523842, -1.914256,    1.5603718,   1.8646374, -1.9526998 ]


    gripper_init_qpos = [0, 0, 0, 0, 0, 0]
    init_qpos = arm_init_qpos + gripper_init_qpos
    robot.set_qpos(init_qpos)

    if add_joint_damping:
        for joint in robot.get_joints():
            joint.set_drive_property(stiffness=0, damping=10)

    steps = 0
    renderer_controller.show_window()
    while not renderer_controller.should_quit:
        scene0.update_render()
        for i in range(4):
            if balance_passive_force:
                qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True, external=False)
                robot.set_qf(qf)
            scene0.step()
            steps += 1
        renderer_controller.render()

    scene0 = None


if __name__ == '__main__':
    robot_basic_control_demo(fix_robot_root=True, balance_passive_force=True, add_joint_damping=True)