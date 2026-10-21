"""
MimicGen interface for RB-Y1 CloseDrawerLeft (mobipi.envs) using left-arm WholeBodyIK.

Same TCP <-> wrist conversion as MG_RBY1_CloseSingleDoor. Differences:

- object: the drawer handle
- subtask signal: robot contact with the drawer itself (the bodies moved by its
  slide joint, not the surrounding cabinet trim)
- action: the 25-D WholeBody action the drawer demos are recorded with
  (mobipi/collect_demos/auto_close_drawer.py):

      right (6)         absolute WBIK pose of gripper0_right_ft_frame, held
      left (6)          absolute WBIK pose of gripper0_left_ft_frame  <- MimicGen
      torso (6)         absolute joint positions, held (task squat)
      head (2)          absolute joint positions, held (turned toward the drawer)
      base (3)          velocity, zero
      right_gripper (1), left_gripper (1)

  The held values are captured at the first step of every generated episode,
  like the demo script captures them after reset.
"""

import numpy as np

from mimicgen.env_interfaces.robocasa.single_stage.mg_rby1_close_single_door import MG_RBY1_CloseSingleDoor


class MG_RBY1_CloseDrawerLeft(MG_RBY1_CloseSingleDoor):

    RIGHT_WRIST_SITE = "gripper0_right_ft_frame"
    HOLD_PARTS = ("torso", "head")

    def _fixture(self):
        return self.env.drawer

    def get_object_poses(self):
        return {
            "handle": self.get_object_pose(self.env.drawer.handle_name, obj_type="geom"),
        }

    def _drawer_body_ids(self):
        """Bodies moved by the drawer's slide joint (the drawer box, front and handle)."""
        model = self.env.sim.model
        joint_id = model.joint_name2id(f"{self.env.drawer.name}_slidejoint")
        root = int(model.jnt_bodyid[joint_id])
        ids = set()
        for body_id in range(model.nbody):
            current = body_id
            while current != 0:
                if current == root:
                    ids.add(body_id)
                    break
                current = int(model.body_parentid[current])
        return ids

    def _robot_touching_drawer(self):
        model, data = self.env.sim.model, self.env.sim.data
        drawer_bodies = self._drawer_body_ids()
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            robot1 = (model.geom_id2name(g1) or "").startswith("robot0_")
            robot2 = (model.geom_id2name(g2) or "").startswith("robot0_")
            if (robot1 and int(model.geom_bodyid[g2]) in drawer_bodies) or \
                    (robot2 and int(model.geom_bodyid[g1]) in drawer_bodies):
                return True
        return False

    def get_subtask_term_signals(self):
        return {
            "stage_contact_drawer": int(self._robot_touching_drawer()),
            "success": int(self.env._check_success()),
        }

    def _capture_holds(self):
        """Commands that keep every part except the left arm where it starts."""
        robot = self.env.robots[0]
        right_wrist = self._pose_from_site_raw(self.RIGHT_WRIST_SITE)
        holds = {
            "right": np.concatenate([right_wrist[:3, 3], self._rotvec(right_wrist[:3, :3])]),
            "right_gripper": np.array([1.0]),
        }
        for part in self.HOLD_PARTS:
            controller = robot.part_controllers[part]
            holds[part] = self.env.sim.data.qpos[np.asarray(controller.qpos_index, dtype=int)].copy()
        return holds

    @staticmethod
    def _rotvec(rot):
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(rot).as_rotvec()

    def compose_action(self, left_arm_target, gripper_action):
        """Full 25-D WholeBody action: left WBIK target + gripper, everything else held."""
        left_arm_target = np.asarray(left_arm_target, dtype=float).reshape(-1)
        left_gripper = np.asarray(gripper_action, dtype=float).reshape(-1)
        if left_arm_target.size != 6 or left_gripper.size != 1:
            raise ValueError(f"Expected 6-D left WBIK target and 1-D gripper, got "
                             f"{left_arm_target.shape} and {left_gripper.shape}")

        # new episode (MimicGen resets the env before each generation attempt)
        if self.env.timestep == 0 or getattr(self, "_holds", None) is None:
            self._holds = self._capture_holds()

        action_dict = dict(self._holds, left=left_arm_target, left_gripper=left_gripper)
        action = self.env.robots[0].create_action_vector(action_dict)
        if action.shape != (self.env.action_dim,):
            raise RuntimeError(f"Expected {self.env.action_dim}-D WholeBody action, got {action.shape}")
        return action
