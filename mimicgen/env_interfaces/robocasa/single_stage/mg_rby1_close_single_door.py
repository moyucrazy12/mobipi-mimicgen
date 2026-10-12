"""
MimicGen interface for RB-Y1 CloseSingleDoor using left-arm WholeBodyIK.

MimicGen logical EEF:
    gripper0_left_grip_site      (TCP)

WholeBodyIK controlled frame:
    gripper0_left_ft_frame       (wrist)

Recorded WBC action:
    total: 31-D
    left WBIK target: 6-D absolute [x, y, z, rx, ry, rz]
    left gripper: 1-D

The interface converts:
    TCP target <-> wrist WBIK target

before decoding / composing environment actions.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.robosuite import RobosuiteInterface


class MG_RBY1_CloseSingleDoor(RobosuiteInterface):

    TCP_SITE = "gripper0_left_grip_site"
    WRIST_SITE = "gripper0_left_ft_frame"

    def _fixture(self):
        fxtr = getattr(self.env, "fxtr", None)
        if fxtr is None:
            fxtr = getattr(self.env, "door_fxtr", None)
        if fxtr is None:
            raise AttributeError(
                "Could not find CloseSingleDoor fixture as env.fxtr or env.door_fxtr"
            )
        return fxtr

    def _raw_model_data(self):
        model = getattr(self.env.sim.model, "_model", self.env.sim.model)
        data = getattr(self.env.sim.data, "_data", self.env.sim.data)
        return model, data

    def _handle_geom_name(self):
        fxtr = self.env.door_fxtr
        handle_name = getattr(fxtr, "handle_name", None)

        candidates = []

        if handle_name is not None:
            candidates += [
                handle_name,
                f"{handle_name}_main",
                f"{handle_name}_handle",
            ]

        candidates += [
            f"{fxtr.name}_door_handle",
            f"{fxtr.name}_door_handle_main",
            f"{fxtr.name}_door_handle_handle",
        ]

        candidates = list(dict.fromkeys(candidates))

        for name in candidates:
            try:
                self.env.sim.model.geom_name2id(name)
                return name
            except ValueError:
                pass

        raise ValueError(
            f"Could not find door handle geom. Tried: {candidates}"
        )

    def _pose_from_site_raw(self, site_name):
        model, data = self._raw_model_data()

        sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if sid < 0:
            raise ValueError(f"Missing site: {site_name}")

        return PoseUtils.make_pose(
            data.site_xpos[sid].copy(),
            data.site_xmat[sid].reshape(3, 3).copy(),
        )

    def _tcp_to_wrist_transform(self):
        """
        Rigid transform T_tcp_wrist.

        T_world_wrist = T_world_tcp @ T_tcp_wrist
        """
        T_world_tcp = self._pose_from_site_raw(self.TCP_SITE)
        T_world_wrist = self._pose_from_site_raw(self.WRIST_SITE)

        return np.linalg.inv(T_world_tcp) @ T_world_wrist

    def get_robot_eef_pose(self):
        """
        MimicGen works in the TCP frame.
        """
        return self._pose_from_site_raw(self.TCP_SITE)

    def get_controller_base_pose(self):
        body_id = self.env.sim.model.body_name2id("robot0_base")
        base_pos = np.array(
            self.env.sim.data.body_xpos[body_id]
        ).copy()

        base_rot = np.array(
            self.env.sim.data.body_xmat[body_id]
        ).reshape(3, 3).copy()

        return base_pos, base_rot

    def get_object_poses(self):
        return {
            "handle": self.get_object_pose(
                self._handle_geom_name(),
                obj_type="geom",
            )
        }

    def _robot_touching_door(self):
        """
        Return True when any robot geom contacts the moving microwave door.

        The microwave door is identified from its hinge joint rather than
        from fragile geom-name matching.
        """
        fxtr = self._fixture()

        model = self.env.sim.model
        data = self.env.sim.data

        # Same joint used by RoboCasa to determine microwave door state.
        joint_name = f"{fxtr.name}_microjoint"

        try:
            joint_id = model.joint_name2id(joint_name)
        except ValueError as exc:
            raise RuntimeError(
                f"Could not find microwave door joint '{joint_name}'"
            ) from exc

        # Body directly actuated by the microwave hinge.
        door_root_body_id = int(model.jnt_bodyid[joint_id])

        # Include the door body and any children attached to it.
        door_body_ids = set()

        for body_id in range(model.nbody):
            current = body_id

            while current != 0:
                if current == door_root_body_id:
                    door_body_ids.add(body_id)
                    break

                current = int(model.body_parentid[current])

        # Check every active MuJoCo contact.
        for i in range(data.ncon):
            contact = data.contact[i]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            name1 = model.geom_id2name(geom1) or ""
            name2 = model.geom_id2name(geom2) or ""

            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])

            robot1 = name1.startswith("robot0_")
            robot2 = name2.startswith("robot0_")

            door1 = body1 in door_body_ids
            door2 = body2 in door_body_ids

            if robot1 and door2:
                return True

            if robot2 and door1:
                return True

        return False


    def get_subtask_term_signals(self):
        return {
            "stage_contact_door": int(self._robot_touching_door()),
            "success": int(self.env._check_success()),
        }

    # ------------------------------------------------------------------
    # WBC action parsing
    # ------------------------------------------------------------------

    def _split_action(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)

        controller = self.env.robots[0].composite_controller

        splits = getattr(
            controller,
            "_whole_body_controller_action_split_indexes",
            None,
        )

        if splits is None or len(splits) == 0:
            raise RuntimeError(
                "WholeBody controller has no "
                "_whole_body_controller_action_split_indexes"
            )

        expected_dim = max(
            int(end) for _, end in splits.values()
        )

        if expected_dim != action.size:
            raise ValueError(
                f"WholeBody split expects {expected_dim} values, "
                f"but action has {action.size}"
            )

        return {
            name: action[int(start):int(end)]
            for name, (start, end) in splits.items()
        }

    # ------------------------------------------------------------------
    # Recorded action -> MimicGen TCP target
    # ------------------------------------------------------------------

    def action_to_target_pose(self, action, relative=True):
        """
        Recover the TCP target represented by a recorded 31-D WBC action.

        The recorded left-arm action is an ABSOLUTE wrist target:
            [x, y, z, rx, ry, rz]

        but MimicGen operates using TCP targets, so convert:

            world->wrist  ->  world->TCP
        """
        parts = self._split_action(action)

        if "left" not in parts:
            raise ValueError(
                f"No left-arm action found. "
                f"Available parts: {list(parts.keys())}"
            )

        arm_action = np.asarray(
            parts["left"],
            dtype=float,
        ).reshape(-1)

        if arm_action.size != 6:
            raise ValueError(
                f"Expected 6-D left WBIK target, "
                f"got {arm_action.shape}"
            )

        wrist_pos = arm_action[:3]
        wrist_rot = Rotation.from_rotvec(
            arm_action[3:6]
        ).as_matrix()

        T_world_wrist = PoseUtils.make_pose(
            wrist_pos,
            wrist_rot,
        )

        T_tcp_wrist = self._tcp_to_wrist_transform()

        T_world_tcp = (
            T_world_wrist
            @ np.linalg.inv(T_tcp_wrist)
        )

        return T_world_tcp

    def action_to_gripper_action(self, action):
        parts = self._split_action(action)

        if "left_gripper" not in parts:
            raise ValueError(
                f"No left_gripper action found. "
                f"Available parts: {list(parts.keys())}"
            )

        return np.asarray(
            parts["left_gripper"],
            dtype=float,
        ).reshape(-1)

    # ------------------------------------------------------------------
    # MimicGen TCP target -> WBC action
    # ------------------------------------------------------------------

    def target_pose_to_action(self, target_pose, relative=True):
        """
        Convert a MimicGen TCP target into the absolute 6-D wrist target
        expected by the left-arm WholeBodyIK controller.

        MimicGen:
            T_world_tcp

        Controller:
            T_world_wrist

        with:
            T_world_wrist = T_world_tcp @ T_tcp_wrist
        """
        target_pose = np.asarray(
            target_pose,
            dtype=float,
        ).reshape(4, 4)

        T_tcp_wrist = self._tcp_to_wrist_transform()

        T_world_wrist = (
            target_pose
            @ T_tcp_wrist
        )

        wrist_pos, wrist_rot = PoseUtils.unmake_pose(
            T_world_wrist
        )

        wrist_rotvec = Rotation.from_matrix(
            wrist_rot
        ).as_rotvec()

        return np.concatenate([
            wrist_pos,
            wrist_rotvec,
        ])

    # ------------------------------------------------------------------
    # Left WBIK target + gripper -> full 31-D env action
    # ------------------------------------------------------------------

    def compose_action(self, left_arm_target, gripper_action):
        """
        Build the same 31-D WholeBodyIK action structure used during
        demonstration collection.

        During collection we supplied:
            - right arm: absolute joint hold
            - left arm: 6-D absolute WBIK wrist target
            - both grippers

        Other WBC fields are left unspecified, so create_action_vector()
        fills them exactly as it did during collection.
        """
        robot = self.env.robots[0]

        left_arm_target = np.asarray(
            left_arm_target,
            dtype=float,
        ).reshape(-1)

        if left_arm_target.size != 6:
            raise ValueError(
                f"Expected 6-D left WBIK target, "
                f"got {left_arm_target.shape}"
            )

        left_gripper = np.asarray(
            gripper_action,
            dtype=float,
        ).reshape(-1)

        if left_gripper.size != 1:
            raise ValueError(
                f"Expected 1-D left gripper action, "
                f"got {left_gripper.shape}"
            )

        right_ctrl = robot.part_controllers["right"]
        right_qpos_idx = np.asarray(
            right_ctrl.qpos_index,
            dtype=int,
        )

        right_hold_q = self.env.sim.data.qpos[
            right_qpos_idx
        ].copy()

        action_dict = {
            "right": right_hold_q,
            "left": left_arm_target,
            "right_gripper": np.array([1.0]),
            "left_gripper": left_gripper,
        }

        action = robot.create_action_vector(
            action_dict
        )

        if action.shape != (31,):
            raise RuntimeError(
                f"Expected 31-D WholeBodyIK action, "
                f"got {action.shape}"
            )

        return action