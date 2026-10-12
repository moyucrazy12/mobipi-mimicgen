# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
A collection of classes used to represent waypoints and trajectories.
"""
import json
import numpy as np
from copy import deepcopy

import mimicgen
import mimicgen.utils.pose_utils as PoseUtils


class Waypoint(object):
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """
    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (np.array): 4x4 pose target for robot controller
            gripper_action (np.array): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        self.pose = np.array(pose)
        self.gripper_action = np.array(gripper_action)
        self.noise = noise
        assert len(self.gripper_action.shape) == 1


class WaypointSequence(object):
    """
    Represents a sequence of Waypoint objects.
    """
    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be an list of Waypoint objects
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses, 
        gripper actions, and action noise.

        Args:
            poses (np.array): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (np.array): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or np.array): sequence of action noise
                magnitudes that should be applied at each timestep. If a 
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        assert isinstance(action_noise, float) or isinstance(action_noise, np.ndarray)

        # handle scalar to numpy array conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * np.ones((num_timesteps, 1))
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)


class WaypointTrajectory(object):
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """
    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.
        
        Returns:
            waypoint (Waypoint instance)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (np.array): 4x4 target pose

            gripper_action (np.array): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        if (len(self.waypoint_sequences) == 0):
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = np.array([pose for _ in range(num_steps)])
            gripper_actions = np.array([gripper_action for _ in range(num_steps)])
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            poses, num_steps_2 = PoseUtils.interpolate_poses(
                pose_1=last_waypoint.pose,
                pose_2=pose,
                num_steps=num_steps,
            )
            assert num_steps == num_steps_2
            gripper_actions = np.array([gripper_action for _ in range(num_steps + 2)])
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant 
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = (need_interp or need_fixed)

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def execute(
        self, 
        env,
        env_interface, 
        render=False, 
        video_writer=None, 
        video_skip=5, 
        camera_names=None,
        actuation_mode="arm",
        fixed_arm_pose=None
    ):
        """
        Main function to execute the trajectory. Will use env_interface.target_pose_to_action to
        convert each target pose at each waypoint to an action command, and pass that along to
        env.step.

        Args:
            env (robomimic EnvBase instance): environment to use for executing trajectory
            env_interface (MG_EnvInterface instance): environment interface for executing trajectory
            render (bool): if True, render on-screen
            video_writer (imageio writer): video writer
            video_skip (int): determines rate at which environment frames are written to video
            camera_names (list): determines which camera(s) are used for rendering. Pass more than
                one to output a video with multiple camera views concatenated horizontally.
            actuation_moe (str): whether to actuate for "base" or "arm".
            fixed_arm_pose (list): optinal fixed target arm pose for base actuation mode.

        Returns:
            results (dict): dictionary with the following items for the executed trajectory:
                states (list): simulator state at each timestep
                observations (list): observation dictionary at each timestep
                datagen_infos (list): datagen_info at each timestep
                actions (list): action executed at each timestep
                success (bool): whether the trajectory successfully solved the task or not
        """
        write_video = (video_writer is not None)
        video_count = 0

        states = []
        actions = []
        observations = []
        datagen_infos = []
        success = { k: False for k in env.is_success() } # success metrics

        prev_wp_pos = None
        prev_joint_target = None
        global_t = 0

        # iterate over waypoint sequences
        for seq_ind, seq in enumerate(self.waypoint_sequences):

            # iterate over waypoints in each sequence
            for j in range(len(seq)):

                # on-screen render
                if render:
                    env.render(mode="human", camera_name=camera_names[0])

                # video render
                if write_video:
                    if video_count % video_skip == 0:
                        video_img = []
                        for cam_name in camera_names:
                            video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                        video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                        video_writer.append_data(video_img)
                    video_count += 1

                # current waypoint
                waypoint = seq[j]

                wp_pos = waypoint.pose[:3, 3]
                wp_jump = 0.0 if prev_wp_pos is None else np.linalg.norm(wp_pos - prev_wp_pos)

                # current state and obs
                state = env.get_state()["states"]
                obs = env.get_observation()

                # convert target pose to arm action
                action_pose = env_interface.target_pose_to_action(target_pose=waypoint.pose)

                joint_jump = 0.0 if prev_joint_target is None else np.max(np.abs(action_pose - prev_joint_target))

                if wp_jump > 0.03 or joint_jump > 0.2:
                    print(
                        f"[MG DEBUG] t={global_t:4d} seq={seq_ind} "
                        f"wp_jump={wp_jump*1000:7.2f}mm "
                        f"joint_jump={joint_jump:7.3f} "
                        f"wp={np.round(wp_pos, 3)}"
                    )

                prev_wp_pos = wp_pos.copy()
                prev_joint_target = action_pose.copy()
                global_t += 1

                if actuation_mode == "arm":
                    # RB-Y1 can override the stock action layout. The custom interface
                    # converts MimicGen's Cartesian waypoint to a 7-DoF absolute
                    # left-arm target and then builds the full 26-D composite action.
                    if hasattr(env_interface, "compose_action"):
                        if waypoint.noise is not None:
                            action_pose = action_pose + waypoint.noise * np.random.randn(*action_pose.shape)
                        play_action = env_interface.compose_action(
                            left_arm_target=action_pose,
                            gripper_action=waypoint.gripper_action,
                        )
                    else:
                        # maybe add noise to action
                        if waypoint.noise is not None:
                            action_pose += waypoint.noise * np.random.randn(*action_pose.shape)
                            action_pose = np.clip(action_pose, -1., 1.)

                        # add in gripper action
                        play_action = np.concatenate([action_pose, waypoint.gripper_action], axis=0)

                        # mobile base action
                        play_action = np.concatenate([play_action, [0, 0, 0, 0], [-1]], axis=0)

                elif actuation_mode == "base":
                        base_action = env_interface.target_base_pose_to_action(target_base_pose=waypoint.pose)
                        if waypoint.noise is not None:
                            base_action += waypoint.noise * np.random.randn(*base_action.shape)
                            base_action = np.clip(base_action, -1., 1.)

                        if fixed_arm_pose is not None:
                            base_pose = PoseUtils.make_pose(*env_interface.get_controller_base_pose())
                            fixed_arm_pose_wrt_world = PoseUtils.pose_in_A_to_pose_in_B(fixed_arm_pose, base_pose)
                            arm_action = env_interface.target_pose_to_action(target_pose=fixed_arm_pose_wrt_world)
                        else:
                            arm_action = np.zeros([6])

                        # add in gripper action
                        play_action = np.concatenate([arm_action, waypoint.gripper_action, base_action, [0, -1]], axis=0)

                # store datagen info too
                datagen_info = env_interface.get_datagen_info(action=play_action)

                # step environment
                env.step(play_action)

                # collect data
                states.append(state)
                play_action_record = play_action
                actions.append(play_action_record)
                observations.append(obs)
                datagen_infos.append(datagen_info)

                cur_success_metrics = env.is_success()
                for k in success:
                    success[k] = success[k] or cur_success_metrics[k]

        results = dict(
            states=states,
            observations=observations,
            datagen_infos=datagen_infos,
            actions=np.array(actions),
            success=bool(success["task"]),
        )
        return results
