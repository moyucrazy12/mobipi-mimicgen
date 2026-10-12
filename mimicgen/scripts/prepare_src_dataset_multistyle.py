#!/usr/bin/env python3
"""Add MimicGen datagen_info to multi-style RBY1 CloseSingleDoor demos."""

import argparse
import copy
import csv
import json

import h5py
import numpy as np
from tqdm import tqdm

import robosuite
import robocasa  # registers RoboCasa environments
import robomimic.utils.tensor_utils as TensorUtils
from mimicgen.env_interfaces.base import make_interface


def load_manifest(path):
    """Read demo, layout, style and seed from manifest.csv."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        f"demo_{int(row['demo'])}": {
            "layout": int(row["layout"]),
            "style": int(row["style"]),
            "seed": int(row["seed"]),
        }
        for row in rows
    }


def unwrap_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def make_env(metadata, cfg):
    """Recreate the demo environment using controller metadata from the HDF5."""
    kwargs = copy.deepcopy(metadata["env_kwargs"])
    kwargs.update(
        layout_ids=[cfg["layout"]],
        style_ids=[cfg["style"]],
        seed=cfg["seed"],
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera=None,
        use_camera_obs=False,
        ignore_done=True,
    )

    np.random.seed(cfg["seed"])
    env = robosuite.make(
        env_name=metadata["env_name"],
        **kwargs,
    )

    np.random.seed(cfg["seed"])
    env.reset()
    return unwrap_env(env)


def load_recorded_model(env, model_xml):
    """Load the exact recorded MuJoCo XML without triggering a RoboCasa reset."""
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode()

    xml = env.edit_model_xml(model_xml)

    env.close()
    env.deterministic_reset = True
    try:
        env._initialize_sim(xml_string=xml)

        for robot in env.robots:
            robot.reset_sim(env.sim)
            robot.setup_references()
            robot.reset(deterministic=True)

        env.sim_state_initial = env.sim.get_state()
        env.cur_time = 0
        env.timestep = 0
        env.done = False
        env.sim.forward()
    finally:
        env.deterministic_reset = False


def extract_datagen_info(env, interface, states, actions):
    """Restore every recorded state and ask the custom interface for MimicGen info."""
    infos = []

    for state, action in zip(states, actions):
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        infos.append(
            interface.get_datagen_info(action=action).to_dict()
        )

    info = TensorUtils.list_of_flat_dict_to_dict_of_list(infos)

    for key in ("object_poses", "subtask_term_signals"):
        info[key] = TensorUtils.list_of_flat_dict_to_dict_of_list(
            info[key]
        )
        info[key] = {
            name: np.asarray(value)
            for name, value in info[key].items()
        }

    for key in set(info) - {"object_poses", "subtask_term_signals"}:
        info[key] = np.asarray(info[key])

    return info


def write_datagen_info(ep, info, interface_name, interface_type):
    """Replace the episode's datagen_info group."""
    if "datagen_info" in ep:
        del ep["datagen_info"]

    root = ep.create_group("datagen_info")

    for key, value in info.items():
        if isinstance(value, dict):
            group = root.create_group(key)
            for name, array in value.items():
                group.create_dataset(name, data=array)
        else:
            root.create_dataset(key, data=value)

    root.attrs["env_interface_name"] = interface_name
    root.attrs["env_interface_type"] = interface_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--env-interface",
        default="MG_RBY1_CloseSingleDoor",
    )
    parser.add_argument(
        "--env-interface-type",
        default="robosuite",
    )
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    demo_cfg = load_manifest(args.manifest)

    with h5py.File(args.dataset, "a") as f:
        metadata = json.loads(f["data"].attrs["env_args"])

        # Controller comes directly from the recorded dataset.
        if "controller_configs" not in metadata["env_kwargs"]:
            raise RuntimeError(
                "No controller_configs in data.attrs['env_args']"
            )

        demos = sorted(
            f["data"].keys(),
            key=lambda name: int(name.split("_")[-1]),
        )

        if args.n is not None:
            demos = demos[: args.n]

        for demo in tqdm(demos, desc="Preparing MimicGen source"):
            if demo not in demo_cfg:
                raise RuntimeError(
                    f"{demo} is missing from {args.manifest}"
                )

            ep = f[f"data/{demo}"]
            cfg = demo_cfg[demo]

            states = ep["states"][()]
            actions = ep["actions"][()]

            if len(states) != len(actions):
                raise RuntimeError(
                    f"{demo}: states/actions length mismatch"
                )

            if actions.shape[1] != 31:
                raise RuntimeError(
                    f"{demo}: expected 31-D actions, got {actions.shape}"
                )

            print(
                f"\n{demo}: layout={cfg['layout']} "
                f"style={cfg['style']} seed={cfg['seed']}"
            )

            env = make_env(metadata, cfg)

            try:
                load_recorded_model(
                    env,
                    ep.attrs["model_file"],
                )

                expected_width = states.shape[1]
                actual_width = env.sim.get_state().flatten().shape[0]

                if actual_width != expected_width:
                    raise RuntimeError(
                        f"{demo}: recorded state width {expected_width} "
                        f"!= loaded XML width {actual_width}"
                    )

                # Verify the first state before processing the whole trajectory.
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()

                interface = make_interface(
                    name=args.env_interface,
                    interface_type=args.env_interface_type,
                    env=env,
                )

                info = extract_datagen_info(
                    env,
                    interface,
                    states,
                    actions,
                )

                # --------------------------------------------------
                # Convert instantaneous door contact into a latched
                # MimicGen subtask termination signal:
                #
                #   0 0 0 0 1 1 1 1 ...
                #           ^
                #      first contact
                # --------------------------------------------------
                sig = info["subtask_term_signals"]["stage_contact_door"]

                contact_inds = np.flatnonzero(sig)

                if len(contact_inds) == 0:
                    raise RuntimeError(
                        f"{demo}: no robot-door contact detected"
                    )

                boundary = int(contact_inds[0])

                sig[:] = 0
                sig[boundary:] = 1

                print(
                    f"  stage_contact_door boundary: {boundary}"
                )

                write_datagen_info(
                    ep,
                    info,
                    args.env_interface,
                    args.env_interface_type,
                )

                # Keep reconstruction metadata with the generated datagen_info.
                ep["datagen_info"].attrs.update(
                    layout=cfg["layout"],
                    style=cfg["style"],
                    seed=cfg["seed"],
                )

                f.flush()

            finally:
                env.close()

    print(f"\nPrepared {len(demos)} demos in {args.dataset}")


if __name__ == "__main__":
    main()
