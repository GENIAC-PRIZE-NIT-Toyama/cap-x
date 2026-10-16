from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation as SciRotation

from capx.utils.serve_utils import post_with_retries

DEFAULT_URL = os.environ.get("PYROKI_SERVICE_URL", "http://127.0.0.1:8116")


def init_pyroki(
    server_url: str = DEFAULT_URL,
) -> None:
    """
    A *drop-in replacement* for your original init_pyroki(), but instead of
    loading PyRoKi locally, it forwards IK + planning requests to the remote server.

    Downstream code calling ik_solve_fn() or plan_fn() works identically.
    """

    # Normalize trailing slash
    server_url = server_url.rstrip("/")

    # =====================================================
    # IK SOLVER WRAPPER
    # =====================================================
    def ik_solve_fn(
        target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Identical signature to the local solver, but sends JSON to /ik.
        """

        payload = {
            "target_pose_wxyz_xyz": target_pose_wxyz_xyz.tolist(),
            "prev_cfg": prev_cfg.tolist() if prev_cfg is not None else None,
        }

        data = post_with_retries(f"{server_url}/ik", payload, timeout_seconds=15.0)
        joints = np.asarray(data["joint_positions"], dtype=np.float32)

        return joints

    return ik_solve_fn

    # =====================================================
    # PLANNING WRAPPER
    # =====================================================

def init_pyroki_ik_rest(
    robot: str,
    rest_cost_weights: dict[str, float] | None = None,
    server_url: str = DEFAULT_URL,
) -> Callable[..., np.ndarray]:
    """Rest-cost IK for a URDF robot served by the PyRoKi server (e.g. "r1pro").

    The returned function takes the target link, the target pose and the
    current actuated joint positions (server's joint order) and returns the
    solved actuated joint positions in the same order.
    """
    server_url = server_url.rstrip("/")
    weights = dict(rest_cost_weights or {})

    def ik_rest_fn(
        target_link: str,
        target_wxyz: np.ndarray,
        target_position: np.ndarray,
        initial_q: np.ndarray,
    ) -> np.ndarray:
        payload = {
            "robot": robot,
            "target_link": target_link,
            "target_pose_wxyz_xyz": [*map(float, target_wxyz), *map(float, target_position)],
            "initial_q": [float(q) for q in initial_q],
            "rest_cost_weights": weights,
        }
        # The first call per robot includes the JIT compile on the server.
        data = post_with_retries(f"{server_url}/ik_rest", payload, timeout_seconds=60.0)
        return np.asarray(data["joint_positions"], dtype=np.float64)

    return ik_rest_fn


def init_pyroki_fk(robot: str, server_url: str = DEFAULT_URL) -> Callable[[np.ndarray], np.ndarray]:
    """Forward kinematics for a URDF robot served by the PyRoKi server.

    The returned function maps actuated joint positions to an (num_links, 7)
    array of wxyz_xyz link poses, like ``pk.Robot.forward_kinematics``.
    """
    server_url = server_url.rstrip("/")

    def fk_fn(joint_positions: np.ndarray) -> np.ndarray:
        payload = {"robot": robot, "joint_positions": [float(q) for q in joint_positions]}
        data = post_with_retries(f"{server_url}/fk", payload, timeout_seconds=60.0)
        return np.asarray(data["link_poses_wxyz_xyz"], dtype=np.float64)

    return fk_fn


def init_pyroki_trajopt(
    server_url: str = DEFAULT_URL,
) -> None:
    """
    A *drop-in replacement* for your original init_pyroki_trajopt(), but instead of
    loading PyRoKi locally, it forwards IK + planning requests to the remote server.
    """

    def trajopt_plan_fn(
        start_pose_wxyz_xyz: np.ndarray, end_pose_wxyz_xyz: np.ndarray
    ) -> np.ndarray:
        """
        Identical signature to the local solver, but sends JSON to /plan.
        """
        payload = {
            "start_pose_wxyz_xyz": start_pose_wxyz_xyz.tolist(),
            "end_pose_wxyz_xyz": end_pose_wxyz_xyz.tolist(),
        }
        data = post_with_retries(f"{server_url}/plan", payload)
        waypoints = np.asarray(data["waypoints"], dtype=np.float32)
        return waypoints

    return trajopt_plan_fn

    # def plan_fn(
    #     q_start: np.ndarray,
    #     q_goal: np.ndarray,         # ignored — kept only for API-compat
    #     obstacles: list[dict[str, Any]],
    # ) -> dict[str, Any]:
    #     """
    #     Matches your original signature even though the PyRoKi server ignores `q_goal`.

    #     Returns:
    #         { "waypoints": [...], "dt": float }
    #     """

    #     payload = {
    #         "q_start": np.asarray(q_start, dtype=np.float64).tolist(),
    #         "obstacles": obstacles,
    #     }

    #     data = _post_with_retries(f"{server_url}/plan", payload)

    #     return {
    #         "waypoints": data["waypoints"],
    #         "dt": data["dt"],
    #     }

    # # =====================================================
    # # TRAJECTORY EXECUTION (stub)
    # # =====================================================
    # def exec_traj_fn(traj: dict[str, Any]) -> bool:
    #     return True

    # =====================================================
    # Register identical API hooks
    # =====================================================
    # register_ik(ik_solve_fn)
    # register_motion_planner(plan_fn, exec_traj_fn)
