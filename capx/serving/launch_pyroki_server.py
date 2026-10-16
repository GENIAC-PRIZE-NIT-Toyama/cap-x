from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, List

import numpy as np
import pyroki as pk  # type: ignore
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.spatial.transform import Rotation, Slerp

import capx.integrations.motion.pyroki_snippets as pks


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two quaternions.

    Args:
        q_start: Start quaternion in wxyz format
        q_end: End quaternion in wxyz format
        num_steps: Number of interpolation steps

    Returns:
        Array of shape (num_steps, 4) with interpolated quaternions in wxyz format
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    # Create slerp interpolator
    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    # Convert back to wxyz format
    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot: pk.Robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 25,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Args:
        robot: PyRoKi robot model
        target_link_name: Name of the end-effector link
        start_pos: Start position (3,)
        start_wxyz: Start orientation quaternion in wxyz format (4,)
        end_pos: End position (3,)
        end_wxyz: End orientation quaternion in wxyz format (4,)
        num_waypoints: Number of waypoints in the trajectory
        use_prev_cfg: If True, use previous IK solution to bias next solve
        jump_threshold: Max allowed joint change (radians) between waypoints before warning

    Returns:
        Array of shape (num_waypoints, num_joints) with joint configurations
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(15):
                cfg = pks.solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                print(
                    f"Error: {np.linalg.norm(cfg - prev_cfg)}",
                    np.allclose(cfg, prev_cfg, atol=1e-3),
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = pks.solve_ik(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped {np.degrees(joint_diff[joint_idx]):.1f}° "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    # Print warnings summary
    if jump_warnings:
        print(
            f"\nWARNING: {len(jump_warnings)} large joint jump(s) detected (threshold: {np.degrees(jump_threshold):.1f}°):"
        )
        for warning in jump_warnings:
            print(warning)
        print()

    return np.array(trajectory)


# =====================================================
# Logging and FastAPI app
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyroki_server")

app = FastAPI()

_ROBOT = None
_ROBOT_COLL = None
_TARGET_LINK = None


async def _run_in_thread(fn, *args, **kwargs):
    """Run a blocking CPU-bound function without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


# =====================================================
# Pydantic Models
# =====================================================


class IkRequest(BaseModel):
    """IK request with optional prev_cfg for vel-cost IK."""

    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    prev_cfg: list[float] | None = None  # optional


class IkResponse(BaseModel):
    joint_positions: list[float]


class ObstacleEntry(BaseModel):
    type: str
    point: list[float] | None = None
    normal: list[float] | None = None
    center: list[float] | None = None
    radius: float | None = None
    position: list[float] | None = None
    height: float | None = None
    extent: list[float] | None = None  # for box obstacles


class PlanRequest(BaseModel):
    start_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    end_pose_wxyz_xyz: list[float]  # length 7: wxyz quaternion + xyz position
    obstacles: list[dict] | None = None  # optional list of ObstacleEntry
    timesteps: int = 20
    dt: float = 0.02


class PlanResponse(BaseModel):
    waypoints: list[list[float]]
    dt: float


class IkRestRequest(BaseModel):
    """IK with a rest cost, for a robot from ``_URDF_ROBOTS`` picked per request."""

    robot: str
    target_link: str
    target_pose_wxyz_xyz: list[float]  # length 7 (wxyz + xyz)
    initial_q: list[float]  # actuated joints, in ``joint_names`` order
    rest_cost_weights: dict[str, float] = {}  # joint name -> weight; others get 1.0


class IkRestResponse(BaseModel):
    joint_positions: list[float]
    joint_names: list[str]


class FkRequest(BaseModel):
    robot: str
    joint_positions: list[float]


class FkResponse(BaseModel):
    link_poses_wxyz_xyz: list[list[float]]  # one per link, in ``link_names`` order
    link_names: list[str]


# =====================================================
# INTERNAL HELPERS
# =====================================================


def _build_world_coll(obstacles: list[dict[str, Any]] | None):
    if obstacles is None:
        return []
    world = []
    for obj in obstacles:
        t = obj.get("type")
        if t == "halfspace":
            p = np.array(obj["point"], dtype=np.float64)
            n = np.array(obj["normal"], dtype=np.float64)
            world.append(pk.collision.HalfSpace.from_point_and_normal(p, n))
        elif t == "sphere":
            c = np.array(obj["center"], dtype=np.float64)
            r = float(obj["radius"])
            world.append(pk.collision.Sphere.from_center_and_radius(c, np.array([r])))
        elif t == "capsule":
            pos = np.array(obj["position"], dtype=np.float64)
            rad = float(obj["radius"])
            h = float(obj["height"])
            world.append(
                pk.collision.Capsule.from_radius_height(
                    position=pos,
                    radius=np.array([rad]),
                    height=np.array([h]),
                )
            )
        elif t == "box":
            extent = np.array(obj["extent"], dtype=np.float64)
            pos = np.array(obj.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
            world.append(
                pk.collision.Box.from_extent(
                    extent=extent,
                    position=pos,
                )
            )
        else:
            logger.warning(f"Unknown obstacle type '{t}', ignoring")
    return world


# =====================================================
# SERVER INIT — Load Pyroki Only Once
# =====================================================
def set_min_distance_from_limits(urdf: yourdfpy.URDF, min_distance_from_limits: float = 0.15) -> yourdfpy.URDF:
    """
    Set the minimum distance from limits for the robot.
    min_distance_from_limits: float in radians
    """
    for joint in urdf.robot.joints:
        if joint.type == "revolute" and joint.limit is not None:
            if joint.limit.lower is not None and joint.limit.upper is not None:
                joint.limit.lower = joint.limit.lower + min_distance_from_limits
                joint.limit.upper = joint.limit.upper - min_distance_from_limits
    return urdf


def init_pyroki_server(
    robot_urdf_name: str = "panda_description", target_link_name: str = "panda_hand"
):
    global _ROBOT, _ROBOT_COLL, _TARGET_LINK

    logger.info(f"Loading robot URDF '{robot_urdf_name}' with Pyroki...")

    from robot_descriptions.loaders.yourdfpy import load_robot_description

    urdf = set_min_distance_from_limits(load_robot_description(robot_urdf_name))

    _ROBOT = pk.Robot.from_urdf(urdf)
    # _ROBOT_COLL = pk.collision.RobotCollision.from_urdf(urdf)
    sphere_decomposition = json.load(open(Path(__file__).parent / "assets" / "panda_spheres.json"))
    _ROBOT_COLL = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )
    _TARGET_LINK = target_link_name

    logger.info("PyRoki loaded and ready!")


# Robots served from a URDF in this repo, on top of the Panda loaded at startup.
# Clients pick one per request, so one server (one PYROKI_SERVICE_URL) covers
# every simulator. Only kinematics are used, so meshes are not loaded.
_URDF_ROBOTS: dict[str, Path] = {
    # BEHAVIOR R1Pro with the wheel and gripper joints fixed: 4 torso + 2x7 arm joints.
    "r1pro": Path(__file__).parent / "assets" / "r1pro_ik.urdf",
}
_URDF_ROBOTS_LOCK = threading.Lock()


@lru_cache(maxsize=None)
def _load_urdf_robot(name: str) -> pk.Robot:
    import yourdfpy

    logger.info(f"Loading robot '{name}' from {_URDF_ROBOTS[name]}...")
    urdf = yourdfpy.URDF.load(_URDF_ROBOTS[name], load_meshes=False, load_collision_meshes=False)
    return pk.Robot.from_urdf(urdf)


def _get_urdf_robot(name: str) -> pk.Robot:
    if name not in _URDF_ROBOTS:
        raise HTTPException(404, f"Unknown robot '{name}'. Known: {', '.join(_URDF_ROBOTS)}")
    # Requests run on executor threads; load each robot once.
    with _URDF_ROBOTS_LOCK:
        return _load_urdf_robot(name)


# =====================================================
# ROUTES
# =====================================================


def _do_solve_ik(target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None) -> list[float]:
    """Blocking IK solve (CPU-bound)."""
    if prev_cfg is None:
        q = pks.solve_ik(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
        )
    else:
        q = pks.solve_ik_vel_cost(
            robot=_ROBOT,
            target_link_name=_TARGET_LINK,
            target_position=target_pose_wxyz_xyz[-3:],
            target_wxyz=target_pose_wxyz_xyz[:-3],
            prev_cfg=prev_cfg,
        )
    return list(map(float, q))


@app.post("/ik", response_model=IkResponse)
async def solve_ik(req: IkRequest):
    if _ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")

    target_pose_wxyz_xyz = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev_cfg = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None

    try:
        joints = await _run_in_thread(_do_solve_ik, target_pose_wxyz_xyz, prev_cfg)
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")

    return IkResponse(joint_positions=joints)


def _do_plan_motion(req: PlanRequest) -> PlanResponse:
    """Blocking motion planning (CPU-bound)."""
    start_pose = np.array(req.start_pose_wxyz_xyz, dtype=np.float64)
    end_pose = np.array(req.end_pose_wxyz_xyz, dtype=np.float64)

    start_wxyz = start_pose[:4]
    start_position = start_pose[4:]
    end_wxyz = end_pose[:4]
    end_position = end_pose[4:]

    timesteps = req.timesteps
    dt = req.dt

    sol_traj = plan_trajectory_linear_ik(
        robot=_ROBOT,
        target_link_name=_TARGET_LINK,
        start_pos=start_position,
        start_wxyz=start_wxyz,
        end_pos=end_position,
        end_wxyz=end_wxyz,
        num_waypoints=timesteps,
    )
    sol_traj = np.asarray(sol_traj)

    return PlanResponse(
        waypoints=sol_traj.tolist(),
        dt=float(dt),
    )


@app.post("/plan", response_model=PlanResponse)
async def plan_motion(req: PlanRequest):
    if _ROBOT is None or _ROBOT_COLL is None:
        raise HTTPException(503, "Pyroki not initialized")

    try:
        return await _run_in_thread(_do_plan_motion, req)
    except Exception as e:
        logger.exception("Planning failure")
        raise HTTPException(500, f"Motion planning failed: {e}")


def _do_solve_ik_rest(robot: pk.Robot, req: IkRestRequest) -> IkRestResponse:
    """Blocking rest-cost IK solve (CPU-bound)."""
    names = list(robot.joints.actuated_names)
    unknown = set(req.rest_cost_weights) - set(names)
    if unknown:
        raise ValueError(f"Unknown joints in rest_cost_weights: {sorted(unknown)}")
    if len(req.initial_q) != len(names):
        raise ValueError(f"initial_q has {len(req.initial_q)} values, robot has {len(names)} joints")
    pose = np.asarray(req.target_pose_wxyz_xyz, dtype=np.float64)
    q = pks.solve_ik_rest(
        robot=robot,
        target_link_name=req.target_link,
        target_wxyz=pose[:4],
        target_position=pose[4:],
        rest_cost_weights=np.array([req.rest_cost_weights.get(n, 1.0) for n in names]),
        initial_q=np.asarray(req.initial_q, dtype=np.float64),
    )
    return IkRestResponse(joint_positions=list(map(float, q)), joint_names=names)


@app.post("/ik_rest", response_model=IkRestResponse)
async def solve_ik_rest(req: IkRestRequest):
    robot = _get_urdf_robot(req.robot)
    if req.target_link not in robot.links.names:
        raise HTTPException(400, f"Unknown link '{req.target_link}' for robot '{req.robot}'")
    try:
        return await _run_in_thread(_do_solve_ik_rest, robot, req)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("IK failed")
        raise HTTPException(500, f"IK solve failed: {e}")


@app.post("/fk", response_model=FkResponse)
async def forward_kinematics(req: FkRequest):
    robot = _get_urdf_robot(req.robot)
    if len(req.joint_positions) != robot.joints.num_actuated_joints:
        raise HTTPException(
            400,
            f"joint_positions has {len(req.joint_positions)} values, "
            f"robot has {robot.joints.num_actuated_joints} joints",
        )
    poses = await _run_in_thread(
        lambda: np.asarray(robot.forward_kinematics(np.asarray(req.joint_positions, dtype=np.float64)))
    )
    return FkResponse(link_poses_wxyz_xyz=poses.tolist(), link_names=list(robot.links.names))


# =====================================================
# ENTRYPOINT
# =====================================================


def main(
    robot: str = "panda_description",
    target_link: str = "panda_hand",
    port: int = 8116,
    host: str = "127.0.0.1",
):
    init_pyroki_server(robot_urdf_name=robot, target_link_name=target_link)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)