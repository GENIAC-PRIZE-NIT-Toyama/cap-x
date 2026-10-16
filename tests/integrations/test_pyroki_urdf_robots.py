"""The PyRoKi server's per-request URDF robots (/ik_rest, /fk), used by BEHAVIOR R1Pro."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyroki")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from capx.serving import launch_pyroki_server as server  # noqa: E402

TORSO_WEIGHTS = {f"torso_joint{i}": 20.0 for i in range(1, 5)}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server.app)


def _fk(client: TestClient, q: np.ndarray) -> tuple[np.ndarray, list[str]]:
    r = client.post("/fk", json={"robot": "r1pro", "joint_positions": q.tolist()})
    assert r.status_code == 200, r.text
    return np.asarray(r.json()["link_poses_wxyz_xyz"]), r.json()["link_names"]


def test_r1pro_has_torso_and_two_arms(client: TestClient) -> None:
    r = client.post(
        "/ik_rest",
        json={
            "robot": "r1pro",
            "target_link": "left_gripper_link",
            "target_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.5, 0.2, 1.0],
            "initial_q": [0.0] * 18,
        },
    )
    assert r.status_code == 200, r.text
    names = r.json()["joint_names"]
    assert names[:4] == ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"]
    assert names[4:11] == [f"left_arm_joint{i}" for i in range(1, 8)]
    assert names[11:] == [f"right_arm_joint{i}" for i in range(1, 8)]


@pytest.mark.parametrize("link", ["left_gripper_link", "right_gripper_link"])
def test_ik_rest_reaches_a_reachable_pose(client: TestClient, link: str) -> None:
    q_goal = np.zeros(18)
    q_goal[4:] = 0.2  # bend both arms a little; torso stays at rest
    poses, link_names = _fk(client, q_goal)
    target = poses[link_names.index(link)]

    r = client.post(
        "/ik_rest",
        json={
            "robot": "r1pro",
            "target_link": link,
            "target_pose_wxyz_xyz": target.tolist(),
            "initial_q": [0.0] * 18,
            "rest_cost_weights": TORSO_WEIGHTS,
        },
    )
    assert r.status_code == 200, r.text
    reached, _ = _fk(client, np.asarray(r.json()["joint_positions"]))
    # The rest cost trades a little accuracy for staying near the rest pose.
    assert np.linalg.norm(reached[link_names.index(link)][4:] - target[4:]) < 0.02


@pytest.mark.parametrize(
    ("path", "payload", "status"),
    [
        ("/fk", {"robot": "nope", "joint_positions": [0.0] * 18}, 404),
        ("/fk", {"robot": "r1pro", "joint_positions": [0.0] * 3}, 400),
        (
            "/ik_rest",
            {"robot": "r1pro", "target_link": "nope", "target_pose_wxyz_xyz": [1, 0, 0, 0, 0, 0, 0], "initial_q": [0.0] * 18},
            400,
        ),
        (
            "/ik_rest",
            {
                "robot": "r1pro",
                "target_link": "left_gripper_link",
                "target_pose_wxyz_xyz": [1, 0, 0, 0, 0, 0, 0],
                "initial_q": [0.0] * 18,
                "rest_cost_weights": {"nope": 1.0},
            },
            400,
        ),
    ],
)
def test_bad_requests_are_rejected(client: TestClient, path: str, payload: dict, status: int) -> None:
    assert client.post(path, json=payload).status_code == status
