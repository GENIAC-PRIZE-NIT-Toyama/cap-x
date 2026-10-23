"""Snapshot のテストで使う、知覚・IK のサーバに頼らない API と env の組み立て。

本番の API（FrankaControlApi など）は IK や知覚を別サーバに頼むので、手元のテストでは
動かせない。ここでは関節角とグリッパを直接動かす最小の API を登録する。IK の warm start に
相当する「呼ぶたびに変わる API 側の状態」（`cfg`）も持たせ、それが戻ることも確かめる。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.integrations.base_api import ApiBase, register_api

API_NAME = "SnapshotTestJointApi"


class SnapshotTestJointApi(ApiBase):
    def __init__(self, env: Any) -> None:
        super().__init__(env)
        self.cfg: np.ndarray | None = None  # 呼ぶたびに更新される API 側の状態
        self.calls = 0

    def functions(self) -> dict[str, Any]:
        return {
            "get_joints": self.get_joints,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    def get_joints(self) -> np.ndarray:
        """今の関節角（7,）。2 腕の env では 1 本目の腕。"""
        obs = self._env.get_observation()
        joints = obs["robot_joint_pos"] if "robot_joint_pos" in obs else obs["robot0_joint_pos"]
        return np.asarray(joints[:7], dtype=np.float64)

    def move_to_joints(self, joints: np.ndarray) -> None:
        """関節角へ動かす。"""
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self.cfg = joints if self.cfg is None else 0.5 * (self.cfg + joints)
        self.calls += 1
        self._env.move_to_joints_blocking(joints)

    def open_gripper(self) -> None:
        """グリッパを開く。"""
        self._env._set_gripper(1.0)
        for _ in range(20):
            self._env._step_once()

    def close_gripper(self) -> None:
        """グリッパを閉じる。"""
        self._env._set_gripper(0.0)
        for _ in range(20):
            self._env._step_once()


register_api(API_NAME, SnapshotTestJointApi)


def make_env(low_level: Any, *, task_id: str = "snapshot_test", record_video: bool = False):
    """worker と同じ形（生成コードは別プロセス）の `LocalAgentEnv`。"""
    from capx.agent_api import Budget
    from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
    from capx.local_env import LocalAgentEnv
    from capx.remote_env.worker.executor import ProcessExecutor

    code_env = CodeExecutionEnvBase(
        CodeExecEnvConfig(low_level=low_level, apis=[API_NAME], prompt="move", privileged=True)
    )
    code_env.set_process_executor(ProcessExecutor(step_timeout_s=300))
    return LocalAgentEnv(
        code_env, budget=Budget(max_steps=10), task_id=task_id, record_video=record_video
    )


def sim_of(env) -> Any:
    from capx.snapshot.gym_state import simulator_env

    return simulator_env(env.inner.low_level_env).sim


def physics(env) -> dict[str, np.ndarray]:
    data = sim_of(env).data._data
    return {
        "time": np.array(data.time),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy(),
        "ctrl": data.ctrl.copy(),
    }
