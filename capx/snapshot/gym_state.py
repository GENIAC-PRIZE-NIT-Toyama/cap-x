"""Gym（シミュレータ・コントローラ・API 層）の状態を、データだけの dict にして戻す。

**pickle は使わない。** 値は数値・文字列・numpy 配列とその入れ物だけにし、msgpack で運ぶ。
Gym のプロセスは採点を持つので、ここで任意のコードが動く形式は解かない。

MuJoCo の物理状態だけでは足りない。robosuite / LIBERO は、次のステップの動きを決める
状態を Python 側にも持っている。

- MuJoCo: 積分に使う状態（time / qpos / qvel / act / qacc_warmstart / ctrl / 外力 /
  mocap / userdata）と、reset のたびに書き換わるモデルの値（固定物の位置 body_pos /
  body_quat、拭いた跡の geom_rgba など）。LIBERO と robosuite は hard reset で、reset の
  たびにモデルを作り直して固定物を置き直す
- コントローラ: 目標関節角（goal_qpos）、補間器、グリッパの指令値（PandaGripper の
  current_action は前の値に足していく）
- 環境: timestep / cur_time / done、観測の cache（`_obs_cache`）、Observable のサンプリング
  の時計と最後に観測した値、乱数
- capx のラッパ: 目標関節角（_current_joints）・グリッパ（_gripper_fraction）・
  シミュレーションのステップ数・LIBERO の最後の観測 / 報酬 / done
- API 層: IK の warm start（`cfg`。前の解を次の初期値にする）

これらを、決まった根（ラッパ・シミュレータ・API）から辿れる属性のうち **データだけのもの**
として保存する。辿る先は robosuite のロボット・コントローラ・グリッパ・バッファに限る
（モデルの XML やレンダラは辿らない）。カメラの画像は保存せず、復元した物理状態から
描き直す（物理を進めずに `_update_observables(force=True)`）。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from typing import Any

import numpy as np

#: MuJoCo の mjData のうち、積分を続けるのに要るもの（mjSTATE_INTEGRATION 相当）。
MJ_DATA_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "plugin_state",
    "eq_active",
)
#: mjModel のうち、reset で書き換わりうるもの。
MJ_MODEL_FIELDS = ("body_pos", "body_quat", "geom_rgba", "site_rgba", "eq_active0")

#: ラッパ（capx の low-level env）で保存しないもの。録画は試行ごとのもので、復元先では
#: その時点から録り直す。`_current_obs` は LIBERO の最後の観測（画像を含む）で、復元後に
#: 作り直す。`viser_server` 系はデバッグ表示。
WRAPPER_SKIP = frozenset(
    {
        "_frame_buffer",
        "_wrist_frame_buffer",
        "_buffer_bytes",
        "_frame_seqs",
        "_frame_seq_next",
        "_pinned_seqs",
        "_frame_stats",
        "_frame_listener",
        "_frame_budget_bytes",
        "_record_frames",
        "_record_wrist_camera",
        "_subsample_rate",
        "_current_obs",
    }
)
#: シミュレータ（robosuite の env）で辿らない・保存しないもの。`_obs_cache` は画像を含むので
#: Observable と一緒に別に扱う。
ENV_SKIP = frozenset({"_obs_cache", "_observables", "sim", "model", "viewer", "renderer"})

#: 中まで辿るクラスの module。ロボット・コントローラ・補間器・バッファ・グリッパ。
RECURSE_MODULES = (
    "robosuite.robots",
    "robosuite.controllers",
    "robosuite.utils.buffers",
    "robosuite.utils.traj_utils",
    "robosuite.models.grippers",
)

#: これより大きな配列は保存しない（画像や点群が紛れ込んだとき）。保存しなかった属性は
#: manifest に残る（`skipped_large`）。
MAX_ARRAY_BYTES = 4 * 1024 * 1024

_TUPLE = "__tuple__"
_DEQUE = "__deque__"
#: msgpack に載らない大きな整数（PCG64 の乱数の状態は 128 bit）。10 進の文字列で持つ。
_BIGINT = "__bigint__"
#: 別の属性と同じ配列を指す属性の印。値は「path + _SEP + 属性名」。
_ALIAS = "__alias__"
#: object 型の配列（要素が None や数値だけのもの。robosuite のバッファが `array(None)` を持つ）。
_OBJARRAY = "__objarray__"
_SEP = "\x00"


class GymSnapshotError(RuntimeError):
    """Gym の状態を保存・復元できない（対応していない env・互換性がない Snapshot など）。"""


# -- 入口 --------------------------------------------------------------------


def capture(code_env: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """`CodeExecutionEnvBase` の状態を取る。物理は進めない。

    Returns:
        (state, info)。state は msgpack に載る dict。info は manifest に残す要約
        （モデルの指紋・保存しなかった大きな属性）。
    """
    low = code_env.low_level_env
    sim_env = simulator_env(low)
    skipped: list[str] = []
    roots = _roots(code_env, low, sim_env)

    objects: dict[str, dict[str, Any]] = {}
    rngs: dict[str, Any] = {}
    # 同じ配列を指す属性（コントローラの joint_pos と initial_joint など）。復元先でも同じ配列を
    # 指すように、2 つ目からは「どの属性と同じか」だけを残す。共有の仕方は Step を進めると
    # 変わる（reset 直後は共有し、目標を更新すると別れる）ので、保存時の形を再現する。
    first_seen: dict[int, str] = {}
    for path, obj in _walk(roots).items():
        attrs = {}
        for name, value in vars(obj).items():
            if name in _skip_for(path):
                continue
            if _is_generator(value):
                rngs[f"{path}.{name}"] = _rng_state(value)
                continue
            if isinstance(value, np.ndarray) and id(value) in first_seen:
                attrs[name] = {_ALIAS: first_seen[id(value)]}
                continue
            original = value
            value = _as_numpy(value)
            if _is_plain(value):
                if _too_large(value):
                    skipped.append(f"{path}.{name}")
                    continue
                attrs[name] = _encode(value)
                if isinstance(original, np.ndarray):
                    first_seen[id(original)] = f"{path}{_SEP}{name}"
        objects[path] = attrs

    state = {
        "mujoco": _mujoco_state(sim_env.sim),
        "objects": objects,
        "rngs": rngs,
        "observables": _observable_state(sim_env),
        "extra": _extra_state(sim_env),
        "np_random": _encode(_np_global_state()),
        "py_random": _encode(random.getstate()),
        "code_env": {"_step_count": int(getattr(code_env, "_step_count", 0))},
    }
    info = {
        "model_signature": model_signature(sim_env.sim),
        "model_params": model_params_digest(sim_env.sim),
        "env_class": f"{type(low).__module__}.{type(low).__qualname__}",
        "skipped_large": skipped,
    }
    return state, info


def capture_rngs(code_env: Any) -> dict[str, Any]:
    """乱数の状態だけを取る。reset の直前に取っておき、復元で同じ reset を再現するのに使う。

    robosuite / LIBERO は reset のたびにモデル（XML）を作り直し、物体の大きさや固定物の位置を
    乱数で決める（cube_lift のキューブの大きさなど）。物理状態を戻しても、モデルが違えば
    続きの動きが変わる。reset の前の乱数を同じにすれば、同じモデルができる。
    """
    low = code_env.low_level_env
    sim_env = simulator_env(low)
    rngs = {}
    for path, obj in _walk(_roots(code_env, low, sim_env)).items():
        for name, value in vars(obj).items():
            if _is_generator(value) and name not in _skip_for(path):
                rngs[f"{path}.{name}"] = _rng_state(value)
    return {
        "rngs": rngs,
        "np_random": _encode(_np_global_state()),
        "py_random": _encode(random.getstate()),
    }


def restore_rngs(code_env: Any, state: dict[str, Any]) -> None:
    """`capture_rngs` の状態に戻す（reset の直前に呼ぶ）。"""
    low = code_env.low_level_env
    objects = _walk(_roots(code_env, low, simulator_env(low)))
    for key, rng_state in state.get("rngs", {}).items():
        path, _, name = key.rpartition(".")
        obj = objects.get(path)
        if obj is not None and _is_generator(getattr(obj, name, None)):
            _set_rng_state(getattr(obj, name), _decode(rng_state))
    _set_np_global_state(_decode(state["np_random"]))
    random.setstate(_to_tuples(_decode(state["py_random"])))


def restore(code_env: Any, state: dict[str, Any], info: dict[str, Any]) -> None:
    """`capture` の状態を、同じタスクで作った env に戻す。物理は進めない。

    `code_env` は reset 済みであること（hard reset でモデルが作られ、コントローラが
    初期化された状態）。そこに状態を上書きする。
    """
    low = code_env.low_level_env
    sim_env = simulator_env(low)

    env_class = f"{type(low).__module__}.{type(low).__qualname__}"
    if info.get("env_class") and info["env_class"] != env_class:
        raise GymSnapshotError(f"Snapshot の env（{info['env_class']}）と、この worker の env（{env_class}）が違う")
    signature = model_signature(sim_env.sim)
    if info.get("model_signature") != signature:
        raise GymSnapshotError(
            "Snapshot のシミュレータのモデルが、この worker のものと違う（タスクか worker の版が違う）。"
            f" snapshot={info.get('model_signature')} worker={signature}"
        )

    _restore_mujoco(sim_env.sim, state["mujoco"])
    params = model_params_digest(sim_env.sim)
    if info.get("model_params") and info["model_params"] != params:
        raise GymSnapshotError(
            "シミュレータのモデルのパラメータ（物体の大きさ・質量・摩擦など）が Snapshot と違う。"
            "reset で乱数から作られたモデルを再現できなかった"
        )
    _restore_extra(sim_env, state.get("extra") or {})
    # 物理を進めずに、Observable（カメラ画像を含む）を今の状態から計算し直す
    if hasattr(sim_env, "_update_observables"):
        sim_env._update_observables(force=True)

    objects = _walk(_roots(code_env, low, sim_env))
    arrays: dict[str, np.ndarray] = {}
    for path, attrs in state["objects"].items():
        obj = objects.get(path)
        if obj is None:
            raise GymSnapshotError(f"Snapshot にある {path} が、この worker の env に無い")
        for name, value in attrs.items():
            if isinstance(value, dict) and _ALIAS in value:
                setattr(obj, name, arrays[value[_ALIAS]])
                continue
            _assign(obj, name, _decode(value))
            current = getattr(obj, name)
            if isinstance(current, np.ndarray):
                arrays[f"{path}{_SEP}{name}"] = current
    for key, rng_state in state.get("rngs", {}).items():
        path, _, name = key.rpartition(".")
        obj = objects.get(path)
        if obj is not None and _is_generator(getattr(obj, name, None)):
            _set_rng_state(getattr(obj, name), _decode(rng_state))

    _restore_observables(sim_env, state.get("observables") or {})
    _set_np_global_state(_decode(state["np_random"]))
    random.setstate(_to_tuples(_decode(state["py_random"])))
    code_env._step_count = int(state.get("code_env", {}).get("_step_count", 0))

    # LIBERO のラッパは、最後に受け取った観測（画像を含む）から観測を作る。物理を進めずに作り直す。
    if hasattr(low, "_current_obs") and hasattr(sim_env, "_get_observations"):
        low._current_obs = sim_env._get_observations()


def simulator_env(low: Any) -> Any:
    """capx のラッパから、robosuite の env（`sim` を持つもの）を取り出す。"""
    if hasattr(low, "robosuite_env"):
        return low.robosuite_env
    handle = getattr(low, "handle", None)
    if handle is not None and hasattr(handle, "env"):
        # LIBERO: OffScreenRenderEnv（ControlEnv）の中の robosuite の env
        inner = getattr(handle.env, "env", handle.env)
        if hasattr(inner, "sim"):
            return inner
    raise GymSnapshotError(
        f"{type(low).__name__} は Snapshot に対応していない（robosuite と LIBERO だけ）"
    )


def model_signature(sim: Any) -> str:
    """モデルの構造の指紋。互換性のない Snapshot を読み込もうとしたときに止める。

    名前と次元だけを見る（固定物の位置など、reset で変わる値は見ない）。
    """
    import mujoco

    model = sim.model._model
    parts: dict[str, Any] = {
        key: int(getattr(model, key))
        for key in ("nq", "nv", "na", "nu", "nbody", "njnt", "ngeom", "nsite", "nmocap", "neq", "nuserdata")
    }
    for kind, count in (
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    ):
        parts[str(int(kind))] = [mujoco.mj_id2name(model, kind, i) for i in range(count)]
    parts["jnt_type"] = [int(x) for x in model.jnt_type]
    raw = json.dumps(parts, sort_keys=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


#: モデルのパラメータの指紋に入れるもの。reset の乱数で変わりうる値（大きさ・質量・摩擦・
#: 位置）と、動きに効く値。テクスチャやメッシュの頂点のような大きく変わらないものは入れない。
MODEL_PARAM_FIELDS = (
    "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref",
    "geom_solimp", "geom_margin", "geom_gap", "geom_contype", "geom_conaffinity",
    "jnt_range", "jnt_stiffness", "dof_damping", "dof_armature", "dof_frictionloss",
    "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "eq_data",
    "site_pos", "site_size", "cam_pos", "cam_quat", "cam_fovy",
)


def model_params_digest(sim: Any) -> str:
    """モデルのパラメータの指紋。復元した worker のモデルが、保存したときと同じかを確かめる。"""
    model = sim.model._model
    digest = hashlib.sha256()
    for name in MODEL_PARAM_FIELDS:
        value = getattr(model, name, None)
        if isinstance(value, np.ndarray):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(np.asarray([model.opt.timestep, *model.opt.gravity]).tobytes())
    return "sha256:" + digest.hexdigest()


# -- 辿る --------------------------------------------------------------------


def _roots(code_env: Any, low: Any, sim_env: Any) -> list[tuple[str, Any]]:
    roots = [("low", low), ("sim_env", sim_env)]
    for name, api in sorted(getattr(code_env, "_apis", {}).items()):
        roots.append((f"api[{name}]", api))
    return roots


def _skip_for(path: str) -> frozenset[str]:
    if path == "low":
        return WRAPPER_SKIP
    if path == "sim_env":
        return ENV_SKIP
    return frozenset()


def _walk(roots: list[tuple[str, Any]]) -> dict[str, Any]:
    """根から辿れる、状態を保存するオブジェクト。`{path: obj}`。

    根はそのもの（属性は 1 段だけ）。そこから辿るのは `RECURSE_MODULES` のクラスだけ。
    同じオブジェクトに 2 つの道で着いたら、最初の道だけを使う（決まった順に辿る）。
    """
    found: dict[str, Any] = {}
    seen: set[int] = set()

    def visit(obj: Any, path: str) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        found[path] = obj
        for name, value in sorted(vars(obj).items()):
            if name in _skip_for(path):
                continue
            for child, child_path in _children(value, f"{path}.{name}"):
                visit(child, child_path)

    for path, obj in roots:
        visit(obj, path)
    return found


def _children(value: Any, path: str) -> list[tuple[Any, str]]:
    if _recurse_into(value):
        return [(value, path)]
    if isinstance(value, (list, tuple)):
        return [(v, f"{path}[{i}]") for i, v in enumerate(value) if _recurse_into(v)]
    if isinstance(value, dict):
        return [(v, f"{path}[{k}]") for k, v in value.items() if isinstance(k, str) and _recurse_into(v)]
    return []


def _recurse_into(value: Any) -> bool:
    module = type(value).__module__ or ""
    return hasattr(value, "__dict__") and module.startswith(RECURSE_MODULES)


# -- 値 ----------------------------------------------------------------------


def _is_plain(value: Any, depth: int = 0) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, bytes, np.generic)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return value.size <= 1024 and all(
                v is None or isinstance(v, (bool, int, float, str)) for v in value.flat
            )
        return True
    if depth >= 4:
        return False
    if isinstance(value, (list, tuple, deque)):
        return all(_is_plain(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int)) and _is_plain(v, depth + 1) for k, v in value.items())
    return False


def _as_numpy(value: Any) -> Any:
    """jax の配列（IK の warm start）を numpy にする。それ以外はそのまま。"""
    module = type(value).__module__ or ""
    if module.startswith(("jax", "jaxlib")) and hasattr(value, "__array__"):
        return np.asarray(value)
    return value


def _too_large(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.nbytes > MAX_ARRAY_BYTES
    if isinstance(value, (list, tuple, deque)):
        return any(_too_large(v) for v in value)
    if isinstance(value, dict):
        return any(_too_large(v) for v in value.values())
    return False


def _encode(value: Any) -> Any:
    """tuple と deque に印を付ける（msgpack ではどちらも list になる）。"""
    if isinstance(value, int) and not isinstance(value, bool) and not -(2**63) <= value < 2**64:
        return {_BIGINT: str(value)}
    if isinstance(value, tuple):
        return {_TUPLE: [_encode(v) for v in value]}
    if isinstance(value, deque):
        return {_DEQUE: [_encode(v) for v in value], "maxlen": value.maxlen}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.dtype == object:
        return {_OBJARRAY: [_encode(v) for v in value.flat], "shape": list(value.shape)}
    if isinstance(value, np.ndarray):
        # 生きた配列を参照したままにしない（取ったあとに env が中身を書き換える）
        return np.array(value, copy=True, order="C")
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if _TUPLE in value and len(value) == 1:
            return tuple(_decode(v) for v in value[_TUPLE])
        if _DEQUE in value:
            return deque((_decode(v) for v in value[_DEQUE]), maxlen=value.get("maxlen"))
        if _BIGINT in value and len(value) == 1:
            return int(value[_BIGINT])
        if _OBJARRAY in value:
            out = np.empty(len(value[_OBJARRAY]), dtype=object)
            out[:] = [_decode(v) for v in value[_OBJARRAY]]
            return out.reshape(value["shape"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        return value.copy()
    return value


def _to_tuples(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_to_tuples(v) for v in value)
    return value


_MISSING = object()


def _assign(obj: Any, name: str, value: Any) -> None:
    """値を戻す。

    配列は新しい配列に差し替える（復元先で別の属性と共有していると、中身を書き換えたときに
    相手まで変わる。共有は `_ALIAS` で保存時の形に作り直す）。ただし MuJoCo のバッファを
    指す view は、差し替えるとつながりが切れるので中身を書き戻す。list・dict は中身を書き換える。
    """
    current = getattr(obj, name, _MISSING)
    if (
        isinstance(current, np.ndarray)
        and isinstance(value, np.ndarray)
        and current.shape == value.shape
        and current.dtype == value.dtype
        and current.flags.writeable
        and _is_mujoco_buffer(current)
    ):
        current[...] = value
    elif isinstance(current, deque) and isinstance(value, deque) and current.maxlen == value.maxlen:
        current.clear()
        current.extend(value)
    elif type(current) is list and isinstance(value, list):
        current[:] = value
    elif type(current) is dict and isinstance(value, dict):
        current.clear()
        current.update(value)
    else:
        setattr(obj, name, value)


def _is_mujoco_buffer(array: np.ndarray) -> bool:
    """MuJoCo の mjData / mjModel のメモリを指す view か。"""
    base: Any = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return (type(base).__module__ or "").startswith("mujoco") and base is not array


def _is_generator(value: Any) -> bool:
    return isinstance(value, (np.random.Generator, np.random.RandomState))


def _rng_state(rng: Any) -> Any:
    if isinstance(rng, np.random.Generator):
        return _encode({"kind": "generator", "state": rng.bit_generator.state})
    return _encode({"kind": "random_state", "state": rng.get_state(legacy=False)})


def _set_rng_state(rng: Any, state: dict[str, Any]) -> None:
    if state["kind"] == "generator":
        rng.bit_generator.state = state["state"]
    else:
        rng.set_state(state["state"])


def _np_global_state() -> Any:
    return np.random.get_state(legacy=False)


def _set_np_global_state(state: Any) -> None:
    np.random.set_state(state)


# -- MuJoCo ------------------------------------------------------------------


def _mujoco_state(sim: Any) -> dict[str, Any]:
    data, model = sim.data._data, sim.model._model
    out: dict[str, Any] = {"data": {}, "model": {}}
    for name in MJ_DATA_FIELDS:
        if hasattr(data, name):
            value = getattr(data, name)
            out["data"][name] = np.array(value, copy=True) if isinstance(value, np.ndarray) else float(value)
    for name in MJ_MODEL_FIELDS:
        if hasattr(model, name):
            out["model"][name] = np.array(getattr(model, name), copy=True)
    return out


def _restore_mujoco(sim: Any, state: dict[str, Any]) -> None:
    import mujoco

    data, model = sim.data._data, sim.model._model
    for name, value in state.get("model", {}).items():
        target = getattr(model, name, None)
        if target is None or target.shape != np.shape(value):
            raise GymSnapshotError(f"mjModel.{name} の形が違う（Snapshot と worker のモデルが違う）")
        target[...] = value
    fields = state.get("data", {})
    for name, value in fields.items():
        if name == "time":
            data.time = float(value)
            continue
        target = getattr(data, name, None)
        if target is None or target.shape != np.shape(value):
            raise GymSnapshotError(f"mjData.{name} の形が違う（Snapshot と worker のモデルが違う）")
        target[...] = value
    # 位置から導かれる量（xpos・接触・センサ）を計算し直す。物理は進めない。
    mujoco.mj_forward(model, data)
    # mj_forward は warm start を変えないが、念のため保存した値に戻す（次の step の解が一致する）
    if "qacc_warmstart" in fields:
        data.qacc_warmstart[...] = fields["qacc_warmstart"]


# -- Observable --------------------------------------------------------------

_OBSERVABLE_FIELDS = ("_time_since_last_sample", "_current_delay", "_sampled", "_active", "_enabled")


def _is_camera_name(sim_env: Any, name: str) -> bool:
    for camera in getattr(sim_env, "camera_names", None) or ():
        if name.startswith(f"{camera}_") and name[len(camera) + 1 :].split("_")[0] in (
            "image",
            "depth",
            "segmentation",
        ):
            return True
    return False


def _observable_state(sim_env: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"values": {}, "cache": {}}
    for name, observable in getattr(sim_env, "_observables", {}).items():
        fields = {key: _encode(getattr(observable, key)) for key in _OBSERVABLE_FIELDS if hasattr(observable, key)}
        if not _is_camera_name(sim_env, name):
            value = getattr(observable, "_current_observed_value", None)
            if _is_plain(value):
                fields["_current_observed_value"] = _encode(value)
        out["values"][name] = fields
    for key, value in (getattr(sim_env, "_obs_cache", None) or {}).items():
        if not _is_camera_name(sim_env, key) and _is_plain(value) and not _too_large(value):
            out["cache"][key] = _encode(value)
    return out


def _restore_observables(sim_env: Any, state: dict[str, Any]) -> None:
    observables = getattr(sim_env, "_observables", {})
    for name, fields in state.get("values", {}).items():
        observable = observables.get(name)
        if observable is None:
            raise GymSnapshotError(f"Snapshot にある観測 {name} が、この worker の env に無い")
        for key, value in fields.items():
            setattr(observable, key, _decode(value))
    cache = getattr(sim_env, "_obs_cache", None)
    if isinstance(cache, dict):
        # カメラの画像は `_update_observables(force=True)` が入れた今のものを残す
        cache.update({k: _decode(v) for k, v in state.get("cache", {}).items()})


# -- タスク固有 ----------------------------------------------------------------


def _extra_state(sim_env: Any) -> dict[str, Any]:
    """データだけでは表せない、タスク固有の状態。"""
    out: dict[str, Any] = {}
    # robosuite の Wipe（spill_wipe）: 拭いた印のオブジェクトの list。名前で持つ。
    if isinstance(getattr(sim_env, "wiped_markers", None), list):
        out["wiped_markers"] = [marker.name for marker in sim_env.wiped_markers]
    return out


def _restore_extra(sim_env: Any, state: dict[str, Any]) -> None:
    if "wiped_markers" in state:
        markers = {m.name: m for m in sim_env.model.mujoco_arena.markers}
        missing = [name for name in state["wiped_markers"] if name not in markers]
        if missing:
            raise GymSnapshotError(f"拭いた印 {missing} が、この worker の env に無い")
        sim_env.wiped_markers[:] = [markers[name] for name in state["wiped_markers"]]
