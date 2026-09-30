# Remote Development / Agent 分離

> 設計資料「CaP-X Agent 分離設計」で固まった方針に、Docker 隔離・uv extras・
> `local` / `remote` ワークスペース分割・`web-ui/` 削除を織り込んだもの。
> 議論の経過と未反映の検討は artifact 側にあり、このファイルは**確定した決定のみ**を持つ。

## 1. 目的

メンバーが各自の Agent を試作できるよう、Gym・Bench・Agent の依存を一方向に剥がす。
性能比較は保つため、共通の最小インターフェースを置く。
シミュレーションは GPU マシン、Agent は手元 PC という開発形態も同じ口で支える。

- **試作しやすい** — `trial.py` を書き換えずに自分の Agent を差し込める
- **依存を一方向に** — Gym と Bench は Agent（LLM・プロンプト戦略）を知らない
- **比較できる** — Agent が違っても、同じ seed・同じ予算・同じ指標で評価する
- **遠隔で回せる** — 強いマシンでシミュレーション、手元で Agent

非ゴール: `CodeExecutionEnvBase` の中身の作り替え／既存 YAML 約 200 本の一括書き換え（互換レイヤで吸収）。

## 2. 三層と AgentEnv

```
Bench   seed・予算・成果物・指標・evaluate()
  │
  ├─ Gym    CodeExecutionEnvBase / Base env / API 層（変更しない）
  └─ Agent  LLM・ループ・プロンプト・VDM（AgentEnv の型だけ知る）
```

- Gym は Bench も Agent も import しない
- Bench は Gym を使い、Agent は Agent プロトコルだけ知る
- **Agent は `AgentEnv` 越しにしか Gym に触れない**。`CodeExecutionEnvBase` を直接呼ばない

```python
# capx/agent_api/types.py
class AgentEnv(Protocol):
    def step(self, code: str, *, capture_video: bool = False) -> StepResult: ...
    def render(self, camera: str = "main") -> bytes: ...   # JPEG
    def close(self) -> None: ...
```

`StepResult` に **含めない**もの: `reward` / `task_completed` / Gym の生 `info` /
low-level observation / 内部 config / セッション管理用 credential。

`AgentEnv` の実体は 2 つ。`LocalAgentEnv` は同一プロセスの `CodeExecutionEnvBase` を包む。
`RemoteAgentEnv` は ZMQ 越しに GPU マシンの env を呼ぶ。Agent からは区別できない。
`LocalAgentEnv` は worker コンテナの中身としても再利用するので、実装は 2 つのままで重複しない。

Bench から見た分岐は `make_agent_env()` の 1 回だけで、`trial.py` の中には入らない。

## 3. ディレクトリ

```
capx/                      # フレームワーク
  agent_api/               # TaskSpec / StepResult / AgentEnv / Budget / Agent   [軽]
    components/            #   vdm.py, code_extract.py
  bench/                   # 成果物保存・指標・集計・リトライ・並列               [軽]
  local_env.py             # LocalAgentEnv（in-process）                        [重]
  remote_env/
    client.py              #   RemoteAgentEnv                                   [軽]
    server/                #   ホスト側。セッション台帳と docker 起動            [軽]
    worker/                #   コンテナ内。env を 1 つ持つ                       [重]
  envs/                    # Gym（変更しない）                                   [重]
  baselines/               # 同梱 Agent: capagent0.py, oracle.py                [軽]
env_configs/               # ベンチマーク（既存）
local/
  trial.py                 # フル引数。capx.bench を呼ぶ
  agents/
remote/
  trial.py                 # 引数を絞る。capx.bench を呼ぶ
  agents/                  # 参加者が書く。1 ファイル 1 エージェント
  README.md
  .env.example
```

`local/` と `remote/` はワークスペース、`capx/` はライブラリ。初心者は `remote/` だけを開く。

実装を `local/` に移さず `capx/` に残すのは、`pyproject.toml` の
`[tool.setuptools.packages.find] include = ["capx*"]` により `local/` がパッケージに
含まれないため。`tests/` と `verl_agent_reward/` は `capx.*` を import している。

同梱 Agent を `capx/baselines/` に置くのは、ユーザーの `agents/` と 1 文字違いで
並ぶのを避けるため。

## 4. ネットワークと通信

```
手元PC                                    GPUマシン
┌───────────────┐                        ┌──────────────────────────┐
│ Bench         │──HTTP :8200─────────▶  │ backend（セッション台帳）  │
│ Agent         │                        │   docker run              │
│ RemoteAgentEnv│══ZMQ :19500+═════════▶ │  worker コンテナ           │
└───────┬───────┘                        │   LocalAgentEnv            │
        │ HTTP                           │        │ (--internal)      │
        ▼                                │        ▼                   │
  vLLM 192.168.0.30:8000                 │  perception-proxy → .200   │
                                         └──────────────────────────┘
```

コントロール面とデータ面を分ける。backend はデータ経路に入らない
（介在させると構造が複雑になり、HTTP の帯域を埋める）。

Cloudflare Tunnel は使わず全部ポート解放（平文 HTTP）。外部からは自前 VPN。
インターネットへの直接公開ではない。

### コントロール面 — HTTP（開放ポート、例 8200）

| メソッド | パス | 内容 |
| --- | --- | --- |
| POST | `/sessions` | `{task_id, client_public_key}` → `{session_id, zmq_endpoint, server_public_key, task_id}` |
| DELETE | `/sessions/{id}` | 終了。自分のセッションだけ（他人のものは 404） |
| GET | `/sessions` | 自分のセッション一覧 |
| GET | `/tasks` | 選べるタスク（allowlist） |
| GET | `/health` | 認証なし |

**採点は `/finalize` ではなく、ZMQ の `evaluate` で worker から直接取る。**
当初は backend 経由（`/finalize`）にする設計だったが、守りたかったのは
「クライアントが値を送って書き換えられない」ことで、それは `evaluate` が
worker の計算結果を**読む**だけの操作であれば満たされる。backend を経由させると
データ面に backend が入り、構造が複雑になる。

クライアントから任意の `env_config` を受けない。Hydra 風の `_target_` を含む設定を
受理すると任意 import / instantiate の入口になるので、`task_id` のような狭い入力を
受けてサーバ所有の registry で展開する。

コントロール面にも認証が要る。CURVE は ZMQ データ面しか守らない。
ユーザー認証、session ownership、同時セッション quota、作成 rate limit、
idle / hard lifetime、task の allowlist、audit log が最低限。

### データ面 — ZMQ（RemoteEnv ↔ コンテナ直結）

| メッセージ | 向き | 内容 |
| --- | --- | --- |
| `step` | → | `{code, capture_video}` → `StepResult` |
| `render` | → | `{camera}` → JPEG |
| `frame` | ← | 実行中のフレーム（サーバ push、購読時のみ） |
| `ping` | ⇄ | ハートビート |

`ROUTER`（コンテナ）/ `DEALER`（クライアント）。1 ソケットでリクエスト応答とサーバ
push を兼ねる。シリアライズは既存の `capx/utils/msgpack_server_client_utils.py`
（msgpack + msgpack_numpy）を流用するが、message size 上限・schema・version が
無いため、着想の流用にとどめ新しい protocol module を作る。

### ポート（公開するのは 1 セッション 1 つ）

backend は worker に HTTP で話さない。生存確認も、backend 専用の CURVE 鍵で
`ping` する。だからコンテナが 127.0.0.1 に公開する HTTP ポートは要らない。

```
session_id ─┬─ container:  capx-ws-<id>
            ├─ network:    capx-ws-net-<id>  (--internal)
            ├─ zmq_port:   19500-19599  → 0.0.0.0 公開（RemoteEnv 用）
            ├─ curve_keys: セッションごとに生成
            └─ last_active → idle 回収
```

### 認証

**鍵はクライアントが生成し、公開鍵だけを backend に登録する。**
backend が `client_secret_key` を生成して平文 HTTP で返すと、盗聴された時点で
CURVE が無力化する。

```python
client_public_key, client_secret_key = zmq.curve_keypair()   # 秘密鍵は PC から出さない
resp = requests.post(f"{SERVER_URL}/sessions", json={
    "task_id": "cube_stack",
    "client_public_key": client_public_key.decode("ascii"),
})
```

worker 側は ZAP authenticator で登録済み公開鍵だけを許可する。CURVE server mode を
有効にしただけで許可クライアントが限定されるとは仮定せず、allowlist の試験を行う。

### 隔離

コンテナの受信を開けても送信の隔離は壊れない。`--internal` ネットワークを主インター
フェースにしてデフォルトルートを持たせないため。ただし **`0.0.0.0` 公開構成での
再検証が必要**（workshop の "verified on this host" は `127.0.0.1` 公開での検証）。

SAM3 / GraspNet / PyRoKi は `192.168.0.200` の 8114 / 8115 / 8116。コンテナはルートを
持たないので、固定宛先の `perception-proxy-*`（socat）をそのセッションのネットワークに
繋いで到達させる。`*_SERVICE_URL` は backend が `docker run` の `-e` で proxy 宛に
書き換えて注入するので、**参加者はこれらを設定しない**。

GPU は `--runtime nvidia`（`--gpus` ではない）。

### コンテナ制限

```
--memory 4g  --cpus 2  --pids-limit 512
--cap-drop ALL  --security-opt no-new-privileges
/workspace は read-only、/tmp は tmpfs 1g
```

加えて非 root user、`--read-only` rootfs、`--ulimit nofile` / `core=0`、seccomp、
image の digest pin、stdout/stderr とコードサイズの上限。

同時セッション数 × 4GB が GPU マシンの RAM を超えないよう、backend に上限セッション数を持たせる。

## 5. 公平性とセキュリティ

信頼境界を 3 段階に分ける。

1. 誤使用防止
2. 悪意ある生成コードへの防御
3. 悪意ある参加者への防御

**`expose_env=False` が担保できるのは 1 だけ。** 名前空間から `env` / `APIS` を落としても、
公開 API が bound method なので `goto_pose.__self__._env` で到達できる。`gc.get_objects()`、
`__globals__`、`__closure__` も使える。

> `expose_env=False` は accidental leakage と互換性管理のための API 制限であり、
> セキュリティ・不正防止境界ではない。

2 の境界は **プロセス分離**。生成コード実行プロセスと Gym 所有プロセスを分け、
API を RPC スタブとして渡す。reward オブジェクトが生成コード側のプロセスに存在しない状態を作る。

3 の境界は **サーバ側採点**。`evaluate` は worker が計算した結果を読むだけで、
クライアントは値を送らない。

oracle ソースも worker runtime image から除く。生成コードがパッケージソースを読めるなら、
`oracle_code` をクラス属性や YAML に置いたままでは取得できる。

### 実行時間

| 名前 | 何を測る | 役割 |
| --- | --- | --- |
| `execution_time_s` | worker が `step(code)` を処理している単調時計時間。知覚 API 待ちを含み、LLM 待ちを含まない | **比較用の上限**（既定 1000 秒） |
| `trial_wall_clock_s` | セッション開始から終了まで。LLM 待ち・通信待ち込み | 安全網（大きめ） |
| `agent_time_s` | 差分 | 記録のみ、上限なし |

計測は `time.monotonic_ns()`。worker / backend 側を権威とする。

**soft deadline は worker、hard deadline は backend が `docker kill`。**
生成コードは `signal.alarm(0)` や `signal.signal(SIGALRM, SIG_IGN)` で worker 内の
タイムアウトを無効化できるため、worker の soft deadline だけでは止められない。

### リトライ

| 種類 | 扱い |
| --- | --- |
| 予算超過（`execution_time_s` / `trial_wall_clock_s`） | **リトライしない**。その時点の状態で採点し、失敗として記録 |
| インフラ障害（コンテナ異常終了、ZMQ 切断、GPU OOM、セッション作成失敗） | **リトライする**（最大 3 回）。回数を結果に記録 |

タイムアウト後の env は再利用しない。**retry ごとに新規セッションを作る。**
同一 `request_id` の重複は cached response を返し、結果が不明なら session を破棄する。

## 6. 画像とメモリ

| 項目 | Robosuite | LIBERO |
| --- | --- | --- |
| 解像度 | 512 × 512 RGB | 800 × 512 RGB |
| 1 枚あたり | 0.75 MB | 1.17 MB |
| 記録間隔 | 5 sim step ごと（`_SUBSAMPLE_RATE`） | 同じ |
| エピソード上限 | 1500 sim step（`max_steps`） | 同じ |
| 正常時の最大枚数 | 300 枚 = 225 MB | 300 枚 = 351 MB |

**`truncated` の判定は `step(code)` が終わってからなので、1 回の `step()` の中で無限
ループするとバッファが青天井に伸びる。**

対策: **上限 900MB（全カメラ合計の byte budget）。超過時は間引き率を倍にする**
（バッファを 1 つ飛ばしで半分に間引き、以降の `_subsample_rate` を 5 → 10 に上げる）。
記録停止は結末を、リングバッファは冒頭を失うが、間引きならエピソード全体が残る。

枚数ではなく byte で持つのは、解像度が simulator で違うため。
900MB は Robosuite で 1200 枚相当（wrist 併用で 600 枚）、LIBERO で 768 枚相当（同 384 枚）。

間引きは `turn_frame_ranges` を壊す。フレームに `seq` / `sim_step` / `turn_id` / `camera`
を持たせ、index ではなく `turn_id` で引く。各 turn の先頭と末尾のフレームは pin する。

動画書き出しは固定 30fps なので、sampling rate を変えると時間軸が歪む。各フレームの
`sim_step`、effective sampling rate の変更履歴、original / retained / dropped の枚数を
成果物 metadata に記録する。

バッファは生の numpy のまま保持し、**JPEG 化は送信時のみ**。mp4 書き出しでデコード・
再エンコードが挟まらず、リプレイ画質が落ちない。

depth はバッファに入らない（`_record_frame()` は `depth=False`）。観測 dict 側にあり、
step ごとに置き換わるので累積しない。depth や segmentation を Agent に送る場合は
PNG（可逆）が必要で、JPEG は数値を壊す。

`--tmpfs /tmp:rw,size=1g` はメモリ cgroup に算入される。

### ストリーム

購読時のみ、10 fps、最新 1 枚のみ、JPEG q80。workshop の実装が既に「最新の 1 枚だけ送る」
形（`last_sent = count - 1`）なので、`STREAM_POLL_INTERVAL_SECONDS = 0.1` をそのまま使える。

購読制にするのが重要で、ベンチ実行（表示する人がいない）では帯域も JPEG エンコードの
CPU も使わない。JPEG 40KB はシーン依存の見積もりなので、代表タスクから 1000 枚程度
採取して q70/q80/q90 の p50/p95/max と encode 時間を実測する。

**ストリームを受けて表示するものは現時点で存在しない。** 手元 PC のローカル WebUI は今後作る。

## 7. simulator の切り替え

`pyproject.toml` の `conflicts` 宣言で robosuite と libero は排他。libero は自分用の
robosuite fork（`capx/third_party/libero_dependencies/robosuite`）を持ち込むため、
**1 つの venv・1 つの worker イメージに両方は入らない。**

**手元には simulator を入れない**ので、手元の extra が遠隔側を縛ることはない。

```
手元PC            : uv sync --extra remote       ← simulator なし
コンテナ（イメージA）: uv sync --extra robosuite    ← Dockerfile
コンテナ（イメージB）: uv sync --extra libero       ← Dockerfile.libero
```

タスクごとにどのイメージを使うかを backend が持ち、`docker run` の際に選ぶ。
local モードは従来どおり 1 venv 1 simulator。

`cap-x-workshop` の main（`6fa890a`）にこの構造が既に入っているので、LIBERO 対応は
ゼロから作るのではなく**移植**になる。

```python
# workshop/backend/config.py
TaskRuntime = Literal["robosuite", "libero"]

@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    config_path: str
    runtime: TaskRuntime = "robosuite"

# workshop/backend/session_manager.py
WORKER_IMAGES: dict[str, str] = {
    "robosuite": "capx-workshop-worker:latest",
    "libero":    "capx-workshop-worker-libero:latest",
}
worker_image = WORKER_IMAGES[runtime]
```

設計の経緯は `WORKSHOP_LIBERO_ENV.md`（cap-x-workshop）にある。

### Robosuite 固有の値を焼き込まない

LIBERO / BEHAVIOR を後から足せるよう、以下を守る。

- バッファ上限は byte budget（枚数ではない）
- 解像度は `_render_width` / `_render_height` から読む。512 を定数で書かない
- カメラ名は `render_camera_names` / `save_camera_name` を使う。`"robot0_robotview"` を直書きしない
- `TaskSpec.cameras` を worker が埋め、クライアントはそれを読むだけ
- `POST /sessions` の入力は `task_id`。クライアントから `env_config` を受けない

## 8. 設定ファイル

YAML を 4 ブロックに分ける。`agent:` の `_target_` は既存の `instantiate()` で解決できる。

```yaml
env:                          # Gym。現行のまま
  _target_: capx.envs.tasks.franka.franka_pick_place.FrankaPickPlaceCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level: franka_robosuite_cubes_low_level
    privileged: false
    apis: [FrankaControlApi]

api_servers: [...]            # Gym の周辺インフラ

bench:
  trials: 100
  num_workers: 12
  output_dir: ./outputs/franka_robosuite_cube_stack
  record_video: true
  budget: {execution_time_s: 1000, trial_wall_clock_s: 3000}
  feedback_level: stdout+image

agent:
  _target_: capx.baselines.capagent0.CaPAgent0
  model: google/gemini-3.1-pro-preview
  multi_turn_prompt: |
    ...
```

### 互換方針

- 既存 YAML は当面そのまま動かす。`agent:` が無ければ最上位の旧キー
  （`use_img_differencing` など）と CLI 引数を `CaPAgent0` の設定へ詰め替え、
  `DeprecationWarning` を出す
- `multi_turn_prompt` が `env.cfg` にあっても読む（警告付き）
- CLI は `LaunchArgs` を `BenchArgs` と `AgentArgs` に分割

### Agent の書式

ファイルパス + フラットな CLI。規約は「ファイルが `Agent` クラスを定義し、
`run(env, task, budget)` を持つ」の 1 つだけ。

```bash
uv run remote/trial.py --task cube_stack --agent agents/my_agent.py --model <model>
```

```python
# agents/my_agent.py
class Agent:
    def __init__(self, ctx):        # 任意。model / server_url / api_key / wire が入る
        self.ctx = ctx

    def run(self, env, task, budget):
        code = my_llm(task.default_prompt, self.ctx)
        env.step(code)
        return AgentResult()
```

`--model` などは `ctx` に入って渡るので、`--agent.model` のような入れ子指定は不要。

### remote の既定値

`--total-trials` の既定は 1、`--num-workers` は未指定（1 セッション逐次）。
明示すればその通りに動く。

**seed は trial 番号で、trial は 1 から始まる。** `capx/envs/runner.py:216` が
`list(range(1, config["total_trials"] + 1))`、`capx/envs/trial.py:648` が
`env.reset(options={"trial": trial}, seed=trial)`。既定では seed 1（0 ではない）。
1 始まりを維持する（既存挙動を変えないため）。

実行後の画面:

```
$ uv run remote/trial.py --task cube_stack --agent agents/my_agent.py
...
task_completed : False
reward         : 0.0
steps_used     : 3
```

参加者はこれを見て Agent を直し、また回す。このループは Bench から人間への一方向なので、
生成コードに正解を渡すことなく成立する。

## 9. 指標

| 指標 | 定義 |
| --- | --- |
| `task_completion_rate` | 最終状態の `task_completed` の平均（**主指標**） |
| `exec_ok_rate` | 最後の step が例外なく走った割合（現行の "success"。名前を改める） |
| `mean_reward` | 最終状態の reward の平均 |
| `steps_used` | `step()` の呼び出し回数 |
| `llm_calls` / `tokens` | 同梱クライアント経由は自動集計、自前呼び出しは自己申告（参考値） |
| `execution_time_s` / `trial_wall_clock_s` | 上記の時計 2 本 |
| `infrastructure_retries` | インフラ障害によるリトライ回数 |

`query_model` は現状 SDK の `usage` を捨てているので、返り値に追加する。
ただしカスタム Agent が直接 SDK を呼べる以上、Bench は完全な token 数を観測できない。
同梱 Agent の参考指標として位置づける。

結果に添えるもの: config、seed 範囲、Budget、`feedback_level`、Agent 名と設定、git commit。

## 10. 削除・移動

| 現在 | 移動先 | 持ち主 |
| --- | --- | --- |
| `capx/envs/launch.py` の CLI | `local/trial.py` | Bench |
| `capx/envs/launch.py` の `_run_web_ui` / `_ensure_frontend_built` | 削除 | — |
| `capx/envs/launch.py` の API サーバ検証 | `capx/bench/` | Bench |
| `capx/envs/runner.py` | `capx/bench/runner.py` | Bench |
| `_capture_initial_visual_feedback` / `_describe_initial_scene` | `capx/baselines/capagent0.py` | Agent |
| `_get_visual_differencing_feedback` / `_get_video_differencing_feedback` | `capx/agent_api/components/vdm.py` | Agent |
| `_query_initial_code` / `_handle_multi_turn_step` / `_parse_multi_turn_decision` | `capx/baselines/capagent0.py` | Agent |
| `_build_multi_turn_decision_prompt` | `capx/baselines/prompts.py` | Agent |
| `_extract_code` | `capx/agent_api/components/code_extract.py` | Agent |
| `use_oracle_code` の分岐 | `capx/baselines/oracle.py` | Agent |
| `_save_trial_artifacts` ほか | `capx/bench/artifacts.py` | Bench |
| `_run_single_trial` の残り、`TrialSummary`、リトライ・timeout | `capx/bench/` | Bench |
| `_patch_libero_goal` | env の `reset()` へ。`TaskSpec.instruction` を解決済みで返す | Gym |
| `evolve_skill_library`（`trial.py:919-932`） | 削除（読み出し側が未接続の死んだ経路） | — |
| `web-ui/` | 削除 | — |
| `capx/web/session_manager.py` | `capx/remote_env/server/` へ流用 | Bench |

`capx/skills/` モジュールは残す（stdlib のみで無害。読み出し側を繋げば復活できる）。
README の「auto-synthesized skill libraries」に対応する実働機能は
`scripts/skill_library_compilation/` と `FrankaControlApiReducedSkillLibrary`（API 階層）で、
そちらは今回の改修対象外。

`capx/web/` は「trial 全体をサーバで回す」設計なので、step 単位の API は別ルータとして
追加し、流用するのはセッション管理までにとどめる。

`CAPX_ARCHITECTURE.md` は日付付きスナップショットとして据え置く
（今回触るファイルへの言及が 36 箇所あり、1 行だけ直しても整合しない）。

## 11. 依存分離と remote/ の独立

### remote/ はルートと別の uv プロジェクト

手元PC（Windows / Linux / macOS のいずれもあり得る）の参加者は `remote/` から uv を実行する。
**ルートの `uv.lock` / `.venv` とは一切混ざらない。**

```
cap-x/
├── pyproject.toml     ┐
├── uv.lock            ├── GPUマシン用。robosuite / torch / CUDA / submodule
├── .venv/             ┘
│
└── remote/
    ├── pyproject.toml ┐
    ├── uv.lock        ├── 手元PC用。完全に独立
    └── .venv/         ┘
```

ルートの `[tool.uv.workspace] exclude` に `"remote"` を明記してあるので、
workspace としても取り込まれない。

参加者の手順は次の 1 行だけで、**submodule の取得も CUDA も要らない**。これは満たすべき要件。

```bash
cd remote && uv sync
```

`remote/pyproject.toml` に `[tool.uv] environments` は**書かない**。未指定なら全
プラットフォーム向けに解決されるため、手元PC の OS を問わない。

**実測（2026-09-26）**: `uv lock` が 44 パッケージを 494ms で解決、`uv sync` 成功、
import 確認済み。lock は `win_amd64` 49 件 / `manylinux_x86_64` 35 件 /
`macosx_arm64` 44 件を含む。

### ルートの universal lock は CUDA 無しでは通らない

```
error: Failed to generate package metadata for
       `nvidia-curobo==0.7.8 @ editable+capx/third_party/curobo`
OSError: CUDA_HOME environment variable is not set.
```

`uv lock` は **universal lock なので全 extra を解決する**。`nvidia-curobo` は
`curobo` extra 限定だが、`[tool.uv.sources]` で editable path 依存として宣言されて
いるため、uv がメタデータ生成のために setup.py を実行し、それが CUDA_HOME を要求する。

つまり `[tool.uv] environments` に手元PC のプラットフォームを足しても解決しない。
**この問題を避けるために `remote/` を独立させている。**

一方、**外部プロジェクトから `capx` にパス依存するのは失敗しない**。extra を選ばなければ
`nvidia-curobo` は依存グラフに入らず、メタデータ生成も走らない
（実測: 210 パッケージを 17.89 秒で解決）。

### remote/ から capx を使う条件

`remote/` は最終的に `capx.agent_api` の型と `capx.remote_env.client` を使う。
そのためには `capx` をパス依存として追加する必要があるが、**現状の base 依存が重いため
今はできない**。上記 210 パッケージには以下が含まれる。

```
torch, open3d, ray, transformers, robosuite, sam3, pyroki
```

`sam3` は submodule（`capx/third_party/sam3`）へのパス依存、`pyroki` は git 依存。
submodule を取得していない参加者の手元では `uv sync` が失敗する。

```
remote/ が capx にパス依存できる条件
    = capx の base dependencies が軽いこと
    = Phase 3.5（base の 19 パッケージを env extra へ降格）が完了していること
```

**Phase 3.5 は Phase 4C の前提ではなく、`remote/` 成立の前提そのもの。**

### base の降格

`[project] dependencies`（extras ではなく必須側）に 19 個の重いパッケージが入っている。

```
torch, torchvision, open3d, pyroki, sam3, ray, transformers,
pyrender, viser, trimesh, decord, pycocotools, h5py, scipy,
matplotlib, imageio[ffmpeg], opencv-python-headless,
robot_descriptions, yourdfpy
```

責務別に分ける。

```toml
[project]
dependencies = ["pydantic", "tyro", "rich", "requests", "openai", "pyyaml"]

[project.optional-dependencies]
server = ["fastapi", "uvicorn"]
env    = ["gymnasium", "numpy", "torch", ...]
robosuite = [...]
libero = [...]
```

成立させる条件は **`capx/agent_api/` と `capx/bench/` が重い依存を import しないこと**。
`capx/envs/__init__.py:3` の `from . import simulators` が import 時に環境登録を走らせるので、
remote の import graph に `capx.envs` が 1 箇所でも混ざると崩れる。

```python
def test_remote_import_is_light():
    import sys
    import capx.agent_api
    import capx.bench

    forbidden = {"torch", "open3d", "ray", "transformers", "robosuite"}
    assert forbidden.isdisjoint(sys.modules)
```

### 完了条件

Phase 3.5 は次で検証する。

- **submodule 無しの clone で `cd remote && uv sync` が通る**
- `capx` をパス依存に追加した状態でそれが成立する
- import 衛生テストが通る

## 12. フェーズ

各フェーズは前の挙動を変えずに進み、前後で成果物と summary が一致することを確認する。

| Phase | 内容 | 完了条件 |
| --- | --- | --- |
| 0A | 信頼境界と契約の明文化 | ネットワーク隔離を実測で確認。スキーマと seed 規約が確定 |
| 0B | 既存テストで基準を取る | oracle と quick スモークの結果を控える |
| 1 | 型と Local ファサード | oracle が reward 1.0 を出す |
| 2 | CaPAgent0 の抽出 | oracle と quick スモークが Phase 0B と同程度 |
| 3 | 設定・CLI 分割と `local/` | 既存 YAML が警告つきで従来どおり動く |
| 3.5 | 依存分離（extras） | **submodule 無しの clone で `cd remote && uv sync` が通る**。import 衛生テストが通る |
| 4A | worker コンテナ（Robosuite） | プロセス境界・バッファ・サーバ側採点が動く（request/response のみ） |
| 4B | backend・認証・quota | 認証付きで 1 セッション作成・破棄が通る |
| 4C | RemoteAgentEnv・`remote/` | 手元PC から 1 trial 回り、Local と結果が一致 |
| 4D | ストリーミング | 購読・backpressure・最新 1 枚が動く |
| 5 | 指標の整備 | 異なる Agent の結果が同じ形式で並べられる |
| 6 | LIBERO 対応 | LIBERO イメージで remote から 1 trial 回る。ワークショップ main からの移植が主体 |
| 7 | BEHAVIOR 対応 | 調査から。`capx/third_party/b1k` は uv workspace 外 |

### Phase 0B — 既存のテストで前後を比べる

専用の golden 比較は**作らない**。このベンチは同じ条件で走らせても同じ結果に
ならないので、結果を記録して突き合わせる方式が成立しない。

確認した事実:

- **reward は連続値**（距離ベース）で、同じ seed でも実行ごとに変わる。
  即座に例外で終わる——ロボットが 1 ミリも動かない——コードでも
  0.003 / 0.000 と揺れる
- **seed は robosuite に渡っていない**。`robosuite_cubes.py:122-125` が
  `seed` から作るのは capx 側の `self._rng` だけで、`self.robosuite_env.reset()`
  には渡らない。物体の初期配置を決めるのは robosuite 自身の RNG なので、
  `seed=1` を指定しても毎回違う配置から始まる
- oracle の経路は知覚サーバ（SAM3 ほか）の生死に依存する。サーバが落ちていれば
  `ValueError: No sam3 detections` で終わり、reward も成否も変わる

代わりに既存のものを使う。各フェーズの前後で走らせ、結果が同程度かを見る。

```bash
# oracle が reward 1.0 を出すか（知覚サーバが要る）
uv run --no-sync --active tests/test_environments.py \
    --env_name franka_robosuite_pick_place_code_env

# 10 trial のスモーク（LLM が要る）
./scripts/regression_test.sh quick
```

`scripts/regression_test.sh` は `QUICK_MIN_COMPLETED=2`（10 trial 中 2 件以上）の
ように**幅を持たせた基準**で判定する。揺れる系に対してはこちらが正しい形。

この方式で捕まえられないもの: multi-turn の往復数、`num_regenerations` /
`num_finishes` の集計、成果物の命名。Phase 2 でそこを移すときは、
コードレビューで見るしかない。

### seed が効いていない件

上記は今回のリファクタとは独立した問題だが、**設計の前提に関わる**。
本書 §1 は「同じ seed・同じ予算・同じ指標で評価する」としているが、
現状 seed は初期配置を固定していない。Agent 同士のスコアを比較するなら、
`robosuite_env.reset()` に seed を渡すか、robosuite 側の RNG を直接
シードする必要がある。Phase 5（指標の整備）で扱う。

## 13. 未決

なし。実装中に判断が要るものは Phase 0A で確定させる。
