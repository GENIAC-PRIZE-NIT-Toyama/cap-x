# CaP-X remote

自分の Agent を書いて、GPU マシンのロボットシミュレータで動かす場所。
このフォルダの外は触らなくていい。

## 準備

- **Python 3.10〜3.13** と [uv](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`）。
  シミュレータも GPU も要らない（Windows / macOS / Linux）
- **このリポジトリ**。submodule は要らない（`git clone` だけでよい）
- **GPU マシンに届くこと**。管理者に聞いた GPU マシンの、セッションを作るポート（例 8080）と、
  シミュレータに繋ぐポート（19500〜19599）の両方に届く必要がある。学外からは VPN を繋いでおく
- **LLM のエンドポイント**（OpenAI 互換。vLLM など）。手元 PC から直接呼ぶ

GPU マシン側の準備（管理者向け）は [`docs/remote-server-setup.md`](../docs/remote-server-setup.md)。

## はじめかた

```bash
git clone <このリポジトリの URL> cap-x
cd cap-x/remote
cp .env.example .env      # 管理者に聞いた値を入れる
uv sync
uv run --env-file .env trial.py
```

`agents/example_agent.py` が動けば準備完了。

動かないときは、原因を一度に調べられる:

```bash
uv run --env-file .env doctor.py --agent agents/my_agent.py
```

## タスクを選ぶ

`--task` で選ぶ。省略すると `cube_stack`。

```bash
uv run --env-file .env trial.py --task libero_object_3
```

| 種類 | 名前 |
|---|---|
| robosuite | `cube_stack`, `cube_lifting`, `cube_restack`, `nut_assembly`, `spill_wipe`, `two_arm_lift` |
| LIBERO（標準 5 suite、130 タスク） | `libero_spatial_0`〜`9`, `libero_object_0`〜`9`, `libero_goal_0`〜`9`, `libero_10_0`〜`9`, `libero_90_0`〜`89` |

今選べる一覧は `doctor.py` でも見られる。LIBERO のタスクは、やること（ゴール文）が
タスクごとに違う。`task.instruction` に入っている。

### Low-Level Primitives で回す

タスク名の後ろに `_low` を付けると、同じタスクを Low-Level Primitives の API で回せる
（例 `cube_stack_low`, `libero_90_46_low`）。`two_arm_lift` 以外の全タスクにある。
シーン・ゴール・判定は同じで、Agent に渡る関数（`task.api_docs`）だけが違う。

| | 主な関数 |
|---|---|
| High-Level（`cube_stack` など） | `get_object_pose`, `sample_grasp_pose`, `goto_pose`, `open_gripper` / `close_gripper` |
| Low-Level（`cube_stack_low` など） | `get_observation`, `segment_sam3_text_prompt` / `segment_sam3_point_prompt`, `plan_grasp`, `get_oriented_bounding_box_from_3d_points`, `solve_ik`, `move_to_joints`, `open_gripper` / `close_gripper` |

Low-Level は 1 つの動作を細かく書くので、ステップ数を多く使いがち。High と比べるときは、
予算超過（`budget`）で終わった試行の数も見る。

## 自分の Agent を書く

`agents/example_agent.py` をコピーして中身を書き換える。

```bash
cp agents/example_agent.py agents/my_agent.py
uv run --env-file .env trial.py --agent agents/my_agent.py
```

クラス名は `Agent` のまま、`BaseAgent` を継承して `run` を書く（例のとおり）。

- `env.step(code)` … ロボットを動かす Python コードを送る。結果が返る
- `env.render()` … 今の画像（JPEG）
- `task.instruction` / `task.api_docs` … やることと、使える関数の説明

LLM は手元から直接呼ぶ（`OPENAI_BASE_URL`）。GPU マシンからは呼ばない。
LLM を使う例は `agents/multiturn_agent.py`（コードを書かせて実行し、結果を返して次を書かせる）。

## 結果

実行が終わると `task_completed` と `reward` が表示される。
Agent の中からは見えない（`env.step()` が返すのは stdout / stderr だけ）。

結果は `remote/outputs/<モデル名>/<タスク名>/<実行した日時>/` に保存される（どこから実行しても同じ。
実行ごとにフォルダが分かれる）。置き場所は `--output-dir` で変えられ、`--no-save` で保存しない。
試行ごとのフォルダ（`trial_01_.../`）に次が入る。

| 場所 | 中身 |
|---|---|
| `result.json` | 結果（task_completed / reward / 使ったステップ数 / トークン数 / 予算 など） |
| `steps/` | 各ステップで実行したコードと、その stdout / stderr |
| `prompts_and_responses/`, `all_responses.json` | LLM の入出力（`capx.llm.client.query_model` を使った分） |
| `videos/` | ターンごとの動画と、つないだ動画（`combined.mp4`） |
| `images/` | `env.render()` で取った画像 |
| `artifacts/` | `AgentResult(artifacts={...})` で渡したもの |

主な指標は `task_completed`（タスクを達成したか）。`exec_ok` は最後のステップの
コードがエラーなく走ったかで、タスクの成否ではない。

## 評価スイート

いくつかのタスクをまとめて評価するときは、タスクを YAML に書いて `suite.py` で回す。

```yaml
# suites/my_suite.yaml
name: my_suite      # 保存先のフォルダ名。省略するとファイル名
trials: 3           # 各タスクの試行回数。省略すると 1
seed: 42            # 書くと全試行で固定。省略すると試行ごとにランダム
max_steps: 8        # 1 試行で env.step() を呼べる回数（1〜20）。省略すると 10
tasks:
  - cube_stack
  - libero_object_3
```

```bash
uv run --env-file .env suite.py suites/my_suite.yaml --agent agents/my_agent.py
```

最後に、タスクごとの指標の表が出る。用意してあるスイート:

| スイート | タスク | 使いどころ |
|---|---|---|
| `suites/quick.yaml` | robosuite 3 + LIBERO-90 5 + LIBERO-10 2（10 タスク） | 素早く確かめる。`full` の部分集合 |
| `suites/full.yaml` | robosuite 5（`two_arm_lift` を除く）+ LIBERO-90 20 + LIBERO-10 全 10（35 タスク） | 本評価 |
| `suites/quick-low.yaml` / `suites/full-low.yaml` | `quick` / `full` と同じタスクの `_low` 版 | Low-Level Primitives で回し、High と比べる |

LIBERO-90 は、操作の種類・シーン・複雑さが重ならないように選んでいる（理由は YAML のコメント）。
どちらも `max_steps: 20`。ただし GPU マシン側の上限が今は 10 なので、それを上げるまでは
11 回目の `env.step()` で止まり、予算超過（`budget`）として数えられる。

`--parallel N`（1〜4）で、N タスクずつ同時に回せる。セッションを開く前に GPU マシンの
空き（`GET /sessions` のセッション数とメモリ）を確かめ、混んでいれば空くまで待つ（最大 30 分）。
並列のときは、ほかの人のために 2 枠を空けておく。並べると GPU・LLM サーバを取り合うので、
`wall_clock` は 1 本ずつより長くなる。並べた数が違う実行どうしで時間を比べない。

| 指標 | 意味 |
|---|---|
| `pass@1` | タスクを達成した試行の割合（Zero-shot。Agent は試行ごとに読み込み直す） |
| `error_rate` | 推論の失敗（生成コードのエラー・Agent の例外・`max_steps` などの予算超過）に遭遇した試行の割合 |
| `wall_clock` | 1 試行にかかった時間（LLM の待ちを含む） |

全体の値はタスクごとの値の平均（macro）。

失敗は 3 種類に分けて扱う。

- **推論の失敗**: Agent の結果として数える
- **インフラの失敗**（GPU マシンとの接続断、LLM のタイムアウト・接続断）: 同じ試行を最大 3 回やり直す。
  それでも駄目なら、その試行は指標の分母から外し、`infra` の行に件数を出す。
  LLM に繋がらないままのときは、残りも全部同じになるのでスイートを止める
- **設定の失敗**（LLM の認証エラー、GPU マシンがセッションを断った（4xx）、Agent を読み込めない、
  実行中に Agent ファイルが書き換わった）: その場で止まる。直してからもう一度実行する

失敗の種類は、Agent から飛んできた例外で見分ける。**LLM の呼び出しで起きた例外は
`try` で捕まえずにそのまま投げる**。握りつぶすと、LLM のタイムアウトが Agent の失敗として数えられる。

同梱の LLM クライアント（`capx.llm.client.query_model`）は、LLM サーバが 429 / 5xx を
返すと、中で数分待ってやり直し続ける。このやり直しは `suite.py` からは見えず、待った時間は
`wall_clock` に入る。`wall_clock` が飛び抜けて長い試行は、LLM サーバの混雑を疑う。

結果は `outputs/<モデル名>/suite_<名前>/<日時>/` に、タスクごと・試行ごとに保存される
（中身は `trial.py` と同じ）。集計は `summary.json`。途中で止まったときは、それまでに
回した試行だけで集計して終わる。続きからは回せないので、直してからもう一度実行する。
実行中に Agent ファイルを書き換えると止まる（違う Agent の結果が混ざるため）。
集計だけやり直すときは `uv run suite_summary.py <実行フォルダ>`。

## トレース（Langfuse）

`.env` に `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` があれば、
`trial.py` も `suite.py` も、試行ごとの挙動を Langfuse に送る。キーが無ければ送らない
（実行は今までどおり）。キーがあっても `--no-trace` で止められる。キーと URL は管理者に聞く。
誰の実行かは `CAPX_USER` で区別する（ローマ字で書く。日本語だと付かない）。

Agent には何も書かなくてよい。OpenAI SDK の呼び出し（`capx.llm.client.query_model` も、
自分で作った `OpenAI()` も）と `env.step()` が自動で記録される。OpenAI SDK 以外で LLM を
呼んだ分は記録されない。

| Langfuse | 中身 |
|---|---|
| Session | 1 回の実行。ID は保存先と同じ `<モデル名>/<タスク名 or suite_名前>/<日時>` |
| Trace | 1 試行（名前はタスク名）。input に task・instruction・seed、output に結果と動画（`combined.mp4`）。User は `CAPX_USER`、Version は Agent ファイルのハッシュ |
| Generation | LLM の呼び出し。プロンプト・応答・トークン数。画像もそのまま見える |
| Tool `env.step` | 送ったコードと、`ok` / `stdout` / `stderr`。エラーは WARNING で色が付く |
| Score | `reward` / `task_completed` / `exec_ok` / `failure_kind`（失敗したときだけ） |

呼んだ順に並ぶので、LLM がどのコードを書き、どう失敗して、次に何を直したかを 1 画面で追える。
タグは `trial` か `suite` と `suite:<名前>`。

GPU マシン・LLM サーバの障害でやり直した試行は、やり直すたびに別のトレースになる。
失敗したほうには `infra_failed` タグが付き、score は付かない（集計から外れるのと同じ）。
`--no-save` のときは動画を保存しないので、動画も送らない。

`--trace-step-videos` を付けると、`env.step` ごとにその step の動画も送る。どの step で
何が起きたかを動画で追えるが、Langfuse の容量が試行あたり約 2 倍（10 step で 2.5〜3.6MB、
20 step で 5〜7MB）になる。失敗した試行を 1 つ詳しく見たいときに、`trial.py` で回し直す使い方を想定している。

## 注意

`agents/` のファイルは**あなたの PC でそのまま実行される**。他人の Agent を
コピーするときは中身を読んでから動かす。
