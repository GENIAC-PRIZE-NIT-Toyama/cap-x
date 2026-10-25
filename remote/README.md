# CaP-X remote

自分の Agent を書いて、GPU マシンのロボットシミュレータで動かす場所。
このフォルダの外は触らなくていい。

GPU マシン側の準備（管理者向け）は [`docs/remote-server-setup.md`](../docs/remote-server-setup.md)。

## はじめかた

上から順に、ターミナルで打つ。

### 1. 用意する

- **git**
- **uv**（Python とライブラリを入れるツール）。入れたらターミナルを開き直す

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- **管理者に聞くもの**: GPU マシンの URL、LLM の URL・モデル名・キー。学外からは VPN を繋いでおく

### 2. ダウンロードして設定する

```bash
git clone <このリポジトリの URL> cap-x
cd cap-x/remote
cp .env.example .env
```

`.env` をエディタで開いて、次の 5 つを書く。

| 項目 | 書くもの |
|---|---|
| `CAPX_ENV_SERVER_URL` | GPU マシンの URL |
| `OPENAI_BASE_URL` | LLM の URL |
| `OPENAI_API_KEY` | LLM のキー。管理者に言われなければ `dummy` のまま |
| `OPENAI_BASE_MODEL` | LLM のモデル名。分からなければ空のまま 4 へ進むと、選べる名前が出る |
| `CAPX_LLM_WIRE` | 管理者に言われなければ `chat` のまま |

### 3. 必要なものを入れる

```bash
uv sync
```

初回はダウンロードで時間がかかる。

### 4. つながるか確かめる

```bash
uv run doctor.py
```

GPU マシンと LLM に届くかを順に調べる。全部 OK なら次へ。駄目な項目には直し方が出る。

### 5. 例の Agent で 1 回動かす

```bash
uv run trial.py --agent agents/example_agent.py --task cube_stack
```

同梱の例の Agent で、キューブを積むタスク（`cube_stack`）を 1 回。終わると `task_completed`
（成功したか）と `reward` が出る。成功しなくても、最後まで走れば準備完了。

### 6. 自分の Agent を作る

```bash
cp agents/example_agent.py agents/my_agent.py
```

`agents/my_agent.py` を開いて、`run` の中を書き換える。

- `task.instruction` … やること（ゴール文）
- `task.api_docs` … ロボットを動かすのに使える関数の説明
- `env.step(code)` … ロボットを動かす Python コードを送る。stdout / stderr が返る
- `env.render()` … 今の画像（JPEG）

### 7. 自分の Agent を 1 タスクで試す

```bash
uv run trial.py --agent agents/my_agent.py --task cube_stack
```

ほかのタスクで試すときは `--task` を書き換える（[タスク一覧](#タスク一覧)）。

### 8. まとめて評価する

```bash
uv run suite.py suites/quick.yaml --agent agents/my_agent.py
```

10 タスクを順に回し、最後にタスクごとの成功率の表が出る。本評価は `suites/full.yaml`（35 タスク）。

---

ここから下は、必要になったときに読む。

## 動作環境

- Python 3.10〜3.13。手元に無ければ `uv sync` が自動で入れる。シミュレータも GPU も要らない
  （Windows / macOS / Linux）
- このリポジトリの submodule は要らない（`git clone` だけでよい）
- GPU マシンの 2 つのポートに届く必要がある。セッションを作るポート（例 8080）と、
  シミュレータに繋ぐポート（19500〜19599）
- LLM（OpenAI 互換。vLLM など）は手元 PC から直接呼ぶ。GPU マシンからは呼ばない

## 設定ファイル（`.env`）

`remote/.env` は、どこから実行しても自動で読まれる（すでに設定されている環境変数は上書きしない）。
`.env` 以外を使うときだけ `--env-file` を付ける（例: `uv run trial.py --agent agents/my_agent.py --task cube_stack --env-file openai.env`）。

はじめかたの 5 つのほかに、Langfuse（[トレース](#トレースlangfuse)）と Snapshot（[Snapshot](#過去の-step-から分岐するsnapshot)）の
項目がある。どちらも任意。

## タスク一覧

| 種類 | 名前 |
|---|---|
| robosuite | `cube_stack`, `cube_lifting`, `cube_restack`, `nut_assembly`, `spill_wipe`, `two_arm_lift` |
| LIBERO（標準 5 suite、130 タスク） | `libero_spatial_0`〜`9`, `libero_object_0`〜`9`, `libero_goal_0`〜`9`, `libero_10_0`〜`9`, `libero_90_0`〜`89` |

```bash
uv run trial.py --agent agents/my_agent.py --task libero_object_3
```

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

## オプション

主なもの（`--help` で全部）。

`trial.py`（1 つのタスクを回す）:

| オプション | 既定 | 内容 |
|---|---|---|
| `--agent` | `agents/example_agent.py` | Agent ファイル |
| `--task` | `cube_stack` | タスク名（[タスク一覧](#タスク一覧)） |
| `--total-trials` | 1 | 試行回数 |
| `--seed` | 試行番号 | 初期配置の seed |
| `--model` | `.env` の `OPENAI_BASE_MODEL` | LLM のモデル名 |
| `--output-dir` | `remote/outputs/` | 結果を置く場所 |
| `--no-save` | | 結果を保存しない |
| `--record-video False` | 結果を保存するときは保存する | 動画を保存しない |
| `--no-trace` | | Langfuse に送らない |
| `--trace-step-videos` | | `env.step` ごとの動画も Langfuse に送る |
| `--env-file` | `remote/.env` | 別の設定ファイルを読む |

`suite.py`（スイートの YAML を最初に書く。例 `uv run suite.py suites/quick.yaml --agent agents/my_agent.py`）:

| オプション | 既定 | 内容 |
|---|---|---|
| `--agent` | `agents/example_agent.py` | Agent ファイル |
| `--parallel` | 1 | 同時に回すタスクの数（1〜10） |
| `--model` | `.env` の `OPENAI_BASE_MODEL` | LLM のモデル名 |
| `--output-dir` | `remote/outputs/` | 結果を置く場所 |
| `--no-record-video` | | 動画を保存しない |
| `--no-trace` | | Langfuse に送らない |
| `--trace-step-videos` | | `env.step` ごとの動画も Langfuse に送る |
| `--env-file` | `remote/.env` | 別の設定ファイルを読む |

## Agent の書き方

`agents/example_agent.py` をコピーして中身を書き換える。クラス名は `Agent` のまま、
`BaseAgent` を継承して `run` を書く（例のとおり）。

- `env.step(code)` … ロボットを動かす Python コードを送る。結果が返る
- `env.render()` … 今の画像（JPEG）
- `task.instruction` / `task.api_docs` … やることと、使える関数の説明

LLM は手元から直接呼ぶ（`OPENAI_BASE_URL`）。GPU マシンからは呼ばない。

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
uv run suite.py suites/my_suite.yaml --agent agents/my_agent.py
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

`--parallel N`（1〜10）で、N タスクずつ同時に回せる。セッションを開く前に GPU マシンの
空き（`GET /sessions` のセッション数とメモリ）を確かめ、混んでいれば空くまで待つ（最大 30 分）。
並列のときは、ほかの人のために 2 枠を空けておく。並べると GPU・LLM サーバを取り合うので、
`wall_clock` は 1 本ずつより長くなる。並べた数が違う実行どうしで時間を比べない。

```bash
uv run suite.py suites/full.yaml --agent agents/my_agent.py --parallel 3
```

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

## 過去の Step から分岐する（Snapshot）

`.env` に `CAPX_SNAPSHOT_ENABLED=true` を書くと、reset の直後と毎 `env.step()` の後に、
シミュレータと生成コードの変数・関数の状態を GPU マシン側に保存する。Snapshot ID は
Langfuse の `env.step` の metadata（`snapshot_after`）、`result.json` の `snapshots`、
`steps/step_NN.log` に残る。

あとで、その ID から新しい worker で再開して、別のコードを試せる。前の Step で定義した
関数や変数はそのまま使える。元の trial・Trace・Snapshot は変わらない。

```bash
uv run examples/resume_simple.py --snapshot snap_... --task libero_object_3 --code "print(helper())"
uv run examples/fork_compare.py --snapshot snap_... --task libero_object_3 --original old.py --fixed new.py
```

使い方・仕組み・制約は [docs/Snapshot.md](../docs/Snapshot.md)。

## うまくいかないとき

自分の Agent ファイルまで含めて、原因を一度に調べられる:

```bash
uv run doctor.py --agent agents/my_agent.py
```

GPU マシン・認証・空き・LLM・Agent ファイルを順に調べて、駄目な項目には直し方を出す。

## 注意

`agents/` のファイルは**あなたの PC でそのまま実行される**。他人の Agent を
コピーするときは中身を読んでから動かす。
