# CaP-X remote

自分の Agent を書いて、GPU マシンのロボットシミュレータで動かす場所。
このフォルダの外は触らなくていい。

## はじめかた

```bash
cd remote
cp .env.example .env      # 管理者に聞いた値を入れる
uv sync
uv run --env-file .env trial.py
```

`agents/example_agent.py` が動けば準備完了。

## 自分の Agent を書く

`agents/example_agent.py` をコピーして中身を書き換える。

```bash
cp agents/example_agent.py agents/my_agent.py
uv run --env-file .env trial.py --agent agents/my_agent.py
```

- `env.step(code)` … ロボットを動かす Python コードを送る。結果が返る
- `env.render()` … 今の画像（JPEG）
- `task.instruction` / `task.api_docs` … やることと、使える関数の説明

LLM は手元から直接呼ぶ（`OPENAI_BASE_URL`）。GPU マシンからは呼ばない。

## 結果

実行が終わると `task_completed` と `reward` が表示される。
Agent の中からは見えない（`env.step()` が返すのは stdout / stderr だけ）。

## 注意

`agents/` のファイルは**あなたの PC でそのまま実行される**。他人の Agent を
コピーするときは中身を読んでから動かす。
