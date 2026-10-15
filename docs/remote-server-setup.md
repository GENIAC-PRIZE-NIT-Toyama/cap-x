# リモート実行の GPU マシン側の準備（管理者向け）

参加者が手元 PC の `remote/trial.py` から使う、GPU マシン側の準備。参加者向けの手順は
[`remote/README.md`](../remote/README.md)、設計は [`docs/RemoteDevelopment.md`](RemoteDevelopment.md)。

```
手元 PC ──HTTP :8080──────▶ backend（セッションを作り、worker を docker run する）
        ══ZMQ :19500〜19599═▶ worker コンテナ（1 セッション 1 つ。シミュレータと生成コード）
                                 └─ 中継（capx-proxy-*）─▶ 知覚 API（SAM3 / GraspNet / PyRoKi / Molmo2）
```

## 0. 前提

- Docker と NVIDIA Container Toolkit（worker は `--runtime nvidia` で起動する）。実行するユーザーが
  `docker` を sudo なしで使えること
- uv と Python 3.10
- 知覚 API（SAM3・GraspNet・PyRoKi）が、このマシンか同じネットワークのどこかで動いていること
  （起動は README の「Perception API / PyRoKi / OpenAI 互換 API のセットアップ」）。
  LIBERO を使うなら Molmo2（OpenAI 互換の VLM サーバ）も
- ディスク: worker image は 1 つ約 12GB（robosuite 用と LIBERO 用の 2 つ）。ビルドキャッシュでさらに増える

## 1. submodule を取る

リポジトリ直下で。

```bash
git submodule update --init capx/third_party/sam3 capx/third_party/robosuite \
    capx/third_party/LIBERO-PRO capx/third_party/libero_dependencies/robosuite
```

## 2. worker image を作る

```bash
docker build -f docker/worker/Dockerfile -t capx-worker:latest .                # robosuite
docker build -f docker/worker/Dockerfile.libero -t capx-worker-libero:latest .  # LIBERO
```

コードを更新したら（`git pull`）、worker の中で動くコードが変わったときは作り直す。古い image を
消してから作ると、ディスクを使い切らない（`docker image prune`、`docker builder prune`）。
別の名前の image を使うときは、backend を起動する環境で `CAPX_WORKER_IMAGE` /
`CAPX_WORKER_IMAGE_LIBERO` を設定する（digest での固定もできる）。

## 3. 知覚 API への中継を起動する

worker は外へ出られないネットワーク（`--internal`）に置かれ、知覚 API へはこの中継だけを通る。

```bash
cd docker/backend
cp .env.example .env    # 知覚 API の実際の宛先（host:port）と MOLMO2_VLM_URL を書く
docker compose up -d
cd ../..
```

`docker ps` に `capx-proxy-sam3` / `capx-proxy-graspnet` / `capx-proxy-pyroki` / `capx-proxy-molmo`
が出れば動いている。

## 4. backend の環境と設定

```bash
uv venv -p 3.10 && source .venv/bin/activate
uv sync --active --extra robosuite --extra server

cp .envrc.example .envrc   # 次の値を書く
source .envrc
```

| 変数 | 内容 |
| --- | --- |
| `MOLMO2_VLM_URL` / `MOLMO2_VLM_MODEL` | Molmo2 の宛先とモデル名。backend が worker に渡す（LIBERO で使う） |
| `CAPX_GPU_UUIDS` | worker に割り当てる GPU（`nvidia-smi -L` の UUID をカンマ区切り）。省略すると全 GPU が見える |
| `CAPX_ENV_SERVER_TOKENS` | 認証を有効にするときだけ（`名前:トークン` をカンマ区切り）。既定は認証なし |
| `CAPX_WORKER_MEMORY_ROBOSUITE` / `CAPX_WORKER_MEMORY_LIBERO` | worker 1 つのメモリの上限を変えるときだけ（既定 `4g` / `5g`） |

## 5. backend を起動する

```bash
uv run --no-sync --active -m capx.remote_env.server
```

主なオプション（`--help` で全部）:

| オプション | 既定 | 内容 |
| --- | --- | --- |
| `--port` | 8080 | 参加者がセッションを作る HTTP のポート |
| `--port-start` / `--port-end` | 19500 / 19600 | worker に割り当てる ZMQ のポートの範囲（終わりは含まない） |
| `--max-sessions` | 20 | 同時セッション数の上限 |
| `--memory-budget-gb` | 100 | 全 worker のメモリの上限の合計。このマシンの RAM から OS などの分を残した値にする |
| `--idle-ttl-min` / `--max-lifetime-min` | 20 / 120 | 使われていないセッションと、長すぎるセッションを回収するまでの時間（分） |
| `--public-host` | なし | 参加者に教える ZMQ の宛先。省略すると、参加者が繋いだ宛先をそのまま使う |

起動すると、前回の backend が残した worker を片付けてから受け付けを始める。止めるときは Ctrl-C（動いている worker も片付ける）。

## 6. ポートを開ける

参加者の PC から、次の 2 つに届く必要がある。ファイアウォール（ufw など）で許可する。

- backend: `--port`（既定 8080）
- worker: `--port-start`〜`--port-end` の範囲（既定 19500〜19599）

## 7. 動作を確かめる

GPU マシンで:

```bash
curl http://localhost:8080/health     # 返事があれば backend は動いている
curl http://localhost:8080/sessions   # 使用中の数・上限・メモリの使用量
```

参加者の PC で（`remote/README.md` の準備のあと）:

```bash
cd remote
uv run --env-file .env doctor.py      # 到達・タスク一覧・空きなどを順に調べる
uv run --env-file .env trial.py       # cube_stack を 1 回
```

## うまくいかないとき

- **`POST /sessions` が 500 になる / すぐ失敗する**: worker のログを見る（`docker ps -a` で
  `capx-w-*` を探し、`docker logs <名前>`）。image を作り直していない、submodule が無い、など
- **ポートが使用中で backend が起動しない**: `--port` で別のポートにする（参加者の
  `CAPX_ENV_SERVER_URL` も合わせる）
- **参加者が「返事が無い」で失敗する**: 19500〜19599 が届いていない（ファイアウォール・VPN）か、
  worker がメモリ不足で強制終了された（`docker events --filter event=oom`）
- **LIBERO で物体が見つからない**: Molmo2 の中継（`capx-proxy-molmo`）と `MOLMO2_VLM_URL` を確かめる
