"""Snapshot / Resume / Fork。Agent Step の境界で、Gym と生成コードの Namespace を保存して戻す。

    blob.py       Snapshot 1 つの形式（msgpack。pickle を含む部分は bytes のまま運ぶ）
    namespace.py  policy プロセスの globals を cloudpickle で保存・復元する（policy プロセスの中だけで使う）
    gym_state.py  シミュレータ・コントローラ・API 層の状態を、データだけの dict にする
    store.py      backend が使う永続化先（S3 / MinIO、テスト用のディレクトリ）
    s3.py         S3 互換ストレージを requests だけで呼ぶ最小のクライアント（SigV4）

この package の `__init__` は何も import しない。backend（blob / store）と
policy プロセス（namespace）が、シミュレータを import せずに使えるようにするため。
設計と制約は docs/Snapshot.md。
"""
