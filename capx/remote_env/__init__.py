"""手元PC ↔ GPU マシンの通信。

- `protocol`: msgpack のメッセージ形式
- `client.RemoteAgentEnv`: 手元PC 側。`AgentEnv` として振る舞う
- `worker.server`: コンテナ内。`LocalAgentEnv` を ZMQ で見せる
- `server`: backend。セッション管理と docker 起動（HTTP）

コントロール面（HTTP）とデータ面（ZMQ）を分ける。backend はデータ経路に
入らない——介在させると構造が複雑になり、HTTP の帯域を埋める。
"""
