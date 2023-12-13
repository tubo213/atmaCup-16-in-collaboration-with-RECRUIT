# atmacup-16-in-collaboration-with-recruit

## Requirements
- [rye](https://github.com/mitsuhiko/rye)

## 環境構築

## パスの設定

```bin/conf/dir/local.yaml```を自分の環境に合わせて設定

```yaml
data_dir: # データのディレクトリ
processed_dir: # 前処理済みデータのディレクトリ
output_dir: # 出力ディレクトリ
model_dir: # モデルのディレクトリ
sub_dir: ./
```

## torch,pytorch_geometricの設定

pyproject.tomlを自分の環境に合わせて設定

```toml
[[tool.rye.sources]]
name = "torch"
url = "https://download.pytorch.org/whl/cu121" # 書き換える
type = "index"

[[tool.rye.sources]]
name = "pytorch-geometric"
url = "https://data.pyg.org/whl/torch-2.1.0+cu121.html" # 書き換える
type = "find-links"
```

### 仮想環境の作成
```bash
rye sync
```

### 仮想環境の有効化
```bash
. .venv/bin/activate
```

## 前処理

1. `./data/`配下にデータを解凍

2. prepare_data.pyを実行
```bash
rye run python bin/prepare_data.py
```

3. (Optional) PNAConvを利用する場合はprepare_deg.pyを実行
```bash
rye run python -m bin/prepare_deg.py k=1,2,3
```

## 学習

以下コマンドでCV=0.397,LB=0.428をのモデルを学習

```bash
rye run python bin/train.py k=3 num_layers=10 model=pdn trainer.use_amp=True exp_name=exp001
```

[hydra](https://hydra.cc/docs/intro/)を利用しているため、グリッドサーチが簡単にできます。
以下は、model=pdn,gat,transformer,lr=0.01,0.001,0.0001の組み合わせを全て試すコマンドです。

```bash
rye run python bin/train.py -m model=pdn,gat,transformer lr=0.01,0.001,0.0001 exp_name=exp002
```

## 推論

exp001のモデルを利用して推論。  
./output/inference/exp001/singleに推論結果が出力されます。

```bash
rye run python bin/inference.py exp_name=exp001 phase=test use_amp=false
```