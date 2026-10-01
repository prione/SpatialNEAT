# SpatialNEAT

標準NEATを土台にした、2D/3D ES-HyperNEATの配置探索と再帰ニューラルネットワークの独立Pythonパッケージです。
発達は任意の便利機能です。0.2.0では成長・成熟・刈り込み・再編・近似圧縮を個別の操作として扱えます。
`evo`、PyBullet、身体モデル、課題、報酬関数には依存しません。

GitHub: [prione/SpatialNEAT](https://github.com/prione/SpatialNEAT)。PyPIには公開していません。
互換性のため、配布名は `growth-hyperneat`、Pythonのインポート名は `growth_hyperneat` を維持しています。
公開用ライセンスは未選定です。公開リポジトリであることだけで、再配布・改変の許諾を意味するものではありません。

## インストール・実行

リポジトリを取得して実行します:

```sh
git clone https://github.com/prione/SpatialNEAT.git
cd SpatialNEAT
python -m pip install -e ".[test]"
python -m pytest -q
python examples/growth_demo.py
python examples/development_demo.py
python examples/evolve_memory.py --generations 3 --population 8
```

Python 3.10以上、NumPy、`neat-python==0.92`が必要です。NEATの対応版は検証した
0.92に固定しています。既存`evo`との互換性のためであり、最新NEAT対応を意味しません。

## 役割とAPI

| モジュール | 責任 |
| --- | --- |
| `model` | 不変な神経・領域・Substrate・疎なネットワーク定義、整合性検証 |
| `expression` | CPPN座標問い合わせ、四分木/八分木、分散・帯域抽出、反復探索、到達可能性の整理 |
| `runtime` | リーク付き同期RNN、学習差分保持、保存と復元。従来の`grow`入口も維持 |
| `development` | 発達の調整役。変更候補の試験、採用、成長・刈り込み・移動、取り消し |
| `migration` | 明示した構造変更に対する状態・重み・成熟度の移行、到達性・安定性検証 |
| `maturation` | 接続ゲートを時間に応じて有効化する独立した規則 |
| `compression` | 任意の隠れ神経統合。状態・重みを射影する近似圧縮で、生物の成長とは区別 |
| `neat_adapter` | 標準`DefaultGenome`のCPPN化と標準NEAT設定の読み込み |

```python
from growth_hyperneat import (
    ESConfig, ESDeveloper, GrowingRNN, NeatCPPN, Node, Region, Substrate,
)

substrate = Substrate(
    inputs=(Node("sensor", (0, 0, -1), "input"),),
    outputs=(Node("motor", (0, 0, 1), "output"),),
    regions=(Region("core", (-1, -1, -1), (1, 1, 1), leak=0.2),),
)
developer = ESDeveloper(ESConfig(max_depth=2))
# genome: 標準neat.DefaultGenome、config: 標準neat.Config
result = developer.develop(NeatCPPN(genome, config), substrate)
brain = GrowingRNN(result.network)
action = brain.step({"sensor": 0.5})["motor"]
```

`load_neat_config(dimensions=3, population_size=12)`で同梱設定を読み込めます。
`neat.Population(config).run(evaluate_genomes, generations)`にそのまま接続でき、独自のゲノム、
交叉、変異、種分化、繁殖クラスは不要です。各評価では新しいRNNを作って状態を独立させます。
`express_genome(...)`は配置探索とRNN作成をまとめる便利関数です。

## 配置・接続を決める方法

入力/出力の位置・役割は利用側が指定します。隠れ神経の位置・数を固定指定する必要はありません。
入力の出力側接続パターンから隠れ候補を抽出し、隠れから反復探索し、出力の入力側パターンも
抽出します。分散が大きい領域を細分化し、両側の隣接点との値の差による帯域判定を行います。
最後に入力から到達可能で、かつ出力へ到達できる隠れ神経・接続だけを残します。

2Dは四分木、3Dは八分木です。複数の`Region`で中央/局所モジュールを表現できます。
領域間接続は標準では許可され、`connection_filter(source_node, target_node)`で制限できます。
座標は自動正規化されません。全領域・アンカーを一貫した座標系で指定してください。
重なった領域は別モジュールとして扱われ、同じ位置でも異なる神経IDになります。

標準のCPPN入力は`[source座標, target座標, 距離, 1]`で、3Dは8入力、2Dは6入力です。
CPPNの第0出力を[-1, 1]にクリップして接続パターンに使い、`weight_scale`で実重みへ変換します。
CPPNは`query(values)`を持つ任意のステートレス実装で置換できます。
`ESDeveloper(features=...)`で座標ベクトル2本から特徴を作る関数も差し替え可能です。

これはRisi & Stanleyの[Iterated ES-HyperNEAT論文](https://groups.csail.mit.edu/EVO-DesignOpt/gecco2011Proceedings/proceedings/p1539.pdf)
の分割・分散・帯域抽出・ネットワーク完成手順を参照した独立実装です。
3D、再帰接続、複数領域、探索予算、成長は追加機能で、原論文の実験を再現したものではありません。
同梱Pureplesの実装はコピーしていません。

## 再帰実行と成長

RNNはすべての非入力神経（出力も含む）を前時刻の状態から同期更新します。
`h_next = h + leak * (tanh(sum) - h)`で、自己接続と循環接続を扱えます。
入力→隠れ→出力には少なくとも2回の`step`が必要です。空間上の上下方向を時間方向に
読み替えず、Pureplesの「RecurrentNetworkという容器に入れた実質FF」とは異なります。
バイアスが必要なら定数を供給する入力アンカーを追加します。

```python
# 利用側が成長時期を決め、新領域/新アンカーを含むSubstrateを用意する。
candidate = developer.develop(cppn, grown_substrate)
report = brain.grow(candidate.network)

# 外部の学習器が更新量を決める。受理された差分を返す。
accepted = brain.set_delta("hidden-id", "motor", delta=0.01)

saved = brain.snapshot()  # json.dumps(..., allow_nan=False)で保存可能
restored = GrowingRNN.from_snapshot(saved)
```

`grow`は追加型です。再探索で候補から消えた神経/接続も既存RNNには残し、既存神経の状態、
発現重み、学習差分、成熟途中の接続を維持します。既存IDの位置・役割・leak変更は拒否し、
削除/移動/統合は別の発達操作として明示的に指定します。領域IDと範囲を維持して新領域を追加する
使い方が基本です。
領域の細分化で候補IDが変わっても、古い神経は残って新しい神経が追加されます。
「変更された領域のみを探索する」増分探索キャッシュはありません。

新しい神経状態は0、新接続は既定8ステップでゲート0→1に成熟します。
既定の段階的成熟を使い、既存入力が同じなら追加直後の最初のステップの既存出力は変わりません。その後の挙動が不変で
ある保証はありません。神経・接続の予算超過や既存IDの意味の変更は、更新前に拒否します。

既定では各神経への再帰重みの絶対値和を0.98以下に制限します。成長時は既存重みと学習差分を
変更せず、新接続だけが残り予算を使います。`set_delta`もこの予算を守るようにクリップします。
予算を使い切った行では新再帰接続の重みが0になり得ます。
これは保守的な収縮制約で、強い持続記憶を制限するトレードオフがあります。
`recurrent_limit=None`で解除できますが、利用側で動的安定性を評価してください。

## 発達: 成長・成熟・刈り込み・再編

ここでの「発達」は構造変化を扱う上位APIです。成長判断、重要度推定、報酬、学習、課題に
依存する規則は含めません。操作の可否や旧技能を維持できたかの判断は利用側の責任です。

```python
from growth_hyperneat import Development, MigrationPolicy, Node

development = Development(brain)
change = development.grow(candidate.network)  # 追加のみ。取り消し可能
development.mature(ticks=2)                   # 接続成熟のみ。神経状態は進めない
change.rollback()                            # 変更前のRNN全状態へ戻す

# 神経削除は接続も連動して除去する。既定では入出力の変更を拒否する。
change = development.prune(
    nodes=("unneeded-hidden-id",),
    edges=(("source-id", "target-id"),),
    policy=MigrationPolicy(require_output_paths=True),
)

# ID・状態・重み・学習差分を保った座標変更。これだけなら計算は変わらない。
change = development.move({"hidden-id": (0.1, 0.2, 0.3)})

# CPPNを指定すると、移動した神経に接する既存辺だけを再発現する。
change = development.move({"hidden-id": (0.2, 0.2, 0.3)}, cppn=cppn)
```

`step`は神経状態と成熟度を1ステップ進めます。追加の`mature`は神経状態を進めず成熟時間だけを
加算するので、通常の時系列実行で毎回両方を呼ぶ必要はありません。
`Maturation(steps=8).advance(gates, ticks=2)`はRNNなしでも使えます。

削除は対象の状態・接続・学習差分だけを除去し、残った神経は維持します。重要度による自動削除は
行いません。`topology_health(network)`で入力から到達できない出力、入力→出力経路に属さない
隠れ神経を調べられます。これはゼロ重み・成熟前ゲートも含む潜在的な構造経路で、機能や性能の
保証ではありません。`require_output_paths=True`はその構造検査に基づく追加の拒否条件です。

座標変更時、重み維持ならRNNの計算は同じですが、CPPNを再発現した場合とは幾何との対応が
異なります。再発現は既存辺の重み更新のみで、ESで位置・接続を再探索する操作ではありません。
CPPN再発現の重みは即時変更され、成長のような滑らかな切り替えを保証しません。
`preserve_learning=False`は変更全体の学習差分を捨てる明示的なオプションです。

## 正確な構造変更と移行ポリシー

`development.reconfigure(candidate_network, policy)`または`brain.reconfigure(...)`は、候補の
構造を正確に採用します。候補から消えた神経/辺も削除するため、追加専用の`grow`とは異なります。
座標・leakの変更は可能ですが、同じIDの役割変更と2D↔3D変更は拒否します。

`MigrationPolicy`の既定は次のとおりです。

- 同じ神経IDの状態、同じ接続キーの実行時の発現重み・学習差分・成熟度を保持する。
- 新しい神経の状態は0、新しい接続は成長時と同じ段階的成熟を使う。
- 入力/出力の削除・追加・名前変更・順序変更は拒否する。必要な場合だけ
  `allow_port_changes=True`を指定し、利用側のセンサー/行動アダプターも更新する。
  `Development.grow`は明示的な追加操作なので、新しいポートの追加を許す。
- 候補の重みに更新したい場合は`weight_mode="reexpress"`、一部の辺だけなら
  `reexpress_edges=((source, target), ...)`を指定する。
- 再帰重みの制限は維持する。変更のない辺は保持し、変更・転送された辺だけが残り予算を使う。
  正規化が必要な場合は学習差分を維持したまま発現重みを調整し、`normalized_edges`に記録する。

別IDへの状態移行には`StateTransfer(target, ((source, coefficient), ...))`を使います。
係数は非負で合計1、同じ役割の神経に限ります。重みの転送には
`WeightTransfer(source, target, ((old_source, old_target, coefficient), ...))`を使えます。
明示的な転送がない削除・新規IDには、近さによる推測で記憶を移す処理はありません。
外部optimizer/適格度トレースはライブラリの所有物ではなく、利用側で同じ移行を行います。

## 候補の試験・採用・取り消し

```python
prepared = development.preview(candidate_network, MigrationPolicy())
trial = prepared.trial()              # 本体とは独立したRNN
# trialで旧課題や新課題を評価。必要なら外部学習器で再学習する。
change = development.commit(prepared) # 候補を採用。試験中の状態は持ち込まない
change.rollback()                    # 変更前の全RNN状態へ巻き戻す
```

候補は採用前に整合性・資源予算・有限値・再帰重み制限を検証します。失敗時は本体を変更しません。
候補作成後に本体の状態・学習重み・成熟度が変わった場合や、別の構造変更をした場合、古い候補の
採用を拒否します。`trial()`は毎回独立したRNNを作るため、その活動・学習は本体へ影響しません。
試験後の学習済み候補を持ち込みたい場合は、利用側が明示的に`brain.restore(trial.snapshot())`を
使います。この操作は既存の取り消しハンドルを無効化します。

`rollback`は最新の構造変更のハンドルだけが有効です。変更後の試験で進んだ神経状態・学習・
成熟度も巻き戻します。外部環境、報酬履歴、乱数、外部学習器の状態は巻き戻しません。
古いハンドルの取り消しや二度目の取り消しは拒否します。複数段階の履歴管理が必要なら
`snapshot`で利用側が管理します。スナップショットのschema 1は維持し、0.1.0の保存も読めます。

## 任意の圧縮: 神経統合

```python
prepared = development.preview_merge(
    ("hidden-a", "hidden-b"),
    Node("merged", (0, 0, 0), "hidden", leak=0.2),
    coefficients=(0.5, 0.5),
)
trial = prepared.trial()
# 利用側で性能・再学習を評価してから、development.commit(prepared)で採用する。
```

統合は成長とは別の近似圧縮です。隠れ神経のみを対象に、状態と入力側の接続行を加重平均し、
出力側の接続列を合計します（線形射影`W_new = R W_old P`）。自己接続・循環接続、学習差分も
同じ対応で射影します。非線形のtanhと異なるleakがあるため、元のRNNと同じ機能になる保証は
ありません。似た活動の神経を選ぶ、許容誤差を測る、再学習する判断は利用側に残します。

統合対象の接続ゲートが同一ならその成熟度を保持します。異なるゲートを統合する場合は、現在の
ゲートを重みに折り込んで統合辺を成熟済みにします。元の各辺の将来の成熟過程は失われます。
`development.merge(...)`は即時採用する便利関数ですが、実用途では`preview_merge`による試験を
推奨します。削除した元IDをESから再探索すれば再追加され得るため、抑制/再追加の方針も利用側で
指定してください。

## 探索・資源・学習の境界

- `initial_depth`、`max_depth`、分散・帯域の閾値で探索解像度を制御します。粗い標本化では
  細い帯域や対称パターンを見落とし得るため、ESは最適な配置/数を保証しません。
- `max_hidden_nodes`は整理前の候補数、`max_edges`も整理前の接続数の上限です。
  `max_queries`は1回の探索のユニークなCPPN問い合わせ数です。上限超過時は
  `ExpressionLimitError`を送出し、不完全なネットワークを黙って返しません。
- 定数パターンなどでは隠れ神経0、接続0も正常な結果です。直結を許す場合は
  `direct_links=True`を指定します。これは標準ES探索に加えたオプションです。
- `brain.complexity`は神経数・隠れ数・接続数を返します。これをfitnessへどう反映するかは
  利用側が決めます。自動的な複雑さペナルティや成長判断はありません。
- 報酬学習、予測学習、適格度トレース、optimizerは実装していません。外部学習器は
  `set_delta`を通じて重みを更新し、独自状態は別途保存します。これはAdaptive ES-HyperNEAT
  の学習規則一式を実装したものでもありません。
- `snapshot`はRNNだけを保存します。NEATの集団・種・乱数の保存はNEAT側の責任です。
  並列評価では独立したCPPN/RNNを用意し、同じ実行インスタンスを共有しないでください。

現在の`evo/`の固定配置はまだこのライブラリへ切り替えていません。身体への結び付けは別の
アダプターで扱い、旧結果との比較なしにESの性能向上を主張しないためです。
