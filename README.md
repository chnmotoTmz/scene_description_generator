# Scene Description Generator

シーン説明生成ツール - 動画のシーン説明を自動生成するPythonライブラリ

## 機能

- SRTファイルからシーン説明を自動生成
- 複数言語対応（日本語、英語、中国語、韓国語）
- シナリオファイルを利用した生成
- ウェブ検索機能（オプション）
- 使いやすいGUIインターフェース

## インストール

```bash
pip install -r requirements.txt
```

## 環境設定

1. `.env`ファイルを作成し、以下の環境変数を設定：

```env
GEMINI_API_KEY=your_gemini_api_key
SERPER_API_KEY=your_serper_api_key
```

## 使用方法

### GUIアプリケーションの起動

```bash
python scene_description_generator_gui.py
```

### プログラムからの使用

```python
from scene_description_generator import generate_captions_from_scenario, add_scene_descriptions_to_srt

# シナリオファイルを使用する場合
generate_captions_from_scenario(
    input_srt="input.srt",
    output_srt="output.srt",
    scenario_file="scenario.txt",
    scene_duration=10,
    language="ja"
)

# シナリオなしで生成する場合
add_scene_descriptions_to_srt(
    input_srt="input.srt",
    output_srt="output.srt",
    scene_duration=10,
    language="ja"
)
```

## ライセンス

MIT License 