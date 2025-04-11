# 動画解析・編集アシスタント

強化されたシーン検出機能とプレビュー付きGUIを備えた動画解析・編集支援ツールです。

## 機能概要

- **強化されたシーン検出**: 音声と映像の両方を使用した高精度なシーン分割
- **プレビュー付きGUI**: サムネイルギャラリーとリアルタイムプレビュー機能
- **音声認識**: 動画内の音声を自動的に文字起こし
- **シーン分析**: 各シーンの内容を自動的に分析
- **編集支援**: シーン単位での編集作業をサポート

## インストール方法

必要なパッケージをインストールします：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使い方

```bash
python video_enhanced_refactored.py
```

アプリケーションが起動したら、以下の手順で操作します：

1. 「動画を選択」ボタンをクリックして処理したい動画ファイルを選択
2. 「フォルダを選択」ボタンでフォルダ内の動画を一括選択することも可能
3. 「処理開始」ボタンをクリックして解析を開始
4. 処理が完了したら「詳細表示」ボタンで結果を確認

### シーンプレビュー機能

- サムネイルギャラリーから任意のシーンをクリック
- 「再生」ボタンでシーンのプレビューを再生
- 「前のシーン」「次のシーン」ボタンでシーン間を移動

## 主要コンポーネント

### EnhancedSceneDetector

音声と映像の両方を使用した高精度なシーン検出を行うクラスです。

```python
from enhanced_scene_detection import EnhancedSceneDetector

# 初期化
detector = EnhancedSceneDetector(min_scene_duration=3.0)

# 進捗コールバックを設定
detector.set_progress_callback(lambda progress, message: print(f"{progress}%: {message}"))

# シーン検出を実行
scenes = detector.detect_scenes("input_video.mp4", "output_directory")
```

主な機能：
- 適応型閾値によるシーン境界検出
- 短いシーンの自動フィルタリング
- 音声と映像のシーン境界の統合

### ScenePreviewGUI

サムネイルギャラリーとプレビュー機能を提供するGUIコンポーネントです。

```python
from scene_preview_gui import ScenePreviewGUI

# Tkinterウィンドウ内に配置
preview_gui = ScenePreviewGUI(parent_widget, vlc_instance, player)

# シーンデータを設定
preview_gui.set_scenes("video_path.mp4", scenes_data)

# コールバック関数を設定
preview_gui.set_on_scene_selected_callback(on_scene_selected)
preview_gui.set_on_scene_play_callback(on_scene_play)
```

主な機能：
- サムネイルギャラリー表示
- シーンプレビュー再生
- シーン間のナビゲーション

## 設定オプション

### シーン検出の設定

`EnhancedSceneDetector`クラスの初期化時に以下のパラメータを調整できます：

- `min_scene_duration`: 最小シーン長さ（秒）
- `hist_threshold`: ヒストグラム相関閾値（低いほど敏感）
- `pixel_threshold`: ピクセル差分閾値（高いほど敏感）
- `sample_rate`: サンプリングレート（フレーム数）

### GUI表示の設定

`ScenePreviewGUI`クラスはTkinterベースのGUIを提供します。親ウィジェット内に配置して使用します。

## 技術的詳細

### シーン検出アルゴリズム

1. **第1パス**: 動画全体をサンプリングして差分データを収集し、適応型閾値を算出
2. **第2パス**: 算出された閾値を使用してシーン境界を検出
3. **フィルタリング**: 短すぎるシーンを統合
4. **キーフレーム抽出**: 各シーンの代表的なフレームを抽出

### 音声認識と統合

1. WhisperClientを使用して音声認識を実行
2. 無音区間を検出してシーン境界候補とする
3. 映像ベースのシーン境界と統合して最終的なシーン分割を決定

## 依存ライブラリ

- OpenCV: 映像処理
- NumPy: 数値計算
- PyTorch: 機械学習（音声認識）
- Tkinter: GUI
- VLC: メディア再生
- PIL: 画像処理

## トラブルシューティング

### VLCが見つからない場合

VLCがインストールされていない場合、プレビュー機能は制限されますが、他の機能は正常に動作します。

### GPU処理が利用できない場合

PyTorchがGPUを検出できない場合、自動的にCPU処理モードで実行されます。処理速度は低下しますが、機能は維持されます。

### エラーログの確認

エラーが発生した場合は、`logs`ディレクトリ内のログファイルを確認してください。詳細なエラー情報が記録されています。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
