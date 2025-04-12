#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GoPro AI編集アシスタント Gradioインターフェース
"""

import os
import sys
import json
import time
import logging
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

# Jinja2バージョンの問題を回避
try:
    import jinja2
    print(f"Current Jinja2 version: {jinja2.__version__}")
except ImportError:
    pass

# Gradioインポート
import gradio as gr

# 自作モジュールのインポート
from enhanced_scene_detection import EnhancedSceneDetector
try:
    from api_client import WhisperClient, GeminiClient
except ImportError:
    print("APIクライアントをインポートできませんでした。モックバージョンを使用します。")

# ロギング設定
user_home = os.path.expanduser("~")
logs_dir = os.path.join(user_home, "gradio_logs")
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_dir, f"gradio_app_{datetime.now():%Y%m%d_%H%M%S}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenCVの設定
cv2.setNumThreads(4)
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"  # デフォルト4096から増加

# PyTorchの可用性をチェック
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__} が利用可能です")
    if torch.cuda.is_available():
        logger.info(f"CUDA {torch.version.cuda} が利用可能です")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPUが検出されませんでした。CPU処理モードで実行します。")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorchが利用できません。CPU処理モードで実行します。")

class GradioVideoEnhanced:
    """
    Gradioを使用したGoPro AI編集アシスタント
    """
    
    def __init__(self):
        """アプリケーションの初期化"""
        logger.info("GradioVideoEnhanced初期化中...")
        self.scene_detector = EnhancedSceneDetector()
        
        # API Clientsはモック版を作成（依存関係の問題を避けるため）
        self.whisper_client = self.create_mock_whisper_client()
        self.gemini_client = self.create_mock_gemini_client()
        logger.info("モックAPIクライアント初期化完了")
            
        # 処理状態の管理
        self.is_processing = False
        self.current_progress = 0
        self.current_message = ""
        self.processing_results = {}
        
        logger.info("GradioVideoEnhanced初期化完了")
        
    def create_mock_whisper_client(self):
        """モックWhisperClientを作成"""
        class MockWhisperClient:
            def process_video(self, video_path, min_silence=1.0, start_time=0.0, max_duration=None):
                logger.info(f"モック音声認識: {video_path}")
                return {
                    "transcripts": [
                        {"start": 0, "end": 10, "text": "モック音声認識の例文です。"},
                        {"start": 10, "end": 20, "text": "これはデモ用の字幕です。"}
                    ],
                    "scene_boundaries": [0, 10, 20, 30]
                }
        return MockWhisperClient()
        
    def create_mock_gemini_client(self):
        """モックGeminiClientを作成"""
        class MockGeminiClient:
            def analyze_scene(self, image_path):
                logger.info(f"モックシーン分析: {image_path}")
                return {
                    "scene_type": "屋外",
                    "location": "公園",
                    "objects": ["木", "空", "草"],
                    "keywords": ["自然", "晴れ", "公園"]
                }
        return MockGeminiClient()

    def update_progress(self, progress: float, message: str):
        """進捗状況を更新する"""
        self.current_progress = progress
        self.current_message = message
        logger.info(f"進捗: {progress:.1f}% - {message}")
        return progress, message
    
    def select_video(self, video_file):
        """動画ファイルが選択された時の処理"""
        if not video_file:
            return "動画ファイルが選択されていません。", None
            
        logger.info(f"動画ファイルが選択されました: {video_file.name}")
        return f"選択された動画: {os.path.basename(video_file.name)}", video_file
    
    def process_video(self, video_file, min_scene_duration, hist_threshold, pixel_threshold):
        """動画処理のメイン関数"""
        if not video_file:
            return "エラー: 動画ファイルが選択されていません。", [], None, None
            
        self.is_processing = True
        results = {
            "scenes": [],
            "thumbnails": [],
            "transcripts": [],
            "metadata": {}
        }
        
        try:
            video_path = video_file.name
            logger.info(f"処理開始: {video_path}")
            
            # 1. シーン検出
            self.update_progress(5, "シーン分析の準備中...")
            
            # シーン検出器の設定
            self.scene_detector.set_progress_callback(self.update_progress)
            self.scene_detector.min_scene_duration = min_scene_duration
            self.scene_detector.hist_threshold = hist_threshold
            self.scene_detector.pixel_threshold = pixel_threshold
            
            # 出力ディレクトリを作成
            output_dir = os.path.join(user_home, "scene_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # シーン検出実行
            self.update_progress(10, "シーン検出を実行中...")
            scenes_data = self.scene_detector.detect_scenes(
                video_path=video_path,
                output_dir=output_dir
            )
            
            # 検出結果の表示
            num_scenes = len(scenes_data["scenes"]) - 1 if scenes_data["scenes"] else 0
            logger.info(f"シーン検出結果: {num_scenes}個のシーンを検出")
            
            # 2. 音声認識（WhisperClientがある場合）
            transcripts = []
            if self.whisper_client:
                self.update_progress(50, "音声認識を実行中...")
                try:
                    whisper_result = self.whisper_client.process_video(video_path)
                    transcripts = whisper_result.get("transcripts", [])
                    logger.info(f"音声認識完了: {len(transcripts)}個のセグメントを検出")
                except Exception as e:
                    logger.error(f"音声認識エラー: {str(e)}")
            else:
                logger.warning("WhisperClientが利用できないため音声認識をスキップします")
            
            # 3. シーンの分析と結果の準備
            self.update_progress(80, "シーンデータを準備中...")
            
            # サムネイル画像のパスを収集
            thumbnails = []
            for i, (start, end) in enumerate(zip(scenes_data["scenes"][:-1], scenes_data["scenes"][1:])):
                thumbnail_path = os.path.join(output_dir, f"scene_{i:04d}.jpg")
                if os.path.exists(thumbnail_path):
                    thumbnails.append(thumbnail_path)
            
            # 結果の更新
            results["scenes"] = scenes_data["scenes"]
            results["thumbnails"] = thumbnails
            results["transcripts"] = transcripts
            results["metadata"] = {
                "filename": os.path.basename(video_path),
                "duration": scenes_data.get("duration", 0),
                "num_scenes": num_scenes,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.update_progress(100, "処理完了")
            self.processing_results = results
            
            # 結果のサマリーを生成
            summary = f"処理完了: {num_scenes}個のシーンを検出しました"
            
            # サムネイルのギャラリーとシーン情報を表示用に整形
            thumbnails_html = self.generate_thumbnails_html(results["thumbnails"])
            scenes_info = self.generate_scenes_info(
                results["scenes"], 
                results["thumbnails"],
                results["transcripts"]
            )
            
            return summary, scenes_info, thumbnails_html, json.dumps(results, indent=2)
            
        except Exception as e:
            logger.error(f"処理エラー: {str(e)}", exc_info=True)
            self.update_progress(0, f"エラー: {str(e)}")
            return f"エラー: {str(e)}", [], None, None
        finally:
            self.is_processing = False
    
    def generate_thumbnails_html(self, thumbnails):
        """サムネイル画像をHTMLギャラリー形式で表示"""
        if not thumbnails:
            return "<p>サムネイルがありません</p>"
        
        html = '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
        for i, thumb in enumerate(thumbnails):
            html += f'<div style="text-align: center;"><img src="file={thumb}" style="width: 160px; height: 90px; object-fit: cover;"><p>シーン {i+1}</p></div>'
        html += '</div>'
        return html
    
    def generate_scenes_info(self, scenes, thumbnails, transcripts):
        """シーン情報を表形式で整理"""
        if not scenes or len(scenes) <= 1:
            return []
        
        # 各シーンの情報を収集
        scenes_info = []
        for i, (start, end) in enumerate(zip(scenes[:-1], scenes[1:])):
            # このシーンに関連するトランスクリプトを収集
            scene_transcript = ""
            for t in transcripts:
                t_start = t.get("start", 0)
                t_end = t.get("end", 0)
                # トランスクリプトがシーンの範囲内にあるか確認
                if t_end >= start and t_start <= end:
                    scene_transcript += t.get("text", "") + " "
            
            scene_data = {
                "scene_number": i + 1,
                "start_time": self.format_time(start),
                "end_time": self.format_time(end),
                "duration": self.format_time(end - start),
                "thumbnail": thumbnails[i] if i < len(thumbnails) else None,
                "transcript": scene_transcript.strip()
            }
            scenes_info.append(scene_data)
        
        return scenes_info
    
    def format_time(self, seconds):
        """秒数を時:分:秒.ミリ秒 形式にフォーマット"""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int((seconds % 1) * 1000):03d}"

def create_gradio_interface():
    """Gradioのインターフェースを作成"""
    app = GradioVideoEnhanced()
    
    with gr.Blocks(title="動画解析・編集アシスタント") as interface:
        gr.Markdown("# 動画解析・編集アシスタント")
        gr.Markdown("強化されたシーン検出機能を備えた動画解析・編集支援ツール")
        
        with gr.Row():
            with gr.Column(scale=1):
                # ファイル選択と処理オプション
                video_input = gr.File(label="動画ファイルを選択")
                video_info = gr.Textbox(label="選択ファイル情報", interactive=False)
                
                # パラメータ設定
                with gr.Accordion("詳細設定", open=False):
                    min_scene_duration = gr.Slider(
                        minimum=0.5, maximum=10, value=3.0, step=0.5,
                        label="最小シーン長さ（秒）"
                    )
                    hist_threshold = gr.Slider(
                        minimum=0.5, maximum=0.95, value=0.9, step=0.05,
                        label="ヒストグラム相関閾値"
                    )
                    pixel_threshold = gr.Slider(
                        minimum=5, maximum=50, value=20, step=5,
                        label="ピクセル差分閾値"
                    )
                
                # 処理ボタン
                process_btn = gr.Button("処理開始", variant="primary")
                
                # 進捗表示
                progress = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="進捗状況", interactive=False
                )
                progress_msg = gr.Textbox(label="状態", interactive=False)
            
            with gr.Column(scale=2):
                # 処理結果表示
                result_tabs = gr.Tabs()
                with result_tabs:
                    with gr.TabItem("シーン分析"):
                        summary = gr.Textbox(label="処理結果", interactive=False)
                        scenes_info = gr.Markdown(label="シーン情報")
                    
                    with gr.TabItem("サムネイル"):
                        thumbnails_html = gr.HTML(label="シーンサムネイル")
                    
                    with gr.TabItem("JSON結果"):
                        json_output = gr.Code(language="json", label="JSONデータ")
        
        # イベントハンドラー
        video_input.change(
            fn=app.select_video,
            inputs=[video_input],
            outputs=[video_info, video_input]
        )
        
        process_btn.click(
            fn=app.process_video,
            inputs=[video_input, min_scene_duration, hist_threshold, pixel_threshold],
            outputs=[summary, scenes_info, thumbnails_html, json_output]
        )
    
    return interface

if __name__ == "__main__":
    # 環境情報をログに記録
    logger.info("=== システム情報 ===")
    logger.info(f"Python バージョン: {sys.version}")
    if TORCH_AVAILABLE:
        logger.info(f"PyTorch バージョン: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"CUDA バージョン: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"利用可能GPU数: {torch.cuda.device_count()}")
    logger.info("=== 環境変数 ===")
    logger.info(f"OPENCV_FFMPEG_READ_ATTEMPTS: {os.environ.get('OPENCV_FFMPEG_READ_ATTEMPTS', 'not set')}")
    
    # Gradioアプリを起動
    interface = create_gradio_interface()
    interface.launch(share=False)