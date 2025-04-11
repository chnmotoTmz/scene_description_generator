import os
import sys
import json
import time
import shutil
import logging
import threading
import subprocess
import glob
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import functools
import psutil

# サードパーティライブラリ
import cv2
import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorchが利用できません。CPU処理モードで実行します。")

# GUI関連のインポート
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

# 自作モジュールのインポート
from enhanced_scene_detection import EnhancedSceneDetector
from scene_preview_gui import ScenePreviewGUI
from api_client import WhisperClient, GeminiClient

# OpenCVのFFmpeg読み取り試行回数を増加（マルチストリーム処理の安定性向上）
cv2.setNumThreads(4)
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"  # デフォルト4096から増加

# ロギング設定
# logsディレクトリが存在しない場合は作成
os.makedirs("logs", exist_ok=True)

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # DEBUGレベルに設定

# ファイルハンドラの設定
log_file = f"logs/enhanced_{datetime.now():%Y%m%d_%H%M%S}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
))
logger.addHandler(file_handler)

# コンソールハンドラの設定
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
logger.addHandler(console_handler)

# システム情報のログ出力
logger.info("=== システム情報 ===")
logger.info(f"Python バージョン: {sys.version}")
logger.info(f"OS: {os.name} - {sys.platform}")
if TORCH_AVAILABLE:
    logger.info(f"PyTorch バージョン: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA バージョン: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"利用可能GPU数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  - 総メモリ: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  - CUDA Capability: {props.major}.{props.minor}")
else:
    logger.info("PyTorch は利用できません")

logger.info("=== 環境変数 ===")
logger.info(f"OPENCV_FFMPEG_READ_ATTEMPTS: {os.getenv('OPENCV_FFMPEG_READ_ATTEMPTS', 'Not set')}")
logger.info(f"GEMINI_API_KEY設定状態: {'設定済み' if os.getenv('GEMINI_API_KEY') else '未設定'}")

# エラーハンドリング用のデコレータ
def log_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # 関数名と引数をログに記録
            logger.info(f"関数開始: {func.__name__}")
            logger.debug(f"引数: args={args}, kwargs={kwargs}")
            
            # メモリ使用状況を記録
            log_memory_usage()
            
            # 関数実行
            result = func(*args, **kwargs)
            
            # 成功時のログ
            logger.info(f"関数成功: {func.__name__}")
            if result is not None:
                logger.debug(f"戻り値の型: {type(result)}")
                if isinstance(result, (list, dict)):
                    logger.debug(f"戻り値のサイズ: {len(result)}")
            
            return result
        except Exception as e:
            # エラー情報を詳細に記録
            logger.error(f"関数 {func.__name__} でエラー発生")
            logger.error(f"エラーの種類: {type(e).__name__}")
            logger.error(f"エラーメッセージ: {str(e)}")
            logger.error("スタックトレース:", exc_info=True)
            
            # メモリ使用状況を記録
            log_memory_usage()
            
            # 環境情報を記録
            logger.error("=== エラー発生時の環境情報 ===")
            logger.error(f"Python バージョン: {sys.version}")
            logger.error(f"PyTorch バージョン: {torch.__version__}")
            if TORCH_AVAILABLE and torch.cuda.is_available():
                logger.error(f"CUDA バージョン: {torch.version.cuda}")
                logger.error(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.error(f"利用可能GPU数: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.error(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.error(f"  - 総メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                    logger.error(f"  - CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            # 詳細なスタックトレース
            import traceback
            logger.error("詳細なスタックトレース:\n" + "".join(traceback.format_tb(e.__traceback__)))
            raise
    return wrapper

# メモリ使用状況をログ出力する関数
def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.debug("=== メモリ使用状況 ===")
    logger.debug(f"  - RSS (物理メモリ): {memory_info.rss / 1024**2:.2f} MB")
    logger.debug(f"  - VMS (仮想メモリ): {memory_info.vms / 1024**2:.2f} MB")
    logger.debug(f"  - ページフォールト: {memory_info.pfaults}")
    logger.debug(f"  - ページインフォールト: {memory_info.pageins}")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        logger.debug("=== CUDAメモリ使用状況 ===")
        logger.debug(f"  - 確保済みメモリ: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        logger.debug(f"  - キャッシュメモリ: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        logger.debug(f"  - 最大確保メモリ: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        logger.debug(f"  - 最大キャッシュメモリ: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
        
        # GPUメモリの詳細情報
        for i in range(torch.cuda.device_count()):
            logger.debug(f"=== GPU {i} メモリ詳細 ===")
            logger.debug(f"  - 総メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**2:.2f} MB")
            logger.debug(f"  - 空きメモリ: {torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            logger.debug(f"  - メモリ使用率: {(torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory) * 100:.2f}%")

class VideoNode:
    """ビデオノードクラス - シーンの情報を保持"""
    
    def __init__(self, time_in: float, time_out: float):
        self.time_in = time_in
        self.time_out = time_out
        self.transcript = ""
        self.description = ""
        self.keyframe_path = ""
        self.preview_path = ""
        self.context_analysis = {}
        self.editing_suggestions = {}
        self.tags = []
        self.highlight = False
        self.custom_notes = ""
    
    def to_dict(self) -> dict:
        """ノードを辞書形式に変換"""
        return {
            "time_in": self.time_in,
            "time_out": self.time_out,
            "transcript": self.transcript,
            "description": self.description,
            "keyframe_path": self.keyframe_path,
            "preview_path": self.preview_path,
            "context_analysis": self.context_analysis,
            "editing_suggestions": self.editing_suggestions,
            "tags": self.tags,
            "highlight": self.highlight,
            "custom_notes": self.custom_notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VideoNode':
        """辞書からノードを生成"""
        node = cls(data.get("time_in", 0), data.get("time_out", 0))
        node.transcript = data.get("transcript", "")
        node.description = data.get("description", "")
        node.keyframe_path = data.get("keyframe_path", "")
        node.preview_path = data.get("preview_path", "")
        node.context_analysis = data.get("context_analysis", {})
        node.editing_suggestions = data.get("editing_suggestions", {})
        node.tags = data.get("tags", [])
        node.highlight = data.get("highlight", False)
        node.custom_notes = data.get("custom_notes", "")
        return node

class VideoEnhanced:
    """メインアプリケーションクラス"""
    
    def __init__(self):
        """初期化"""
        # メインウィンドウの設定
        self.root = tk.Tk()
        self.root.title("動画解析・編集アシスタント")
        self.root.geometry("1200x800")
        
        # スタイル設定
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        
        # メインフレーム
        self.main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ヘッダーフレーム
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.header_frame, text="動画解析・編集アシスタント", style='Title.TLabel').pack(side=tk.LEFT, padx=5)
        
        # ファイル選択ボタン
        self.button_frame = ttk.Frame(self.header_frame)
        self.button_frame.pack(side=tk.RIGHT)
        
        self.open_button = ttk.Button(self.button_frame, text="📂 動画を選択", command=self.select_files)
        self.open_button.pack(side=tk.LEFT, padx=5)
        
        self.open_folder_button = ttk.Button(self.button_frame, text="📁 フォルダを選択", command=self.select_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=5)
        
        # 左右のメインペイン
        self.paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # 左ペイン（ファイルリスト）
        self.list_frame = ttk.Frame(self.paned, padding="0 0 0 0")
        
        # リストのラベル
        ttk.Label(self.list_frame, text="処理対象ファイル", style='Header.TLabel').pack(anchor="w", padx=5, pady=(0, 5))
        
        # ファイルリスト
        columns = ("ファイル名", "状態", "シーン数")
        self.file_list = ttk.Treeview(self.list_frame, columns=columns, show="headings", height=15)
        self.file_list.column("ファイル名", width=500)  # 幅を広げる
        self.file_list.column("状態", width=100)
        self.file_list.column("シーン数", width=70)
        
        for col in columns:
            self.file_list.heading(col, text=col)
        
        self.file_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ファイルリストスクロールバー
        list_scrollbar = ttk.Scrollbar(self.list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.configure(yscrollcommand=list_scrollbar.set)
        
        # ファイルリストの選択イベントをバインド
        self.file_list.bind("<<TreeviewSelect>>", self.update_preview)
        
        # アクションボタンフレーム
        self.action_frame = ttk.Frame(self.list_frame)
        self.action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_button = ttk.Button(self.action_frame, text="▶️ 処理開始", command=self.confirm_start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.action_frame, text="⏹️ 停止", command=self.cancel_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.view_button = ttk.Button(self.action_frame, text="👁️ 詳細表示", command=self.show_file_details, state=tk.DISABLED)
        self.view_button.pack(side=tk.RIGHT, padx=5)
        
        # 右ペイン（詳細情報）
        self.details_frame = ttk.Frame(self.paned)
        
        # ScenePreviewGUIの初期化
        self.preview_gui = None
        
        # パネル分割の設定
        self.paned.add(self.list_frame, weight=1)
        self.paned.add(self.details_frame, weight=2)
        
        # ステータスバー
        self.status_frame = ttk.Frame(self.main_frame, relief=tk.SUNKEN, padding=(5, 2))
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="準備完了")
        self.status_label.pack(side=tk.LEFT)
        
        # 進捗バー（ファイル全体）
        self.progress_frame = ttk.LabelFrame(self.status_frame, text="進捗", padding=(5, 2))
        self.progress_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        self.total_progress_var = tk.IntVar(value=0)
        self.total_progress = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                              length=200, mode='determinate', 
                                              variable=self.total_progress_var)
        self.total_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.file_progress_var = tk.IntVar(value=0)
        self.file_progress = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, 
                                             length=100, mode='determinate', 
                                             variable=self.file_progress_var)
        self.file_progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 処理状態の初期化
        self.processing = False
        self.processing_thread = None
        self.current_file = None
        self.selected_files = []
        self.processed_files = set()
        self.current_preview_path = None
        self.current_duration = 0
        self.is_playing = False
        self.update_timer = None
        
        # VLCプレーヤーの初期化
        self.vlc_instance = None
        self.player = None
        try:
            # 実行時にVLCをインポート
            vlc_module = __import__('vlc')
            self.vlc_instance = vlc_module.Instance()
            self.player = self.vlc_instance.media_player_new()
            logger.info("VLCプレーヤーを初期化しました")
            
            # ScenePreviewGUIの初期化
            self.preview_gui = ScenePreviewGUI(self.details_frame, self.vlc_instance, self.player)
            
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"VLCライブラリが見つかりません: {str(e)}")
            # VLCなしでもGUIを初期化
            self.preview_gui = ScenePreviewGUI(self.details_frame)
            
        except Exception as e:
            logger.error(f"VLCプレーヤー初期化エラー: {str(e)}")
            # エラー時もGUIを初期化
            self.preview_gui = ScenePreviewGUI(self.details_frame)
        
        # Gemini Clientの初期化
        try:
            self.gemini_client = GeminiClient()
            logger.info("Gemini Clientを初期化しました")
        except Exception as e:
            logger.error(f"Gemini Client初期化エラー: {str(e)}")
            # フォールバック：シンプルな機能を持つダミーインスタンスを作成
            self.gemini_client = None
        
        # WhisperClientの初期化
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.whisper_client = WhisperClient(model_size="large", device="cuda")
            else:
                self.whisper_client = WhisperClient(model_size="base", device="cpu")
            logger.info("WhisperClientを初期化しました")
        except Exception as e:
            logger.error(f"WhisperClient初期化エラー: {str(e)}")
            self.whisper_client = None
        
        # シーン検出器の初期化
        self.scene_detector = EnhancedSceneDetector(min_scene_duration=3.0)
        
        # 終了時の処理を設定
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.update()
    
    def select_files(self):
        """動画ファイルを選択"""
        file_paths = filedialog.askopenfilenames(
            title="処理する動画ファイルを選択",
            filetypes=[
                ("動画ファイル", "*.mp4 *.avi *.mov *.mkv"),
                ("すべてのファイル", "*.*")
            ]
        )
        
        if not file_paths:
            return

        self.selected_files = sorted(file_paths)  # ファイル名でソート
        self.update_file_list()
        
        # 処理ボタンを有効化
        if self.selected_files:
            self.process_button.config(state="normal")
            self.update_status("準備完了", f"{len(self.selected_files)}個の動画ファイルが選択されました")
    
    def select_folder(self):
        """フォルダを選択し、その中の動画ファイルを一括で選択"""
        folder_path = filedialog.askdirectory(
            title="処理するフォルダを選択"
        )
        if not folder_path:
            return

        # 動画ファイルの拡張子
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        
        # フォルダ内の動画ファイルを検索
        video_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            messagebox.showwarning(
                "警告",
                "選択されたフォルダ内に動画ファイルが見つかりませんでした。"
            )
            return

        self.selected_files = sorted(video_files)  # ファイル名でソート
        self.update_file_list()
        
        # 処理ボタンを有効化
        if self.selected_files:
            self.process_button.config(state="normal")
            self.update_status("準備完了", f"{len(self.selected_files)}個の動画ファイルが見つかりました")
    
    def update_file_list(self):
        """選択されたファイルをツリービューに表示"""
        # ツリービューをクリア
        for item in self.file_list.get_children():
            self.file_list.delete(item)

        for video_path in self.selected_files:
            # 処理状態を確認
            if video_path in self.processed_files:
                status = "処理済"
                # ノードファイルからシーン数を取得
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
                nodes_file = os.path.join(output_dir, "nodes.json")
                
                if os.path.exists(nodes_file):
                    try:
                        with open(nodes_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        scenes = data.get("scenes", data.get("nodes", []))
                        scene_count = len(scenes)
                    except:
                        scene_count = "?"
                else:
                    scene_count = "?"
            else:
                status = "未処理"
                scene_count = "-"
            
            # ファイル名のみを表示
            file_name = os.path.basename(video_path)
            
            self.file_list.insert("", "end", values=(file_name, status, scene_count))
    
    def update_status(self, status, message=None):
        """ステータスバーを更新"""
        self.status_label.config(text=status)
        if message:
            logger.info(message)
    
    def confirm_start_processing(self):
        """処理開始の確認"""
        if not self.selected_files:
            messagebox.showwarning("警告", "処理するファイルが選択されていません。")
            return
        
        # 未処理のファイル数を確認
        unprocessed = [f for f in self.selected_files if f not in self.processed_files]
        
        if not unprocessed:
            messagebox.showinfo("情報", "すべてのファイルが処理済みです。")
            return
        
        # 確認ダイアログ
        message = f"{len(unprocessed)}個のファイルを処理します。続行しますか？"
        if messagebox.askyesno("確認", message):
            self.start_processing()
    
    def start_processing(self):
        """処理を開始"""
        if self.processing:
            return
        
        self.processing = True
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.open_button.config(state="disabled")
        self.open_folder_button.config(state="disabled")
        
        # 進捗バーをリセット
        self.total_progress_var.set(0)
        self.file_progress_var.set(0)
        
        # 別スレッドで処理を実行
        self.processing_thread = threading.Thread(target=self.process_files_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 定期的にスレッドの状態を確認
        self.root.after(100, self.check_thread_status)
    
    def cancel_processing(self):
        """処理をキャンセル"""
        if not self.processing:
            return
        
        if messagebox.askyesno("確認", "処理を中止しますか？"):
            self.processing = False
            self.update_status("キャンセル中", "処理をキャンセルしています...")
            self.stop_button.config(state="disabled")
    
    def check_thread_status(self):
        """処理スレッドの状態を確認"""
        if self.processing and self.processing_thread:
            if self.processing_thread.is_alive():
                # まだ実行中なら再度チェック
                self.root.after(100, self.check_thread_status)
            else:
                # 処理が完了したらUIを更新
                self.processing = False
                self.open_button.config(state="normal")
                self.open_folder_button.config(state="normal")
                self.process_button.config(state="normal")
                self.stop_button.config(state="disabled")
    
    def process_files_thread(self):
        """複数ファイルの処理を実行"""
        try:
            total_files = len(self.selected_files)
            processed_count = 0
            
            for i, video_path in enumerate(self.selected_files):
                if not self.processing:  # キャンセルチェック
                    break
                
                # 処理済みファイルをスキップ
                if video_path in self.processed_files:
                    logger.info(f"スキップ: {os.path.basename(video_path)} (処理済)")
                    processed_count += 1
                    continue
                
                # 進捗更新
                total_progress = (i / total_files) * 100
                self.total_progress_var.set(total_progress)
                self.file_progress_var.set(0)  # 新しいファイルの開始
                self.update_status(
                    f"処理中 ({i+1}/{total_files})",
                    f"{os.path.basename(video_path)}を処理中..."
                )
                
                try:
                    # 個別ファイルの処理
                    self.process_single_video(video_path)
                    processed_count += 1
                    self.processed_files.add(video_path)
                    self.file_progress_var.set(100)  # ファイル完了
                    
                    # ファイルリストの更新
                    self.root.after(0, self.update_file_list)
                    
                except Exception as e:
                    logger.error(f"ファイル処理中にエラー: {str(e)}")
                    continue
            
            # 完了
            if self.processing:
                self.total_progress_var.set(100)
                self.file_progress_var.set(100)
                self.update_status(
                    "完了",
                    f"処理完了: {processed_count}個のファイルを処理しました"
                )
            else:
                self.update_status(
                    "キャンセル",
                    f"処理をキャンセルしました（{processed_count}個処理済）"
                )
                
        except Exception as e:
            logger.error(f"バッチ処理中にエラー: {str(e)}")
            self.update_status("エラー", f"処理中にエラーが発生: {str(e)}")
        
        finally:
            self.processing = False
    
    @log_exceptions
    def process_single_video(self, video_path: str):
        """単一の動画を処理する"""
        logger.info(f"動画処理開始: {video_path}")
        log_memory_usage()
        
        try:
            # 出力ディレクトリの準備
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
            keyframes_dir = os.path.join(output_dir, "keyframes")
            preview_dir = os.path.join(output_dir, "previews")
            
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(keyframes_dir, exist_ok=True)
            os.makedirs(preview_dir, exist_ok=True)
            
            # 進捗コールバック関数
            def update_progress(progress, message):
                self.file_progress_var.set(progress)
                self.update_status(f"処理中: {message}", f"{os.path.basename(video_path)}: {message}")
            
            # シーン検出器に進捗コールバックを設定
            self.scene_detector.set_progress_callback(update_progress)
            
            # 1. 音声認識と無音区間検出
            update_progress(5, "音声認識を開始")
            transcripts = self._extract_transcripts(video_path)
            audio_boundaries = transcripts.get("scene_boundaries", [])
            
            # 2. 映像ベースのシーン検出
            update_progress(30, "シーン検出を開始")
            scene_data = self.scene_detector.detect_scenes(video_path, output_dir)
            
            # 3. 音声と映像のシーン境界を統合
            update_progress(60, "シーン境界を統合")
            merged_boundaries = self.scene_detector.merge_with_audio_boundaries(audio_boundaries)
            
            # 4. シーンデータを生成
            update_progress(70, "シーンデータを生成")
            all_boundaries = merged_boundaries
            
            # 5. ビデオノードを生成
            update_progress(80, "ビデオノードを生成")
            nodes = self._create_video_nodes(
                video_path, all_boundaries, scene_data, transcripts, 
                keyframes_dir, preview_dir
            )
            
            # 6. GoPro メタデータを抽出
            update_progress(90, "メタデータを抽出")
            gopro_metadata = self.extract_gopro_metadata(video_path)
            
            # 7. 動画全体のサマリー情報を生成
            update_progress(95, "サマリー情報を生成")
            summary = self._generate_video_summary(video_path, nodes, transcripts, gopro_metadata)
            
            # 8. 結果を保存
            update_progress(98, "結果を保存")
            self.save_results(video_path, nodes, True, summary)
            
            update_progress(100, "処理完了")
            logger.info(f"動画処理完了: {video_path}")
            
            return nodes
            
        finally:
            log_memory_usage()
            logger.info(f"動画処理完了: {video_path}")
    
    @log_exceptions
    def _extract_transcripts(self, video_path: str) -> dict:
        """音声認識を実行する"""
        logger.info(f"音声認識開始: {video_path}")
        log_memory_usage()
        
        try:
            if self.whisper_client:
                # WhisperClientを使用して音声認識
                result = self.whisper_client.process_video(video_path, min_silence=1.0)
                logger.info(f"音声認識結果: {len(result.get('transcripts', []))}個のセグメント")
                return result
            else:
                logger.warning("WhisperClientが初期化されていないため、音声認識をスキップします")
                return {"transcripts": [], "scene_boundaries": []}
            
        finally:
            log_memory_usage()
            logger.info(f"音声認識完了: {video_path}")
    
    @log_exceptions
    def _create_video_nodes(self, video_path: str, all_boundaries: list, scene_data: list, transcripts: dict, 
                           keyframes_dir: str, preview_dir: str) -> List[VideoNode]:
        """ビデオノードを生成する"""
        logger.info("ビデオノード生成開始")
        logger.debug(f"入力データ: boundaries={all_boundaries}, scene_data={scene_data}")
        
        try:
            nodes = []
            
            # 境界がない場合は動画全体を1つのノードとして扱う
            if not all_boundaries or len(all_boundaries) < 2:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_duration = total_frames / fps
                cap.release()
                
                node = VideoNode(0, total_duration)
                nodes.append(node)
                return nodes
            
            # 各シーン境界からノードを生成
            for i in range(len(all_boundaries) - 1):
                start_time = all_boundaries[i]
                end_time = all_boundaries[i + 1]
                
                # 新しいノードを作成
                node = VideoNode(start_time, end_time)
                
                # シーンデータから一致するものを探す
                matching_scene = None
                for scene in scene_data:
                    scene_start = scene.get("start_time", 0)
                    scene_end = scene.get("end_time", 0)
                    
                    # 時間範囲が重なるシーンを探す
                    if (start_time <= scene_end and end_time >= scene_start):
                        matching_scene = scene
                        break
                
                # トランスクリプトを抽出
                scene_transcripts = []
                for t in transcripts.get("transcripts", []):
                    t_start = t.get("start", 0)
                    t_end = t.get("end", 0)
                    
                    # 時間範囲が重なるトランスクリプトを抽出
                    if (t_start <= end_time and t_end >= start_time):
                        scene_transcripts.append(t)
                
                # トランスクリプトテキストを結合
                transcript_text = " ".join([t.get("text", "") for t in scene_transcripts])
                node.transcript = transcript_text
                
                # シーンデータを適用
                self._apply_scene_data_to_node(node, matching_scene, scene_transcripts, video_path)
                
                # キーフレームがない場合は生成
                if not node.keyframe_path:
                    keyframe_time = (start_time + end_time) / 2
                    keyframe_path = os.path.join(keyframes_dir, f"keyframe_{i:04d}.jpg")
                    
                    if self.extract_keyframe(video_path, keyframe_time, keyframe_path):
                        node.keyframe_path = os.path.relpath(keyframe_path, os.path.dirname(video_path))
                
                # プレビュー動画を生成
                preview_path = os.path.join(preview_dir, f"preview_{i:04d}.mp4")
                if self.generate_preview_clip(video_path, start_time, end_time, preview_path):
                    node.preview_path = os.path.relpath(preview_path, os.path.dirname(video_path))
                
                nodes.append(node)
            
            return nodes
            
        finally:
            logger.info("ビデオノード生成完了")
    
    def _apply_scene_data_to_node(self, node: VideoNode, matching_scene: dict, scene_transcripts: list, video_path: str):
        """
        検出されたシーンデータをノードに適用する
        
        Args:
            node: 更新するVideoNodeオブジェクト
            matching_scene: マッチしたシーンデータ
            scene_transcripts: シーンの文字起こしテキストリスト
            video_path: 動画ファイルのパス
        """
        if matching_scene:
            if matching_scene.get("ai_analysis"):
                node.context_analysis.update({
                    "location_type": matching_scene["ai_analysis"].get("scene_type", ""),
                    "estimated_time_of_day": matching_scene["ai_analysis"].get("time_of_day", ""),
                    "weather_conditions": matching_scene["ai_analysis"].get("weather", ""),
                    "key_activities": matching_scene["ai_analysis"].get("activities", [])
                })
            if matching_scene.get("keyframe_path"):
                node.keyframe_path = os.path.relpath(matching_scene["keyframe_path"], 
                                                os.path.dirname(video_path))

        # 説明は映像分析から簡易生成（トランスクリプトを含めない）
        if not node.transcript and not matching_scene:
            node.description = "無音かつ変化のないシーン"
        elif matching_scene and matching_scene.get("ai_analysis"):
            activities = matching_scene["ai_analysis"].get("activities", [])
            # 活動リストを簡潔に記述（長すぎる場合は要約）
            activity_desc = ", ".join(activities) if activities else "活動なし"
            if len(activity_desc) > 100:  # 長すぎる場合は切り詰める
                activity_desc = activity_desc[:97] + "..."
            node.description = f"映像分析: {activity_desc}"
        else:
            node.description = "音声のみのシーン"
    
    def _generate_video_summary(self, video_path: str, nodes: List[VideoNode], transcripts: dict, gopro_metadata: dict) -> dict:
        """
        動画全体のサマリー情報を生成する
        
        Args:
            video_path: 動画ファイルのパス
            nodes: 生成されたVideoNodeのリスト
            transcripts: 文字起こし結果
            gopro_metadata: GoPro メタデータ
            
        Returns:
            dict: 生成されたサマリー情報
        """
        try:
            transcript_text = " ".join([t["text"] for t in transcripts.get("transcripts", [])])
            descriptions = " ".join([node.description for node in nodes if node.description])  # 各シーンの説明を結合
            
            # 映像分析情報を抽出して結合
            visual_info = []
            for node in nodes:
                if hasattr(node, 'context_analysis') and node.context_analysis:
                    activities = node.context_analysis.get("key_activities", [])
                    if activities:
                        visual_info.append(", ".join(activities))
            
            # サマリー生成
            overview = self._generate_summary_with_gemini(transcript_text, descriptions, visual_info)
            
            summary = {
                "title": os.path.basename(video_path),
                "overview": overview if overview else "動画の説明が生成できませんでした",
                "topics": [],
                "filming_date": "",
                "location": "",
                "weather": "不明",
                "purpose": "",
                "transportation": "不明",
                "starting_point": "",
                "destination": "",
                "scene_count": len(nodes),
                "total_duration": nodes[-1].time_out if nodes else 0,
                "gopro_start_time": gopro_metadata.get("start_time", "") if gopro_metadata else ""
            }
            logger.info(f"Geminiによる動画概要生成: {summary['overview']}")
            return summary
            
        except Exception as e:
            logger.error(f"Geminiによるサマリー生成エラー: {str(e)}")
            return {
                "title": os.path.basename(video_path),
                "overview": "サマリー生成に失敗しました",
                "topics": [],
                "scene_count": len(nodes),
                "total_duration": nodes[-1].time_out if nodes else 0,
                "gopro_start_time": gopro_metadata.get("start_time", "") if gopro_metadata else ""
            }
    
    def _generate_summary_with_gemini(self, transcript_text: str, descriptions: str, visual_info: list) -> str:
        """
        Gemini AIを使用して動画の概要を生成する
        
        Args:
            transcript_text: 文字起こしテキスト
            descriptions: シーン説明のテキスト
            visual_info: 視覚的情報のリスト
            
        Returns:
            str: 生成された概要テキスト
        """
        try:
            if not self.gemini_client:
                return "Gemini APIが利用できないため、概要を生成できませんでした。"
                
            # トランスクリプトと映像分析の両方を使用
            prompt = f"""
            以下の情報を基に、動画全体の簡潔で自然な説明を日本語で生成してください：
            - 文字起こし: "{transcript_text}"
            - シーン説明: "{descriptions}"
            - 視覚情報: "{', '.join(visual_info)}"
            
            説明は2〜3文で、動画の概要や主要な活動を自然に描写してください。
            トランスクリプトが空でも、視覚的な情報を使用して意味のある説明を生成してください。
            """
            
            # Geminiクライアントのメソッドを使用して概要を生成
            overview = self.gemini_client.generate_content(prompt)
            if not overview or "エラー" in overview:
                # 映像分析だけでサマリーを生成
                fallback_prompt = f"""
                以下の視覚情報から、動画全体の説明を日本語で生成してください：
                - シーン説明: "{descriptions}"
                
                説明は2〜3文で、動画の概要や主要な内容を自然に描写してください。
                """
                overview = self.gemini_client.generate_content(fallback_prompt)
            return overview
        except Exception as e:
            logger.error(f"Gemini API呼び出しエラー: {str(e)}")
            # フォールバック：映像分析情報からシンプルな概要を生成
            return "この動画は日常的なシーンを記録したものです。" if not descriptions else f"この動画には、{descriptions[:100]}などのシーンが含まれています。"
    
    def extract_gopro_metadata(self, video_path: str) -> dict:
        """GoProのメタデータを抽出"""
        try:
            command = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            metadata = json.loads(result.stdout)
            
            # 開始時間のみを取得
            start_time = metadata.get("format", {}).get("tags", {}).get("creation_time", "")
            
            return {
                "start_time": start_time
            }
        except Exception as e:
            logger.error(f"メタデータ抽出エラー: {str(e)}")
            return None
    
    def save_results(self, video_path: str, nodes: List[VideoNode], completed: bool = True, summary: dict = None):
        """処理結果を保存（新フォーマット）"""
        output_dir = os.path.join(
            os.path.dirname(video_path),
            "video_nodes_" + os.path.splitext(os.path.basename(video_path))[0]
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # シーンデータを新フォーマットに変換
        scenes = []
        for i, node in enumerate(nodes):
            scene = {
                "scene_id": i,
                "time_in": node.time_in,
                "time_out": node.time_out,
                "duration": node.time_out - node.time_in,
                "transcript": node.transcript,
                "description": node.description,
                "keyframe_path": node.keyframe_path,
                "preview_path": node.preview_path,
                "context_analysis": node.context_analysis,
                "editing_suggestions": node.editing_suggestions,
                "tags": node.tags,
                "highlight": node.highlight,
                "custom_notes": node.custom_notes
            }
            scenes.append(scene)
        
        # 結果をJSON形式で保存
        result = {
            "video_path": video_path,
            "completed": completed,
            "last_update": datetime.now().isoformat(),
            "summary": summary if summary else {},
            "scenes": scenes
        }
        
        with open(os.path.join(output_dir, "nodes.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"処理結果を保存しました: {output_dir}/nodes.json")
    
    def show_file_details(self, event=None):
        """選択したファイルの詳細を表示"""
        if not self.file_list.selection():
            return
        
        item_id = self.file_list.selection()[0]
        file_name = self.file_list.item(item_id, "values")[0]
        
        # ファイルが完全なパスかファイル名だけかを確認
        if os.path.isabs(file_name):
            file_path = file_name
        else:
            # 選択されたファイルがself.selected_filesに存在するか確認
            matching_files = [f for f in self.selected_files if os.path.basename(f) == file_name]
            if matching_files:
                file_path = matching_files[0]
            else:
                # ファイルが見つからない場合、エラーメッセージを表示
                logger.error(f"ファイルが見つかりません: {file_name}")
                messagebox.showerror("エラー", f"ファイルが見つかりません: {file_name}")
                return
                
        # 絶対パスに変換
        abs_file_path = os.path.abspath(file_path)
        logger.info(f"詳細表示 - ファイル: {abs_file_path}")
        
        # ファイルの存在確認
        if not os.path.exists(abs_file_path):
            logger.error(f"ファイルが存在しません: {abs_file_path}")
            messagebox.showerror("エラー", f"ファイルが見つかりません:\n{abs_file_path}")
            return
        
        # ノードファイルのパスを取得
        base_name = os.path.splitext(os.path.basename(abs_file_path))[0]
        output_dir = os.path.join(os.path.dirname(abs_file_path), f"video_nodes_{base_name}")
        nodes_file = os.path.join(output_dir, "nodes.json")
        
        if not os.path.exists(nodes_file):
            messagebox.showinfo("情報", "処理結果が見つかりません。まず動画を処理してください。")
            return
        
        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # シーンデータを取得
            scenes = data.get("scenes", data.get("nodes", []))
            
            # プレビューGUIにシーンデータを設定
            if self.preview_gui:
                self.preview_gui.set_scenes(abs_file_path, scenes)
            
        except Exception as e:
            logger.error(f"詳細表示エラー: {str(e)}")
            messagebox.showerror("エラー", f"詳細表示中にエラーが発生しました: {str(e)}")
    
    def update_preview(self, event=None):
        """選択されたファイルのプレビューを表示"""
        selection = self.file_list.selection()
        if not selection:
            return
        
        item_id = selection[0]
        file_name = self.file_list.item(item_id, "values")[0]
        status = self.file_list.item(item_id, "values")[1]
        
        # 処理済みの場合は詳細表示ボタンを有効化
        if status == "処理済":
            self.view_button.config(state="normal")
        else:
            self.view_button.config(state="disabled")
    
    def extract_keyframe(self, video_path: str, timestamp: float, output_path: str) -> bool:
        """指定時間のキーフレームを抽出"""
        try:
            command = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",  # JPEGの品質（2は高品質）
                "-vf", "scale=640:360",  # サムネイルサイズ
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"キーフレーム抽出成功: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"キーフレーム抽出エラー: {str(e)}")
            return False
    
    def generate_preview_clip(self, video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
        """低解像度のプレビュー動画を生成"""
        try:
            duration = end_time - start_time
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-vf", "scale=640:360",  # 360p
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",  # 高圧縮
                "-tune", "fastdecode",  # 再生速度優先
                "-profile:v", "baseline",  # 互換性重視
                "-level", "3.0",
                "-maxrate", "1M",  # ビットレート制限
                "-bufsize", "2M",
                "-c:a", "aac",
                "-b:a", "64k",  # 低音質
                "-ac", "2",  # ステレオ
                "-ar", "44100",  # サンプリングレート
                "-movflags", "+faststart",
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"プレビュー動画生成成功: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"プレビュー動画生成エラー: {str(e)}")
            return False
    
    def on_closing(self):
        """アプリケーション終了時の処理"""
        if self.processing:
            if messagebox.askyesno("確認", "処理中ですが、終了しますか？"):
                self.processing = False
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """アプリケーション実行"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoEnhanced()
    app.run()
