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
    
    # OSによって異なる属性をtry-exceptで処理
    try:
        logger.debug(f"  - ページフォールト: {memory_info.pfaults}")
    except AttributeError:
        logger.debug("  - ページフォールト: 利用不可（このOSでは対応していません）")
        
    try:
        logger.debug(f"  - ページインフォールト: {memory_info.pageins}")
    except AttributeError:
        logger.debug("  - ページインフォールト: 利用不可（このOSでは対応していません）")
    
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
        
        # 3分味見モードボタン
        self.preview_mode_button = ttk.Button(self.action_frame, text="🍲 3分味見", command=self.confirm_preview_mode)
        self.preview_mode_button.pack(side=tk.LEFT, padx=5)
        
        # 続きから再開ボタン
        self.resume_button = ttk.Button(self.action_frame, text="⏯️ 続きから", command=self.confirm_resume_processing)
        self.resume_button.pack(side=tk.LEFT, padx=5)
        
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
    
    def confirm_preview_mode(self):
        """3分味見モードの確認"""
        if not self.selected_files:
            messagebox.showwarning("警告", "処理するファイルが選択されていません。")
            return
        
        # 3分味見モードの説明
        message = "3分味見モードでは、動画の最初の3分間だけを処理します。\n"
        message += "これにより短時間で結果を確認し、設定の調整ができます。\n\n"
        message += "続行しますか？"
        
        if messagebox.askyesno("確認", message):
            self.start_processing(preview_mode=True)
    
    def confirm_resume_processing(self):
        """続きから再開の確認"""
        if not self.selected_files:
            messagebox.showwarning("警告", "処理するファイルが選択されていません。")
            return
        
        # 途中まで処理済みのファイルを確認
        resumable_files = []
        for video_path in self.selected_files:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            checkpoint_file = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}", "checkpoint.json")
            if os.path.exists(checkpoint_file):
                resumable_files.append(video_path)
        
        if not resumable_files:
            messagebox.showinfo("情報", "再開可能な処理はありません。")
            return
        
        # 再開確認ダイアログ
        message = f"{len(resumable_files)}個のファイルを続きから処理できます。\n"
        message += "処理を再開しますか？"
        
        if messagebox.askyesno("確認", message):
            self.start_processing(resume_mode=True)
    
    def start_processing(self, preview_mode=False, resume_mode=False):
        """処理を開始"""
        if self.processing:
            return
        
        self.processing = True
        self.process_button.config(state="disabled")
        self.preview_mode_button.config(state="disabled")
        self.resume_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.open_button.config(state="disabled")
        self.open_folder_button.config(state="disabled")
        
        # 進捗バーをリセット
        self.total_progress_var.set(0)
        self.file_progress_var.set(0)
        
        # 処理モードをログに記録
        mode_desc = "3分味見モード" if preview_mode else "続きから再開" if resume_mode else "通常モード"
        logger.info(f"処理開始: {mode_desc}")
        
        # 別スレッドで処理を実行
        self.processing_thread = threading.Thread(
            target=self.process_files_thread, 
            args=(preview_mode, resume_mode)
        )
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
    
    def process_files_thread(self, preview_mode=False, resume_mode=False):
        """複数ファイルの処理を実行"""
        try:
            total_files = len(self.selected_files)
            processed_count = 0
            
            for i, video_path in enumerate(self.selected_files):
                if not self.processing:  # キャンセルチェック
                    break
                
                # 処理済みファイルをスキップ（再開モードでなければ）
                if not resume_mode and video_path in self.processed_files:
                    logger.info(f"スキップ: {os.path.basename(video_path)} (処理済)")
                    processed_count += 1
                    continue
                
                # 進捗更新
                total_progress = (i / total_files) * 100
                self.total_progress_var.set(total_progress)
                self.file_progress_var.set(0)  # 新しいファイルの開始
                self.update_status(
                    f"処理中 ({i+1}/{total_files})" + (" [3分味見]" if preview_mode else ""),
                    f"{os.path.basename(video_path)}を処理中..."
                )
                
                try:
                    # 個別ファイルの処理
                    self.process_single_video(video_path, preview_mode=preview_mode, resume_mode=resume_mode)
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
                
                if preview_mode:
                    self.update_status(
                        "3分味見完了",
                        f"3分味見モード完了: {processed_count}個のファイルを部分処理しました"
                    )
                else:
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
    def process_single_video(self, video_path: str, preview_mode=False, resume_mode=False):
        """単一の動画を処理する"""
        logger.info(f"動画処理開始: {video_path}" + (" [3分味見モード]" if preview_mode else "") + (" [続きから再開]" if resume_mode else ""))
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
            
            # チェックポイントファイルのパス
            checkpoint_file = os.path.join(output_dir, "checkpoint.json")
            
            # 低解像度のプレビュー動画を生成（処理速度を向上させるため）
            preview_video_path = os.path.join(output_dir, f"preview_{base_name}.mp4")
            
            # 再開モードの場合、チェックポイントを読み込む
            checkpoint_data = None
            if resume_mode and os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    logger.info(f"チェックポイントを読み込みました: {checkpoint_file}")
                except Exception as e:
                    logger.error(f"チェックポイント読み込みエラー: {str(e)}")
                    checkpoint_data = None
            
            # プレビュー動画の生成（チェックポイントから復元しない場合）
            if not checkpoint_data or "preview_video_path" not in checkpoint_data or not os.path.exists(checkpoint_data.get("preview_video_path", "")):
                self.update_status(f"プレビュー動画を生成中", f"{os.path.basename(video_path)}: 低解像度バージョンを作成")
                self.file_progress_var.set(2)
                
                # 3分味見モードの場合の最大時間
                max_duration_param = []
                if preview_mode:
                    max_duration_param = ["-t", "180"]  # 3分制限
                
                # 低解像度プレビュー動画を生成
                preview_video_path = self._create_preview_video(video_path, preview_video_path, preview_mode)
                
                # チェックポイントに保存
                self._save_checkpoint(checkpoint_file, {"preview_video_path": preview_video_path})
            else:
                # チェックポイントから復元
                preview_video_path = checkpoint_data.get("preview_video_path", "")
                logger.info(f"既存のプレビュー動画を使用: {preview_video_path}")
                
            # 実際の処理で使用する動画パスをプレビュー動画に変更
            processing_video_path = preview_video_path if os.path.exists(preview_video_path) else video_path
            logger.info(f"処理に使用する動画: {processing_video_path}")
            
            # 進捗コールバック関数
            def update_progress(progress, message):
                self.file_progress_var.set(progress)
                self.update_status(f"処理中: {message}" + (" [3分味見]" if preview_mode else ""), 
                                  f"{os.path.basename(video_path)}: {message}")
            
            # シーン検出器に進捗コールバックを設定
            self.scene_detector.set_progress_callback(update_progress)
            
            # 3分味見モードの場合、動画の長さを確認
            max_duration = None
            if preview_mode:
                cap = cv2.VideoCapture(processing_video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    total_duration = total_frames / fps
                    cap.release()
                    
                    # 3分（180秒）で制限
                    max_duration = min(180, total_duration)
                    logger.info(f"3分味見モード: 処理を{max_duration:.2f}秒までに制限します（総時間: {total_duration:.2f}秒）")
            
            # 1. 音声認識と無音区間検出
            update_progress(5, "音声認識を開始")
            if checkpoint_data and "transcripts" in checkpoint_data:
                transcripts = checkpoint_data["transcripts"]
                logger.info("チェックポイントから音声認識結果を復元")
            else:
                transcripts = self._extract_transcripts(processing_video_path, max_duration=max_duration)
            
            audio_boundaries = transcripts.get("scene_boundaries", [])
            
            # 3分味見モードの場合、境界を制限
            if preview_mode and max_duration:
                audio_boundaries = [t for t in audio_boundaries if t <= max_duration]
                if audio_boundaries and audio_boundaries[-1] < max_duration:
                    audio_boundaries.append(max_duration)
            
            # チェックポイントに保存
            self._save_checkpoint(checkpoint_file, {"transcripts": transcripts})
            
            # 2. 映像ベースのシーン検出
            update_progress(30, "シーン検出を開始")
            if checkpoint_data and "scene_data" in checkpoint_data:
                scene_data = checkpoint_data["scene_data"]
                logger.info("チェックポイントからシーン検出結果を復元")
            else:
                scene_data = self.scene_detector.detect_scenes(processing_video_path, output_dir, max_duration=max_duration)
            
            # チェックポイントに保存
            self._save_checkpoint(checkpoint_file, {"transcripts": transcripts, "scene_data": scene_data})
            
            # 3. 音声と映像のシーン境界を統合
            update_progress(60, "シーン境界を統合")
            merged_boundaries = self.scene_detector.merge_with_audio_boundaries(audio_boundaries)
            
            # シーン数が多すぎる場合は制限（パフォーマンス向上のため）
            MAX_SCENES = 50
            if len(merged_boundaries) > MAX_SCENES + 1:
                logger.warning(f"シーン数が多すぎるため、上位{MAX_SCENES}個に制限します: {len(merged_boundaries)-1}個 → {MAX_SCENES}個")
                
                # 元の長さを保存
                total_duration = merged_boundaries[-1] - merged_boundaries[0]
                
                # 均等に間引く - 最初と最後を残す
                step = (len(merged_boundaries) - 1) // MAX_SCENES
                selected_indices = [0] + [i for i in range(step, len(merged_boundaries)-1, step)][:MAX_SCENES-1] + [len(merged_boundaries)-1]
                merged_boundaries = [merged_boundaries[i] for i in selected_indices]
                
                logger.info(f"選択されたシーン境界: {len(merged_boundaries)-1}個")
            
            # 4. シーンデータを生成
            update_progress(70, "シーンデータを生成")
            all_boundaries = merged_boundaries
            
            # チェックポイントに保存
            self._save_checkpoint(checkpoint_file, {
                "transcripts": transcripts, 
                "scene_data": scene_data, 
                "merged_boundaries": merged_boundaries
            })
            
            # 5. ビデオノードを生成（並列処理で高速化）
            update_progress(80, "ビデオノードを生成")
            if checkpoint_data and "nodes" in checkpoint_data:
                # ノードを復元（辞書からオブジェクトに変換）
                node_dicts = checkpoint_data["nodes"]
                nodes = [VideoNode.from_dict(node_dict) for node_dict in node_dicts]
                logger.info(f"チェックポイントからノードを復元: {len(nodes)}個")
            else:
                nodes = self._create_video_nodes(
                    processing_video_path, all_boundaries, scene_data, transcripts, 
                    keyframes_dir, preview_dir
                )
            
            # チェックポイントに保存
            self._save_checkpoint(checkpoint_file, {
                "transcripts": transcripts, 
                "scene_data": scene_data, 
                "merged_boundaries": merged_boundaries,
                "nodes": [node.to_dict() for node in nodes]
            })
            
            # 6. GoPro メタデータを抽出
            update_progress(90, "メタデータを抽出")
            if checkpoint_data and "gopro_metadata" in checkpoint_data:
                gopro_metadata = checkpoint_data["gopro_metadata"]
                logger.info("チェックポイントからGoPro メタデータを復元")
            else:
                gopro_metadata = self.extract_gopro_metadata(video_path)  # 元の動画から抽出
            
            # チェックポイントに保存
            self._save_checkpoint(checkpoint_file, {
                "transcripts": transcripts, 
                "scene_data": scene_data, 
                "merged_boundaries": merged_boundaries,
                "nodes": [node.to_dict() for node in nodes],
                "gopro_metadata": gopro_metadata
            })
            
            # 7. 動画全体のサマリー情報を生成
            update_progress(95, "サマリー情報を生成")
            if checkpoint_data and "summary" in checkpoint_data:
                summary = checkpoint_data["summary"]
                logger.info("チェックポイントからサマリー情報を復元")
            else:
                summary = self._generate_video_summary(video_path, nodes, transcripts, gopro_metadata)
            
            # チェックポイントに保存（ここで完了フラグも保存）
            self._save_checkpoint(checkpoint_file, {
                "transcripts": transcripts, 
                "scene_data": scene_data, 
                "merged_boundaries": merged_boundaries,
                "nodes": [node.to_dict() for node in nodes],
                "gopro_metadata": gopro_metadata,
                "summary": summary,
                "completed": True,
                "preview_mode": preview_mode,
                "timestamp": datetime.now().isoformat()
            })
            
            # 8. 結果を保存
            update_progress(98, "結果を保存")
            self.save_results(video_path, nodes, True, summary, preview_mode=preview_mode)
            
            update_progress(100, "処理完了")
            logger.info(f"動画処理完了: {video_path}" + (" [3分味見モード]" if preview_mode else ""))
            
            return nodes
            
        finally:
            log_memory_usage()
            logger.info(f"動画処理終了: {video_path}")
    
    def _create_preview_video(self, input_path: str, output_path: str, preview_mode: bool = False) -> str:
        """
        低解像度のプレビュー動画を生成する（処理速度向上のため）
        
        Args:
            input_path: 入力動画パス
            output_path: 出力動画パス
            preview_mode: 3分味見モードかどうか
            
        Returns:
            str: 生成されたプレビュー動画のパス
        """
        try:
            start_time = time.time()
            logger.info(f"プレビュー動画の生成を開始: {input_path} -> {output_path}")
            
            # コマンドの基本部分
            command = [
                "ffmpeg", "-y",
                "-i", input_path,
            ]
            
            # 3分味見モードの場合は時間制限を追加
            if preview_mode:
                command.extend(["-t", "180"])  # 3分制限
            
            # 低解像度、高速エンコード設定
            command.extend([
                "-vf", "scale=640:360",  # 360p解像度
                "-c:v", "libx264",
                "-preset", "ultrafast",  # 最も高速なエンコード設定
                "-crf", "28",  # 高圧縮（低品質）
                "-tune", "fastdecode",  # デコード速度優先
                "-profile:v", "baseline",  # 互換性重視
                "-level", "3.0",
                "-maxrate", "1M",  # ビットレート制限
                "-bufsize", "2M",
                "-c:a", "aac",
                "-b:a", "64k",  # 低音質
                "-ac", "1",  # モノラル
                "-ar", "22050",  # 低サンプリングレート
                "-movflags", "+faststart",
                output_path
            ])
            
            # FFmpegを実行
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"プレビュー動画生成エラー: {result.stderr}")
                return input_path  # 失敗した場合は元の動画パスを返す
                
            # 生成された動画の情報を取得
            end_time = time.time()
            duration = end_time - start_time
            
            # ファイルサイズを取得（MB単位）
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            
            logger.info(f"プレビュー動画生成完了: {output_path}")
            logger.info(f"  処理時間: {duration:.2f}秒")
            logger.info(f"  ファイルサイズ: {file_size_mb:.2f}MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"プレビュー動画生成に失敗: {str(e)}")
            return input_path  # 失敗した場合は元の動画パスを返す
    
    def _save_checkpoint(self, checkpoint_file: str, data: dict):
        """チェックポイントデータを保存"""
        try:
            # 既存のチェックポイントがあれば読み込む
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
            else:
                checkpoint_data = {}
            
            # 新しいデータで更新
            checkpoint_data.update(data)
            
            # タイムスタンプを追加
            checkpoint_data["last_update"] = datetime.now().isoformat()
            
            # 保存
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"チェックポイントを保存しました: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"チェックポイント保存エラー: {str(e)}")
    
    @log_exceptions
    def _extract_transcripts(self, video_path: str, max_duration=None) -> Dict:
        """
        音声トランスクリプトとシーン境界を抽出
        
        Args:
            video_path: 動画ファイルのパス
            max_duration: 処理する最大時間（秒）、Noneの場合は全体を処理
        """
        try:
            logger.info(f"音声からトランスクリプトとシーン境界を抽出中: {video_path}" + (f" (最大{max_duration}秒まで)" if max_duration else ""))
            
            # WhisperClientの存在を確認
            if not hasattr(self, 'whisper_client') or self.whisper_client is None:
                logger.error("WhisperClientが初期化されていません。faster-whisperがインストールされているか確認してください。")
                logger.error("pip install faster-whisper torchを実行してインストールしてください。")
                # 空の結果を返す
                return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
            
            # faster-whisperの存在確認を追加
            try:
                import faster_whisper
                logger.info("faster-whisperモジュールが利用可能です。")
            except ImportError:
                logger.error("faster-whisperモジュールがインストールされていません。")
                logger.error("音声認識を有効にするには: pip install faster-whisper torch")
                # 空の結果を返す
                return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
            
            # 音声ファイルの一時パス
            audio_file = os.path.join(
                os.path.dirname(video_path),
                f"temp_audio_{os.path.splitext(os.path.basename(video_path))[0]}.wav"
            )
            
            try:
                # 音声抽出（最大時間制限がある場合）
                if max_duration is not None:
                    logger.info(f"最大時間を{max_duration}秒に制限して処理します")
                    command = [
                        "ffmpeg", "-y",
                        "-i", video_path,
                        "-t", str(max_duration),
                        "-vn",
                        "-ar", "16000",
                        "-ac", "1",
                        "-b:a", "32k",
                        audio_file
                    ]
                    result = subprocess.run(command, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"音声抽出に失敗しました: {result.stderr}")
                        return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
                else:
                    # 通常の音声抽出
                    logger.info(f"動画から音声を抽出中: {video_path} -> {audio_file}")
                    try:
                        self.whisper_client.extract_audio(video_path, audio_file)
                    except Exception as e:
                        logger.error(f"音声抽出に失敗しました: {str(e)}")
                        # 直接FFmpegを使用して抽出を試みる
                        logger.info("FFmpegを使用して音声抽出を再試行します")
                        command = [
                            "ffmpeg", "-y",
                            "-i", video_path,
                            "-vn",
                            "-ar", "16000",
                            "-ac", "1",
                            "-b:a", "32k",
                            audio_file
                        ]
                        result = subprocess.run(command, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.error(f"FFmpegによる音声抽出も失敗しました: {result.stderr}")
                            return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
                
                # 音声ファイルの存在確認
                if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                    logger.error(f"音声ファイルが正しく生成されませんでした: {audio_file}")
                    return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
                
                logger.info(f"音声ファイルサイズ: {os.path.getsize(audio_file)/1024:.2f} KB")
                
                # Whisperで文字起こし
                logger.info(f"音声ファイルの文字起こしを開始: {audio_file}")
                try:
                    segments = self.whisper_client.transcribe(audio_file)
                    logger.info(f"文字起こし完了: {len(segments)}個のセグメント")
                except Exception as e:
                    logger.error(f"文字起こし処理に失敗しました: {str(e)}")
                    logger.error(traceback.format_exc())
                    return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
                
                # セグメントから無音区間と文字起こしを抽出
                scene_boundaries = []
                transcripts = []
                
                logger.info(f"セグメント数: {len(segments)}")
                
                if not segments:
                    logger.warning("文字起こしセグメントが見つかりませんでした")
                    return {"transcripts": [], "scene_boundaries": [0.0, max_duration or 0.0]}
                
                # 無音区間を検出するためのチャンク期間
                chunk_duration = getattr(self.whisper_client, 'chunk_duration', 1.0)
                logger.info(f"無音区間検出のチャンク期間: {chunk_duration}秒")
                
                for i, segment in enumerate(segments):
                    # 最大時間を超えるセグメントは無視
                    if max_duration is not None and segment.start > max_duration:
                        continue
                    
                    # 最大時間で終了時間を制限
                    end_time = segment.end
                    if max_duration is not None and end_time > max_duration:
                        end_time = max_duration
                    
                    # 各セグメントをトランスクリプトに追加
                    transcripts.append({
                        "start": segment.start,
                        "end": end_time,
                        "text": segment.text
                    })
                    
                    # 次のセグメントとの間に無音区間があるか確認
                    if i < len(segments) - 1:
                        next_seg = segments[i + 1]
                        # 最大時間を超える場合はスキップ
                        if max_duration is not None and segment.end > max_duration:
                            continue
                            
                        gap = next_seg.start - segment.end
                        if gap >= chunk_duration:
                            scene_boundaries.append(segment.end)
                            logger.debug(f"無音区間を検出: {segment.end}秒 (ギャップ: {gap:.2f}秒)")
                
                # 最大時間で区切る
                if max_duration is not None and (not scene_boundaries or scene_boundaries[-1] < max_duration):
                    if scene_boundaries and scene_boundaries[-1] < max_duration:
                        scene_boundaries.append(max_duration)
                    elif not scene_boundaries:
                        scene_boundaries = [0.0, max_duration]
                
                # 最低でも開始点と終了点を含める
                if not scene_boundaries:
                    if transcripts:
                        scene_boundaries = [0.0, transcripts[-1]["end"]]
                    else:
                        # トランスクリプトがなければ、最大時間かダミーの境界を設定
                        if max_duration:
                            scene_boundaries = [0.0, max_duration]
                        else:
                            # 動画の長さを取得
                            try:
                                cap = cv2.VideoCapture(video_path)
                                if cap.isOpened():
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    duration = frame_count / fps
                                    scene_boundaries = [0.0, duration]
                                    cap.release()
                                else:
                                    scene_boundaries = [0.0, 60.0]  # ダミー境界
                            except:
                                scene_boundaries = [0.0, 60.0]  # ダミー境界
                
                logger.info(f"文字起こし処理完了: {len(transcripts)}個のセグメント、{len(scene_boundaries)}個の無音区間")
                
                # トランスクリプトの内容を確認（デバッグ用）
                total_text = sum(len(t["text"]) for t in transcripts)
                logger.info(f"トランスクリプト総文字数: {total_text}文字")
                
                if transcripts and total_text > 0:
                    # 最初と最後のセグメントの例を出力
                    logger.info(f"最初のセグメント: {transcripts[0]['text'][:50]}...")
                    if len(transcripts) > 1:
                        logger.info(f"最後のセグメント: {transcripts[-1]['text'][:50]}...")
                
                return {
                    "scene_boundaries": scene_boundaries,
                    "transcripts": transcripts
                }
            
            finally:
                # 一時ファイルを削除
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                        logger.debug(f"一時音声ファイルを削除: {audio_file}")
                    except Exception as e:
                        logger.warning(f"一時ファイル削除に失敗: {str(e)}")
        
        except Exception as e:
            logger.error(f"トランスクリプト抽出中にエラー発生: {str(e)}")
            logger.error(traceback.format_exc())
            # エラー時もダミーのシーン境界を返す
            if max_duration:
                scene_boundaries = [0.0, max_duration]
            else:
                scene_boundaries = [0.0, 60.0]  # ダミー境界
            return {"transcripts": [], "scene_boundaries": scene_boundaries}
    
    @log_exceptions
    def _create_video_nodes(self, video_path: str, scenes: List[Dict], scene_data: Dict, transcripts: Dict, keyframes_dir: str, preview_dir: str) -> List[VideoNode]:
        """検出されたシーンからVideoNodeオブジェクトを作成"""
        start_time = time.time()
        
        # トランスクリプトのデバッグ情報を追加
        if isinstance(transcripts, dict):
            transcripts_list = transcripts.get('transcripts', [])
            boundaries_list = transcripts.get('scene_boundaries', [])
            logger.info(f"{len(scenes)-1 if isinstance(scenes, list) else 0}個のシーンと{len(transcripts_list)}個のトランスクリプトからノードを作成")
            logger.info(f"シーン境界: {len(boundaries_list)}個")
            
            # トランスクリプトの詳細をログ出力
            for i, t in enumerate(transcripts_list[:3]):  # 最初の3つだけを表示
                logger.info(f"トランスクリプト例 #{i}: {t.get('start', 0):.2f}s-{t.get('end', 0):.2f}s: {t.get('text', '')[:50]}...")
        else:
            logger.error(f"transcriptsの型が不正: {type(transcripts)}")
            logger.error(f"transcripts内容: {transcripts}")
            transcripts_list = []
            logger.warning("トランスクリプトが空または不正な形式です。テキスト認識なしで処理を続行します。")
        
        nodes = []
        
        # トランスクリプトの範囲を確認（デバッグ用）
        if transcripts_list:
            first_ts = transcripts_list[0].get('start', 0)
            last_ts = transcripts_list[-1].get('end', 0)
            logger.info(f"トランスクリプト時間範囲: {first_ts:.2f}秒 - {last_ts:.2f}秒 (合計: {last_ts-first_ts:.2f}秒)")
        else:
            logger.warning("トランスクリプトが空です。音声認識が機能していないか、音声データがない可能性があります。")
            logger.warning("音声認識を有効にするには、faster-whisperパッケージがインストールされているか確認してください。")
        
        # 処理時間短縮のためにトランスクリプトをメモリ内でインデックス化
        transcript_index = {}
        for t in transcripts_list:
            # 0.5秒単位で丸めてインデックス化（より効率的な検索のため）
            start_key = int(t.get('start', 0) * 2) / 2
            end_key = int(t.get('end', 0) * 2) / 2
            for time_key in [k/2 for k in range(int(start_key*2), int(end_key*2)+1)]:
                if time_key not in transcript_index:
                    transcript_index[time_key] = []
                transcript_index[time_key].append(t)
        
        for i, (start_time_scene, end_time_scene) in enumerate(zip(scenes[:-1], scenes[1:])):
            try:
                # 進捗ログを追加
                if i % 10 == 0 or i == len(scenes)-2:  # 10シーンごと、または最後のシーンでログ出力
                    logger.info(f"ノード生成中: シーン {i+1}/{len(scenes)-1} を処理中... ({start_time_scene:.2f}s-{end_time_scene:.2f}s)")
                
                # このシーンの時間範囲内にあるトランスクリプトを高速に取得
                scene_transcripts = set()  # 重複を避けるためセットを使用
                
                # 時間範囲内のキーをチェック
                for time_key in [k/2 for k in range(int(start_time_scene*2), int(end_time_scene*2)+1)]:
                    if time_key in transcript_index:
                        for t in transcript_index[time_key]:
                            scene_transcripts.add(t.get('text', ''))
                
                # トランスクリプトをテキストに変換
                transcript_text = " ".join(scene_transcripts)
                
                # 各シーンのトランスクリプト情報をログ出力（最初の数シーンのみ）
                if i < 5:
                    logger.info(f"シーン{i+1} トランスクリプト件数: {len(scene_transcripts)}件")
                    if scene_transcripts:
                        text_length = len(transcript_text)
                        logger.info(f"シーン{i+1} トランスクリプトテキスト長: {text_length}文字")
                        if text_length > 0:
                            logger.info(f"シーン{i+1} トランスクリプト冒頭: {transcript_text[:50]}...")
                
                # キーフレームのパスを確認
                keyframe_path = ""
                if scene_data and i < len(scene_data):
                    keyframe_path = scene_data[i].get('keyframe_path', '')
                    if not keyframe_path or not os.path.exists(keyframe_path):
                        logger.warning(f"シーン {i+1}: キーフレームが見つかりません - {keyframe_path}")
                        # 再抽出を試みる
                        fallback_keyframe = os.path.join(keyframes_dir, f"scene_{i:04d}.jpg")
                        keyframe_time = (start_time_scene + end_time_scene) / 2
                        if self.extract_keyframe(video_path, keyframe_time, fallback_keyframe):
                            keyframe_path = fallback_keyframe
                            logger.info(f"シーン {i+1}: キーフレーム再抽出成功 - {fallback_keyframe}")
                
                # プレビュー動画のパス
                preview_path = os.path.join(preview_dir, f"scene_{i:04d}.mp4")
                
                # VideoNodeの作成
                node = VideoNode(start_time_scene, end_time_scene)
                node.transcript = transcript_text
                node.keyframe_path = keyframe_path
                node.preview_path = preview_path
                
                # シーンデータから追加情報を設定
                if scene_data and i < len(scene_data):
                    self._apply_scene_data_to_node(node, scene_data[i], list(scene_transcripts), video_path)
                
                # プレビュー動画を生成（シーンの長さが最小値を超える場合のみ）
                if end_time_scene - start_time_scene >= 0.5 and not os.path.exists(preview_path):
                    success = self.generate_preview_clip(video_path, start_time_scene, end_time_scene, preview_path)
                    if not success:
                        logger.warning(f"シーン {i+1}: プレビュー動画生成に失敗しました")
                
                nodes.append(node)
                
            except Exception as e:
                logger.error(f"シーン #{i+1}のVideoNode作成中にエラー: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        end_time_proc = time.time()
        processing_time = end_time_proc - start_time
        
        logger.info(f"{len(nodes)}個のVideoNodeを作成しました (処理時間: {processing_time:.2f}秒, 平均: {processing_time/max(1, len(nodes)):.2f}秒/ノード)")
        
        # 作成したノードの内容を確認（デバッグ用）
        nodes_with_transcript = sum(1 for n in nodes if n.transcript)
        logger.info(f"トランスクリプトを含むノード: {nodes_with_transcript}/{len(nodes)}")
        
        if nodes:
            # 最初と最後のノードのトランスクリプトを出力
            logger.debug(f"最初のノードのトランスクリプト: {nodes[0].transcript[:50]}...")
            if len(nodes) > 1:
                logger.debug(f"最後のノードのトランスクリプト: {nodes[-1].transcript[:50]}...")
        
        return nodes
    
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
    
    def save_results(self, video_path: str, nodes: List[VideoNode], completed: bool = True, summary: dict = None, preview_mode: bool = False):
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
            "preview_mode": preview_mode,  # 3分味見モードかどうかのフラグを追加
            "summary": summary if summary else {},
            "scenes": scenes
        }
        
        with open(os.path.join(output_dir, "nodes.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"処理結果を保存しました: {output_dir}/nodes.json" + (" [3分味見モード]" if preview_mode else ""))
    
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
            
            # 長いシーンの場合は短く切り詰める（最大15秒）
            if duration > 15:
                # シーンの最初の10秒と最後の5秒だけを生成
                middle_time = start_time + 10
                command = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-i", video_path,
                    "-t", "10",  # 前半10秒
                    "-vf", "scale=480:270",  # 270p（軽量化）
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "30",  # 超高圧縮
                    "-tune", "fastdecode",  # 再生速度優先
                    "-profile:v", "baseline",  # 互換性重視
                    "-level", "3.0",
                    "-maxrate", "500k",  # ビットレート制限
                    "-bufsize", "1M",
                    "-c:a", "aac",
                    "-b:a", "32k",  # 低音質
                    "-ac", "1",  # モノラル
                    "-ar", "22050",  # 低サンプリングレート
                    "-movflags", "+faststart",
                    output_path
                ]
            else:
                # 通常のシーン（15秒以内）
                command = [
                    "ffmpeg", "-y",
                    "-ss", str(start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-vf", "scale=480:270",  # 270p（軽量化）
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "30",  # 超高圧縮
                    "-tune", "fastdecode",  # 再生速度優先
                    "-profile:v", "baseline",  # 互換性重視
                    "-level", "3.0",
                    "-maxrate", "500k",  # ビットレート制限
                    "-bufsize", "1M",
                    "-c:a", "aac",
                    "-b:a", "32k",  # 低音質
                    "-ac", "1",  # モノラル
                    "-ar", "22050",  # 低サンプリングレート
                    "-movflags", "+faststart",
                    output_path
                ]
            
            # 非表示モードで実行（進捗表示しない）
            subprocess.run(command, check=True, capture_output=True)
            
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

    def _process_video(self, video_path: str) -> List[Dict]:
        """
        ビデオを処理してシーン情報を生成する
        """
        logger.info(f"ビデオ処理開始: {video_path}")
        
        try:
            # ビデオの存在確認
            if not os.path.exists(video_path):
                logger.error(f"ビデオファイルが存在しません: {video_path}")
                return []
            
            # メタデータの抽出（GoPro撮影時間などの取得）
            metadata = self._extract_video_metadata(video_path)
            gopro_start_time = metadata.get('gopro_start_time', '')
            
            if not gopro_start_time:
                logger.warning("GoPro開始時間がメタデータから取得できませんでした。EDL生成に影響する可能性があります。")
            else:
                logger.info(f"GoPro開始時間検出: {gopro_start_time}")
            
            # シーン検出
            scene_data = self._detect_scenes(video_path)
            
            if not scene_data or len(scene_data) == 0:
                logger.error("シーン検出に失敗しました。空の結果を返します。")
                return []
            
            logger.info(f"{len(scene_data)}個のシーンを検出しました")
            
            # 代表的なシーンの選択
            selected_scenes = self._select_representative_scenes(scene_data, video_path)
            
            # 各シーンの詳細データを生成
            annotated_scenes = self._generate_scene_data(selected_scenes, video_path)
            
            # トランスクリプトの抽出と割り当て
            transcripts = self._extract_transcripts(video_path)
            
            if not transcripts:
                logger.warning("音声トランスクリプトが検出されませんでした。音声認識に問題がある可能性があります。")
            else:
                logger.info(f"{len(transcripts)}個の音声セグメントを検出しました")
            
            # ビデオノード作成
            nodes = self._create_video_nodes(annotated_scenes, transcripts)
            
            # AI分析で要約を生成
            summary = self._generate_summary(nodes, transcripts)
            
            # デモモード検出
            is_demo_mode = self._is_demo_mode(summary)
            if is_demo_mode:
                logger.warning("デモモードが検出されました。Gemini APIとの接続に問題がある可能性があります。")
                logger.warning("APIキーの設定とネットワーク接続を確認してください。")
            
            # gopro_start_timeを設定
            summary["gopro_start_time"] = gopro_start_time
            
            # 結果を整形して返却
            return {
                "summary": summary,
                "nodes": [node.to_dict() for node in nodes]
            }
        
        except Exception as e:
            logger.error(f"ビデオ処理中に例外が発生しました: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"summary": {"title": "処理エラー", "error": str(e)}, "nodes": []}

    def _is_demo_mode(self, summary: dict) -> bool:
        """
        Gemini APIの応答がデモモードかどうかを検出する
        
        デモモードの特徴:
        1. filming_dateが1970-01-01で始まる
        2. api_errorフラグがTrueに設定されている
        3. overviewに「APIエラー」や「API呼び出しエラー」が含まれる
        """
        try:
            # 明示的なAPIエラーフラグがある場合
            if summary.get("api_error") == True:
                return True
            
            # 撮影日が1970年（エポック時間の開始）になっている場合
            filming_date = summary.get("filming_date", "")
            if filming_date.startswith("1970-01-01"):
                return True
            
            # 概要にエラーメッセージが含まれる場合
            overview = summary.get("overview", "").lower()
            if "api" in overview and ("エラー" in overview or "error" in overview):
                return True
            
            return False
        except Exception as e:
            logger.error(f"デモモード検出中にエラーが発生: {str(e)}")
            return False

    def _extract_video_metadata(self, video_path: str) -> dict:
        """ビデオファイルからメタデータを抽出する"""
        metadata = {}
        
        try:
            logger.info(f"ビデオメタデータ抽出開始: {video_path}")
            
            # FFmpegを使用してメタデータを抽出
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                video_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(result.stdout)
                
                # 基本的なメタデータを抽出
                if "format" in data:
                    format_data = data["format"]
                    metadata["duration"] = float(format_data.get("duration", 0))
                    metadata["file_size"] = int(format_data.get("size", 0))
                    
                    # メタデータタグを確認
                    tags = format_data.get("tags", {})
                    
                    # GoPro固有のメタデータを検索
                    for key, value in tags.items():
                        if "creation_time" in key.lower():
                            metadata["creation_time"] = value
                            logger.info(f"作成時間メタデータを検出: {value}")
                        
                        # GoPro開始時間を特定のタグから探す
                        if any(gopro_key in key.lower() for gopro_key in ["gopro", "start", "timecode"]):
                            metadata["gopro_start_time"] = value
                            logger.info(f"GoPro開始時間を検出: {key}={value}")
                
                # ストリーム情報を抽出
                if "streams" in data:
                    for stream in data["streams"]:
                        if stream.get("codec_type") == "video":
                            metadata["width"] = stream.get("width")
                            metadata["height"] = stream.get("height")
                            metadata["codec"] = stream.get("codec_name")
                            metadata["frame_rate"] = self._parse_frame_rate(stream.get("r_frame_rate", "0/1"))
                            
                            # ビデオストリームからもタグを確認
                            stream_tags = stream.get("tags", {})
                            for key, value in stream_tags.items():
                                if any(gopro_key in key.lower() for gopro_key in ["gopro", "start", "timecode"]):
                                    metadata["gopro_start_time"] = value
                                    logger.info(f"ビデオストリームからGoPro開始時間を検出: {key}={value}")
                
                # GoPro開始時間が見つからない場合は作成時間を使用
                if "gopro_start_time" not in metadata and "creation_time" in metadata:
                    metadata["gopro_start_time"] = metadata["creation_time"]
                    logger.info("GoPro開始時間が見つからないため、ファイル作成時間を使用します")
            
            except subprocess.CalledProcessError as e:
                logger.error(f"FFprobeの実行に失敗: {e}")
                logger.error(f"stderr: {e.stderr}")
            
            except json.JSONDecodeError as e:
                logger.error(f"メタデータのJSON解析に失敗: {e}")
            
            # GoPro開始時間が見つからない場合はデフォルト値を設定
            if "gopro_start_time" not in metadata:
                # 現在時刻をISO形式で設定
                import datetime
                current_time = datetime.datetime.now().isoformat()
                metadata["gopro_start_time"] = current_time
                logger.warning(f"GoPro開始時間が検出できなかったため、現在時刻を使用します: {current_time}")
                logger.warning("これにより、EDLファイルのタイムコードが正確でない可能性があります")
            
            return metadata
        
        except Exception as e:
            logger.error(f"メタデータ抽出中に例外が発生: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"gopro_start_time": "", "duration": 0}
        
    def _parse_frame_rate(self, frame_rate_str: str) -> float:
        """フレームレート文字列をパースする (例: '30000/1001')"""
        try:
            if '/' in frame_rate_str:
                num, den = map(int, frame_rate_str.split('/'))
                if den == 0:
                    return 0
                return num / den
            return float(frame_rate_str)
        except (ValueError, ZeroDivisionError):
            logger.error(f"フレームレートのパースに失敗: {frame_rate_str}")
            return 30.0  # デフォルト値

if __name__ == "__main__":
    app = VideoEnhanced()
    app.run()
