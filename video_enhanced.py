import os
import sys
import json
import time
import shutil
import logging
import threading
import subprocess
import glob  # globモジュールを追加
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

# OpenCVのFFmpeg読み取り試行回数を増加（マルチストリーム処理の安定性向上）
cv2.setNumThreads(4)

# GUI関連のインポート
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ttkthemes import ThemedTk

# OpenCVのFFmpeg読み取り試行回数を増加（マルチストリーム処理の安定性向上）
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

# メインコンテナ
        root = tk.Tk()
        root.title("動画解析・編集アシスタント")
        root.geometry("1200x800")
        
        main_frame = ttk.Frame(root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ヘッダーフレーム
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="動画解析・編集アシスタント", style='Title.TLabel').pack(side=tk.LEFT, padx=5)
        
        # ファイル選択ボタン
        button_frame = ttk.Frame(header_frame)
        button_frame.pack(side=tk.RIGHT)
        
        self.open_button = ttk.Button(button_frame, text="📂 動画を選択", command=self.select_files)
        self.open_button.pack(side=tk.LEFT, padx=5)
        
        self.open_folder_button = ttk.Button(button_frame, text="📁 フォルダを選択", command=self.select_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=5)
        
        # 左右のメインペイン
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左ペイン（ファイルリスト）
        list_frame = ttk.Frame(paned, padding="0 0 0 0")
        
        # リストのラベル
        ttk.Label(list_frame, text="処理対象ファイル", style='Header.TLabel').pack(anchor="w", padx=5, pady=(0, 5))
        
        # ファイルリスト
        columns = ("ファイル名", "状態", "シーン数")
        self.file_list = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        self.file_list.column("ファイル名", width=500)  # 幅を広げる
        self.file_list.column("状態", width=100)
        self.file_list.column("シーン数", width=70)
        
        for col in columns:
            self.file_list.heading(col, text=col)
        
        self.file_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ファイルリストスクロールバー
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.configure(yscrollcommand=list_scrollbar.set)
        
        # ファイルリストの選択イベントをバインド
        self.file_list.bind("<<TreeviewSelect>>", self.update_preview)
        
        # アクションボタンフレーム
        action_frame = ttk.Frame(list_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.process_button = ttk.Button(action_frame, text="▶️ 処理開始", command=self.confirm_start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(action_frame, text="⏹️ 停止", command=self.cancel_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.view_button = ttk.Button(action_frame, text="👁️ 詳細表示", command=self.show_file_details, state=tk.DISABLED)
        self.view_button.pack(side=tk.RIGHT, padx=5)
        
        # 右ペイン（詳細情報）
        self.details_frame = ttk.Frame(paned)
        
        # 右ペインを上下に分割
        right_paned = ttk.PanedWindow(self.details_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # 上部：サムネイル表示エリア
        thumbnail_frame = ttk.Frame(right_paned)
        right_paned.add(thumbnail_frame, weight=1)
        
        # サムネイル表示
        self.preview_label = ttk.Label(thumbnail_frame, text="サムネイルがありません", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 下部：プレビュー再生エリア
        preview_frame = ttk.Frame(right_paned)
        right_paned.add(preview_frame, weight=1)
        
        # プレビュー再生エリアのタイトル
        ttk.Label(preview_frame, text="プレビュー", style='Header.TLabel').pack(anchor="w", padx=5, pady=5)
        
        # 動画再生フレーム
        video_container = ttk.Frame(preview_frame, relief="solid", borderwidth=1)
        video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 実際の動画表示領域 - tkフレームに変更（背景色設定用）
        self.video_frame = tk.Frame(video_container, width=400, height=300, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # VLCの代わりにデフォルトで表示するラベル
        self.video_label = ttk.Label(self.video_frame, text="動画プレビュー\nファイルを選択してください", 
                                   anchor="center", background="black", foreground="white")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # コントロールパネル
        controls_frame = ttk.Frame(preview_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 再生/停止ボタン
        self.play_button = ttk.Button(controls_frame, text="▶ 再生", command=self.play_preview)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(controls_frame, text="⏹ 停止", command=self.stop_preview)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 再生時間表示
        self.time_label = ttk.Label(controls_frame, text="再生時間: 00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        # パネル分割の設定
        paned.add(list_frame, weight=1)
        paned.add(self.details_frame, weight=2)
        
        # ステータスバー
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, padding=(5, 2))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        
        self.status_label = ttk.Label(status_frame, text="準備完了")
        self.status_label.pack(side=tk.LEFT)
        
        # 進捗バー（ファイル全体）
        self.progress_frame = ttk.LabelFrame(status_frame, text="進捗", padding=(5, 2))
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
            # プレイヤーが初期化できたらラベルを削除
            self.video_label.pack_forget()
            if os.name == 'nt':  # Windows
                self.player.set_hwnd(self.video_frame.winfo_id())
            else:  # Linux/MacOS
                self.player.set_xwindow(self.video_frame.winfo_id())
            logger.info("VLCプレーヤーを初期化しました")
        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"VLCライブラリが見つかりません: {str(e)}")
            self.video_label.config(text="VLCプレーヤーが利用できません\nVLCをインストールしてください")
        except Exception as e:
            logger.error(f"VLCプレーヤー初期化エラー: {str(e)}")
            self.video_label.config(text=f"VLCプレーヤー初期化エラー\n{str(e)[:50]}...")
        
        # Gemini Clientの初期化
        try:
            self.gemini_client = GeminiClient()
            logger.info("Gemini Clientを初期化しました")
        except Exception as e:
            logger.error(f"Gemini Client初期化エラー: {str(e)}")
            # フォールバック：シンプルな機能を持つダミーインスタンスを作成
            self.gemini_client = GeminiClient()  # クラスがダミー実装の場合はこれで問題ない
        
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
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
            nodes_file = os.path.join(output_dir, "nodes.json")
            keyframes_dir = os.path.join(output_dir, "keyframes")

            status = ""
            scene_count = "-"
            thumbnail = "-"
            
            if os.path.exists(nodes_file):
                try:
                    with open(nodes_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("completed"):
                            status = "✓ 完了"
                            # 新フォーマットと旧フォーマットの両方に対応
                            scene_nodes = data.get("scenes", data.get("nodes", []))
                            scene_count = str(len(scene_nodes))
                            # 最初のキーフレームのパスを取得
                            keyframe_files = sorted(glob.glob(os.path.join(keyframes_dir, "*.jpg")))
                            first_keyframe = keyframe_files[0] if keyframe_files else None
                            if first_keyframe:
                                thumbnail = os.path.relpath(first_keyframe, os.path.dirname(video_path))
                            else:
                                thumbnail = "キーフレームが見つかりません"
                        else:
                            status = "⚠ 未完了"
                except Exception as e:
                    logger.error(f"ノードファイル読み込みエラー: {str(e)}")
                    status = "⚠ エラー"

            # ツリービューにファイル情報を挿入
            item_id = self.file_list.insert("", tk.END, values=(
                video_path,  # 完全なパスを表示
                status,
                scene_count,
                thumbnail
            ))
            
            # 処理済みファイルは選択状態に
            if status == "✓ 完了":
                self.file_list.selection_set(item_id)
                
        # 選択アイテムのプレビューを表示
        self.update_preview()
            
        self.update_status("準備完了", f"{len(self.selected_files)}個のビデオファイルが選択されました")

    def confirm_start_processing(self):
        """処理開始の確認"""
        if not self.selected_files:
            messagebox.showerror("エラー", "処理する動画ファイルが選択されていません")
            return
        
        if messagebox.askyesno("確認", "動画ファイルの処理を開始してもよろしいですか？"):
            self.start_batch_processing()

    def confirm_cancel_processing(self):
        """処理キャンセルの確認"""
        if messagebox.askyesno("確認", "処理を中止してもよろしいですか？"):
            self.cancel_processing()

    def start_batch_processing(self):
        """バッチ処理を開始"""
        if not self.selected_files:
            messagebox.showerror("エラー", "処理する動画ファイルが選択されていません")
            return
        
        if self.processing:
            messagebox.showinfo("処理中", "すでに処理が実行中です")
            return
        
        # UI状態の更新
        self.processing = True
        self.open_button.config(state="disabled")
        self.open_folder_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.total_progress_var.set(0)
        self.file_progress_var.set(0)
        
        # スレッドで処理を実行
        self.processing_thread = threading.Thread(
            target=self.process_files_thread
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 定期的に状態をチェック
        self.root.after(100, self.check_thread_status)

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
            # 既存のコード
            ...
            
        finally:
            log_memory_usage()
            logger.info(f"動画処理完了: {video_path}")

    @log_exceptions
    def _extract_transcripts(self, video_path: str) -> dict:
        """音声認識を実行する"""
        logger.info(f"音声認識開始: {video_path}")
        log_memory_usage()
        
        try:
            # 既存のコード
            ...
            
        finally:
            log_memory_usage()
            logger.info(f"音声認識完了: {video_path}")

    @log_exceptions
    def _detect_scenes(self, video_path: str) -> list:
        """シーン検出を実行する"""
        logger.info(f"シーン検出開始: {video_path}")
        log_memory_usage()
        
        try:
            # 既存のコード
            ...
            
        finally:
            log_memory_usage()
            logger.info(f"シーン検出完了: {video_path}")

    @log_exceptions
    def _merge_scene_boundaries(self, scene_data: list, transcripts: dict, video_path: str = None) -> list:
        """シーン境界を統合する"""
        logger.info("シーン境界統合開始")
        logger.debug(f"入力データ: scene_data={scene_data}, transcripts={transcripts}")
        
        try:
            # 既存のコード
            ...
            
        finally:
            logger.info("シーン境界統合完了")

    @log_exceptions
    def _create_video_nodes(self, video_path: str, all_boundaries: list, scene_data: list, transcripts: dict, 
                           keyframes_dir: str, preview_dir: str) -> List[VideoNode]:
        """ビデオノードを生成する"""
        logger.info("ビデオノード生成開始")
        logger.debug(f"入力データ: boundaries={all_boundaries}, scene_data={scene_data}")
        
        try:
            # 既存のコード
            ...
            
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
            if matching_scene.get("representative_frame"):
                node.keyframe_path = os.path.relpath(matching_scene["representative_frame"]["path"], 
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

def analyze_scene_context(self, node: VideoNode, all_nodes: list, transcripts: list) -> dict:
    """シーンの文脈を分析（生成AI）"""
    try:
        # Gemini Clientの初期化
        gemini_client = GeminiClient()
        
        # AIによる分析
        context_analysis = gemini_client.analyze_scene_context(
            node.transcript,
            node.keyframe_path if os.path.exists(node.keyframe_path) else None
        )
        
        if context_analysis:
            logger.info(f"AIによるシーン分析結果: {context_analysis}")
            return context_analysis
        
        # 分析結果がない場合はデフォルト値を返す
        logger.warning("AIによる分析に失敗したため、デフォルト値を返します")
        return {
            "location_type": "不明",
            "estimated_time_of_day": "不明",
            "weather_conditions": "不明",
            "key_activities": [],
            "emotional_tone": "中立",
            "narrative_purpose": "情報提供"
        }
        
    except Exception as e:
        logger.error(f"AIによるシーン分析中にエラー: {str(e)}")
        # エラー時はデフォルト値を返す
        return {
            "location_type": "不明",
            "estimated_time_of_day": "不明",
            "weather_conditions": "不明",
            "key_activities": [],
            "emotional_tone": "中立",
            "narrative_purpose": "情報提供"
        }
    
def generate_editing_suggestions(self, node: VideoNode, all_nodes: list) -> dict:
    """編集提案を生成（生成AI）"""
    try:
        # Gemini Clientの初期化
        gemini_client = GeminiClient()
        
        # ノードデータを準備
        node_data = {
            "transcript": node.transcript,
            "time_in": node.time_in,
            "time_out": node.time_out,
            "context_analysis": node.context_analysis
        }
        
        # AIによる編集提案生成
        suggestions = gemini_client.generate_editing_suggestions(node_data)
        
        if suggestions:
            logger.info(f"AIによる編集提案結果: {suggestions}")
            return suggestions
        
        # 生成に失敗した場合はデフォルト値を返す
        logger.warning("AIによる編集提案生成に失敗したため、デフォルト値を返します")
        return {
            "highlight_worthy": False,
            "potential_cutpoint": False,
            "b_roll_opportunity": "",
            "audio_considerations": ""
        }
        
    except Exception as e:
        logger.error(f"AIによる編集提案生成中にエラー: {str(e)}")
        # エラー時はデフォルト値を返す
        return {
            "highlight_worthy": False,
            "potential_cutpoint": False,
            "b_roll_opportunity": "",
            "audio_considerations": ""
        }
        
def generate_video_summary(self, transcripts: list, nodes: list) -> dict:
    """動画全体の要約を生成（生成AI）"""
    try:
        # Gemini Clientの初期化
        gemini_client = GeminiClient()
        
        # AIによるサマリー生成
        summary = gemini_client.generate_video_summary(transcripts, nodes)
        
        if summary:
            logger.info("AIによる動画要約生成完了")
            return summary
        
        # 生成に失敗した場合はデフォルト値を返す
        logger.warning("AIによる要約生成に失敗したため、デフォルト値を返します")
        return {
            "title": "動画タイトル",
            "overview": "AIによる要約生成に失敗しました",
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
            "gopro_start_time": ""
        }
        
    except Exception as e:
        logger.error(f"AIによる動画要約生成中にエラー: {str(e)}")
        # エラー時はデフォルト値を返す
        return {
            "title": "動画タイトル",
            "overview": f"エラー: {str(e)}",
            "topics": [],
            "scene_count": len(nodes),
            "total_duration": nodes[-1].time_out if nodes else 0
        }
    
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
            "transcript": node.transcript,
            "description": node.description,
            "keyframe_path": node.keyframe_path,
            "preview_path": node.preview_path,
            "duration": node.time_out - node.time_in
        }
        scenes.append(scene)
        
        output_file = os.path.join(output_dir, "nodes.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_path": video_path,
                    "summary": summary,
                    "scenes": scenes,
                    "completed": completed,
                    "last_update": datetime.now().isoformat()
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info(f"結果を保存しました: {output_file} (completed: {completed})")
    
    def update_status(self, stage: str, message: str):
        """UIとログに状態を更新（スレッドセーフ）"""
        logger.info(f"{stage}: {message}")
        self.root.after(0, lambda: self.status_label.config(text=f"{stage}: {message}"))
    
    def cancel_processing(self):
        """処理をキャンセル"""
        if self.processing:
            self.processing = False
            self.update_status("キャンセル", "処理をキャンセルしました")
            # UIの更新
            self.open_button.config(state="normal")
            self.open_folder_button.config(state="normal")
            self.process_button.config(state="normal")
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
        
        # プレビュー再生用の変数を初期化
        self.current_preview_path = abs_file_path
        self.is_playing = False
        self.update_timer = None
        
        # 動画の長さを取得
        try:
            media = self.vlc_instance.media_new(abs_file_path)
            media.parse()
            self.current_duration = media.get_duration() / 1000.0  # ミリ秒を秒に変換
        except Exception as e:
            logger.error(f"動画の長さ取得に失敗: {str(e)}")
            self.current_duration = 0
            
        # 時間表示を更新
        self.time_label.config(text=f"再生時間: 00:00 / {self.format_time(self.current_duration)}")
        
        # 詳細情報ウィンドウを作成
        details_window = tk.Toplevel(self.root)
        details_window.title(f"詳細情報 - {os.path.basename(abs_file_path)}")
        details_window.geometry("1200x800")  # ウィンドウを大きくする
        
        # ノードファイルのパスを取得
        base_name = os.path.splitext(os.path.basename(abs_file_path))[0]
        output_dir = os.path.join(os.path.dirname(abs_file_path), f"video_nodes_{base_name}")
        nodes_file = os.path.join(output_dir, "nodes.json")
        
        if not os.path.exists(nodes_file):
            ttk.Label(details_window, text="処理結果が見つかりません").pack(padx=10, pady=10)
            return
        
        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # メインフレーム（左右に分割）
            main_paned = ttk.PanedWindow(details_window, orient=tk.HORIZONTAL)
            main_paned.pack(fill="both", expand=True, padx=10, pady=10)
            
            # 左側：サマリー情報
            left_frame = ttk.Frame(main_paned)
            # 右側：シーン詳細
            right_frame = ttk.Frame(main_paned)
            
            main_paned.add(left_frame, weight=1)
            main_paned.add(right_frame, weight=2)
            
            # 新フォーマットと旧フォーマットの両方に対応
            summary = data.get("summary", {})
            scene_nodes = data.get("scenes", data.get("nodes", []))
            
            #----- 左側：サマリー情報 -----#
            # ヘッダー（動画タイトルと概要）
            header_frame = ttk.LabelFrame(left_frame, text="動画概要")
            header_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(header_frame, text=f"ファイル名: {os.path.basename(abs_file_path)}", font=("", 12, "bold")).pack(anchor="w", padx=5, pady=5)
            
            if isinstance(summary, dict):
                if "title" in summary:
                    ttk.Label(header_frame, text=f"タイトル: {summary.get('title', '不明')}", font=("", 11)).pack(anchor="w", padx=5, pady=2)
                
                if "overview" in summary:
                    ttk.Label(header_frame, text="概要:", font=("", 10, "bold")).pack(anchor="w", padx=5, pady=2)
                    overview_text = tk.Text(header_frame, height=6, width=40, wrap="word")
                    overview_text.insert("1.0", summary.get('overview', '不明'))
                    overview_text.config(state="disabled")
                    overview_text.pack(fill="x", padx=5, pady=2)
            
            # 基本情報フレーム
            info_frame = ttk.LabelFrame(left_frame, text="基本情報")
            info_frame.pack(fill="x", padx=5, pady=5)
            
            # 2列のグリッドレイアウト
            if isinstance(summary, dict):
                row = 0
                
                # シーン数と総時間
                ttk.Label(info_frame, text="シーン数:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                ttk.Label(info_frame, text=f"{len(scene_nodes)}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                row += 1
                
                if "total_duration" in summary:
                    ttk.Label(info_frame, text="総時間:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('total_duration', 0):.2f}秒").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "gopro_start_time" in summary:
                    ttk.Label(info_frame, text="撮影開始時間:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('gopro_start_time', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "topics" in summary and summary["topics"]:
                    ttk.Label(info_frame, text="主要トピック:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{', '.join(summary.get('topics', []))}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "filming_date" in summary and summary["filming_date"]:
                    ttk.Label(info_frame, text="撮影日:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('filming_date', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "location" in summary and summary["location"]:
                    ttk.Label(info_frame, text="撮影場所:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('location', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "weather" in summary and summary["weather"]:
                    ttk.Label(info_frame, text="天候:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('weather', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "purpose" in summary and summary["purpose"]:
                    ttk.Label(info_frame, text="目的:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('purpose', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "transportation" in summary and summary["transportation"]:
                    ttk.Label(info_frame, text="移動手段:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('transportation', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "starting_point" in summary and summary["starting_point"]:
                    ttk.Label(info_frame, text="出発地点:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('starting_point', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
                
                if "destination" in summary and summary["destination"]:
                    ttk.Label(info_frame, text="目的地:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
                    ttk.Label(info_frame, text=f"{summary.get('destination', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    row += 1
            
            # 処理情報
            ttk.Label(info_frame, text="処理状態:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(info_frame, text=f"{'完了' if data.get('completed') else '未完了'}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
            row += 1
            
            ttk.Label(info_frame, text="最終更新:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(info_frame, text=f"{data.get('last_update', '不明')}").grid(row=row, column=1, sticky="w", padx=5, pady=2)
            
            # サムネイル表示エリア（最初のキーフレーム）
            preview_frame = ttk.LabelFrame(left_frame, text="サムネイル")
            preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
            
            # 最初のシーンのキーフレームを取得
            if scene_nodes and "keyframe_path" in scene_nodes[0]:
                try:
                    keyframe_path = os.path.join(os.path.dirname(abs_file_path), scene_nodes[0]["keyframe_path"])
                    if os.path.exists(keyframe_path):
                        # PILでイメージを読み込み
                        image = Image.open(keyframe_path)
                        # アスペクト比を保持しながらリサイズ（大きく表示）
                        image.thumbnail((350, 250))
                        photo = ImageTk.PhotoImage(image)
                        
                        # 画像を表示（クリック可能）
                        image_label = ttk.Label(preview_frame)
                        image_label.image = photo  # 参照を保持
                        image_label.configure(image=photo)
                        image_label.pack(padx=5, pady=5, fill="both", expand=True)
                        
                        # クリックイベントを追加
                        image_label.bind("<Button-1>", lambda e, v=abs_file_path, t=0: self.play_video_segment(v, t))
                except Exception as e:
                    ttk.Label(preview_frame, text=f"サムネイル表示エラー: {str(e)}").pack(padx=5, pady=5)
            else:
                ttk.Label(preview_frame, text="サムネイルがありません").pack(padx=5, pady=5)
            
            # 動画再生ボタン
            ttk.Button(
                left_frame,
                text="▶ 動画全体を再生",
                command=lambda: self.play_video_segment(abs_file_path, 0)
            ).pack(fill="x", padx=5, pady=10)
            
            #----- 右側：シーン詳細 -----#
            # シーン情報（スクロール可能）
            scenes_label = ttk.Label(right_frame, text="シーン一覧", font=("", 12, "bold"))
            scenes_label.pack(anchor="w", padx=5, pady=5)
            
            # キャンバスとスクロールバー
            canvas = tk.Canvas(right_frame)
            scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # シーン情報（アコーディオン形式）
            scene_frames = []  # 各シーンのフレームを保持
            scene_contents = []  # 各シーンの詳細コンテンツを保持
            expanded = [False] * len(scene_nodes)  # 各シーンの展開状態
            
            for i, node in enumerate(scene_nodes):
                scene_id = node.get("scene_id", i)
                
                # シーンフレーム（アコーディオンヘッダー）
                scene_frame = ttk.Frame(scrollable_frame)
                scene_frame.pack(fill="x", padx=5, pady=2)
                scene_frames.append(scene_frame)
                
                # ヘッダー（クリック可能）
                header_frame = ttk.Frame(scene_frame, style="Card.TFrame")
                header_frame.pack(fill="x")
                
                # 時間情報のフォーマット
                time_in = node['time_in']
                time_out = node['time_out']
                duration = time_out - time_in
                time_info = f"{time_in:.1f}秒 - {time_out:.1f}秒 ({duration:.1f}秒)"
                
                # 展開ボタン
                expand_btn = ttk.Button(
                    header_frame,
                    text="▼" if expanded[i] else "▶",
                    width=2,
                    command=lambda idx=i: self.toggle_scene_details(idx, scene_contents, expanded, scene_frames)
                )
                expand_btn.pack(side="left", padx=(0, 5))
                
                # シーン番号
                ttk.Label(header_frame, text=f"シーン {scene_id + 1}", font=("", 10, "bold")).pack(side="left", padx=5)
                
                # 時間情報
                ttk.Label(header_frame, text=time_info).pack(side="left", padx=5)
                
                # トランスクリプトの先頭を表示（存在する場合）
                if node.get("transcript"):
                    transcript = node["transcript"]
                    # 短く切り詰める
                    short_transcript = transcript[:30] + "..." if len(transcript) > 30 else transcript
                    ttk.Label(header_frame, text=short_transcript).pack(side="left", padx=5)
                
                # ヘッダー全体をクリック可能に
                header_frame.bind("<Button-1>", lambda e, idx=i: self.toggle_scene_details(idx, scene_contents, expanded, scene_frames))
                
                # シーン詳細コンテンツ（初期状態は非表示）
                content_frame = ttk.Frame(scene_frame)
                if expanded[i]:
                    content_frame.pack(fill="x", padx=10, pady=5)
                scene_contents.append(content_frame)
                
                # 詳細情報の作成（非表示状態）
                # 左右に分割
                details_paned = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
                details_paned.pack(fill="x", expand=True, padx=5, pady=5)
                
                # 左：テキスト情報
                text_frame = ttk.Frame(details_paned)
                # 右：キーフレーム
                image_frame = ttk.Frame(details_paned)
                
                details_paned.add(text_frame, weight=3)
                details_paned.add(image_frame, weight=1)
                
                # トランスクリプト
                if node.get("transcript"):
                    transcript_label = ttk.Label(text_frame, text="トランスクリプト:", font=("", 9, "bold"))
                    transcript_label.pack(anchor="w", pady=(5, 0))
                    
                    transcript_text = tk.Text(text_frame, height=3, width=50, wrap="word")
                    transcript_text.insert("1.0", node["transcript"])
                    transcript_text.config(state="disabled")
                    transcript_text.pack(fill="x", pady=2)
                else:
                    ttk.Label(text_frame, text="トランスクリプト: なし", font=("", 9)).pack(anchor="w", pady=2)
                
                # 映像分析（説明）
                if node.get("description"):
                    description_label = ttk.Label(text_frame, text="映像分析:", font=("", 9, "bold"))
                    description_label.pack(anchor="w", pady=(5, 0))
                    
                    description_text = tk.Text(text_frame, height=3, width=50, wrap="word")
                    description_text.insert("1.0", node["description"])
                    description_text.config(state="disabled")
                    description_text.pack(fill="x", pady=2)
                
                # 文脈分析情報
                if node.get("context_analysis"):
                    context_frame = ttk.LabelFrame(text_frame, text="文脈分析")
                    context_frame.pack(fill="x", pady=5)
                    
                    context = node["context_analysis"]
                    # グリッドレイアウトで整理
                    row = 0
                    if "location_type" in context:
                        ttk.Label(context_frame, text="場所の種類:").grid(row=row, column=0, sticky="w", padx=5, pady=1)
                        ttk.Label(context_frame, text=context.get("location_type", "不明")).grid(row=row, column=1, sticky="w", padx=5, pady=1)
                        row += 1
                    
                    if "estimated_time_of_day" in context:
                        ttk.Label(context_frame, text="推定時刻:").grid(row=row, column=0, sticky="w", padx=5, pady=1)
                        ttk.Label(context_frame, text=context.get("estimated_time_of_day", "不明")).grid(row=row, column=1, sticky="w", padx=5, pady=1)
                        row += 1
                    
                    if "weather_conditions" in context:
                        ttk.Label(context_frame, text="天候状態:").grid(row=row, column=0, sticky="w", padx=5, pady=1)
                        ttk.Label(context_frame, text=context.get("weather_conditions", "不明")).grid(row=row, column=1, sticky="w", padx=5, pady=1)
                        row += 1
                    
                    if "key_activities" in context and context["key_activities"]:
                        ttk.Label(context_frame, text="主要な活動:").grid(row=row, column=0, sticky="w", padx=5, pady=1)
                        ttk.Label(context_frame, text=", ".join(context.get("key_activities", []))).grid(row=row, column=1, sticky="w", padx=5, pady=1)
                        row += 1
                
                # アクションボタンフレーム
                action_frame = ttk.Frame(text_frame)
                action_frame.pack(anchor="w", pady=5)
                
                # 再生ボタン
                play_button = ttk.Button(
                    action_frame, 
                    text="▶ シーン再生", 
                    command=lambda v=abs_file_path, t=node["time_in"]: self.play_video_segment(v, t)
                )
                play_button.pack(side="left", padx=5)
                
                # シーン書き出しボタン
                extract_button = ttk.Button(
                    action_frame, 
                    text="📋 シーン保存", 
                    command=lambda v=abs_file_path, s=node["time_in"], e=node["time_out"]: self.extract_video_segment(v, s, e)
                )
                extract_button.pack(side="left", padx=5)
                
                # キーフレーム画像（右側）
                if node.get("keyframe_path"):
                    try:
                        keyframe_path = os.path.join(os.path.dirname(abs_file_path), node["keyframe_path"])
                        if os.path.exists(keyframe_path):
                            # PILでイメージを読み込み
                            image = Image.open(keyframe_path)
                            # アスペクト比を保持しながらリサイズ
                            image.thumbnail((200, 150))
                            photo = ImageTk.PhotoImage(image)
                            
                            # 画像を表示（クリック可能）
                            image_label = ttk.Label(image_frame)
                            image_label.image = photo  # 参照を保持
                            image_label.configure(image=photo)
                            image_label.pack(padx=5, pady=5)
                            
                            # クリックイベントを追加
                            image_label.bind("<Button-1>", lambda e, v=abs_file_path, t=node["time_in"]: self.play_video_segment(v, t))
                            
                            # ホバー時のカーソル変更
                            image_label.bind("<Enter>", lambda e: e.widget.configure(cursor="hand2"))
                            image_label.bind("<Leave>", lambda e: e.widget.configure(cursor=""))
                            
                            # キーフレームラベル
                            ttk.Label(image_frame, text="キーフレーム画像").pack(pady=(0, 5))
                    except Exception as e:
                        ttk.Label(image_frame, text=f"画像表示エラー: {str(e)}").pack(padx=5, pady=5)
                
                # 区切り線（シーン間）
                ttk.Separator(scrollable_frame, orient="horizontal").pack(fill="x", pady=2)
            
        except Exception as e:
            ttk.Label(details_window, text=f"エラー: {str(e)}").pack(padx=10, pady=10)
            logger.error(f"詳細表示中にエラー: {str(e)}")
    
    def toggle_scene_details(self, index, content_frames, expanded, scene_frames):
        """シーン詳細の表示/非表示を切り替え"""
        # 状態を反転
        expanded[index] = not expanded[index]
        
        # 展開ボタンのテキストを更新
        btn = scene_frames[index].winfo_children()[0].winfo_children()[0]
        btn.config(text="▼" if expanded[index] else "▶")
        
        # コンテンツの表示/非表示を切り替え
        if expanded[index]:
            content_frames[index].pack(fill="x", padx=10, pady=5)
        else:
            content_frames[index].pack_forget()
    
    def play_video_segment(self, video_path, start_time):
        """指定時間から動画を再生"""
        try:
            # 絶対パスに変換
            abs_video_path = os.path.abspath(video_path)
            logger.info(f"再生リクエスト - 元のパス: {video_path}")
            logger.info(f"再生リクエスト - 絶対パス: {abs_video_path}")
            
            # ファイルの存在確認
            if not os.path.exists(abs_video_path):
                logger.error(f"ファイルが存在しません: {abs_video_path}")
                messagebox.showerror("エラー", f"ファイルが見つかりません:\n{abs_video_path}")
                return
                
            # プレビュー動画のパスを取得
            base_name = os.path.splitext(os.path.basename(abs_video_path))[0]
            output_dir = os.path.join(os.path.dirname(abs_video_path), f"video_nodes_{base_name}")
            preview_dir = os.path.join(output_dir, "previews")
            
            # ノードファイルから該当シーンのプレビューパスを取得
            nodes_file = os.path.join(output_dir, "nodes.json")
            preview_path = None
            
            if os.path.exists(nodes_file):
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    scenes = data.get("scenes", data.get("nodes", []))  # 新旧フォーマット対応
                    for scene in scenes:
                        if abs(scene["time_in"] - start_time) < 0.1:  # 開始時間が一致するシーンを探す
                            if scene.get("preview_path"):
                                preview_path = os.path.join(os.path.dirname(abs_video_path), scene["preview_path"])
                                logger.info(f"プレビューパスを検出: {preview_path}")
                                break
            
            if preview_path and os.path.exists(preview_path):
                # プレビュー動画を再生
                logger.info(f"プレビュー動画を再生: {preview_path}")
                if os.name == 'nt':  # Windows
                    subprocess.Popen(['start', '', preview_path], shell=True)
                elif os.name == 'posix':  # macOS/Linux
                    if sys.platform == 'darwin':
                        subprocess.Popen(['open', preview_path])
                    else:
                        subprocess.Popen(['xdg-open', preview_path])
                logger.info(f"プレビュー動画再生: {preview_path}")
                return
            
            # プレビューがない場合は元の動画を時間指定で再生
            logger.info(f"元の動画を再生: {abs_video_path}, 開始時間: {start_time}秒")
            if os.name == 'nt':  # Windows
                # VLCがインストールされている場合の時間指定
                try:
                    vlc_path = self.find_vlc_path()
                    if vlc_path:
                        # VLCでタイムスタンプ指定して再生
                        command = [vlc_path, "--start-time", str(int(start_time)), abs_video_path]
                        logger.info(f"VLCコマンド: {' '.join(command)}")
                        subprocess.Popen(command)
                        logger.info(f"VLCで動画再生: {abs_video_path}, 開始時間: {start_time}秒")
                        return
                except Exception as e:
                    logger.warning(f"VLCでの再生に失敗しました: {str(e)}")
                
                # VLCが使えない場合はデフォルトプレーヤーで開く
                logger.info(f"デフォルトプレーヤーで再生: {abs_video_path}")
                os.startfile(abs_video_path)  # Windowsの関連付けられたプレーヤーで開く
            elif os.name == 'posix':  # macOS または Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.Popen(['open', abs_video_path])
                else:  # Linux
                    # VLCが利用可能か確認
                    try:
                        subprocess.run(["which", "vlc"], check=True, capture_output=True)
                        subprocess.Popen(['vlc', '--start-time', str(int(start_time)), abs_video_path])
                        return
                    except subprocess.CalledProcessError:
                        # VLCがない場合はデフォルトプレーヤーで開く
                        subprocess.Popen(['xdg-open', abs_video_path])
            
            logger.info(f"動画再生: {abs_video_path}, 開始時間: {start_time}秒")
        except Exception as e:
            logger.error(f"動画再生エラー: {str(e)}")
            messagebox.showerror("エラー", f"動画の再生に失敗しました: {str(e)}")
    
    def find_vlc_path(self):
        """VLCのパスを探す（Windows用）"""
        # 一般的なVLCのインストールパス
        possible_paths = [
            r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # レジストリからVLCのパスを取得（Windowsのみ）
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\VideoLAN\VLC") as key:
                install_dir, _ = winreg.QueryValueEx(key, "InstallDir")
                vlc_path = os.path.join(install_dir, "vlc.exe")
                if os.path.exists(vlc_path):
                    return vlc_path
        except Exception:
            pass
        
        return None

    def extract_video_segment(self, video_path, start_time, end_time):
        """動画の一部を切り出して保存"""
        try:
            # 保存先を選択
            output_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")],
                initialdir=os.path.dirname(video_path),
                initialfile=f"{os.path.splitext(os.path.basename(video_path))[0]}_scene_{start_time:.1f}-{end_time:.1f}.mp4"
            )
            
            if not output_path:
                return
            
            # FFmpegで切り出し
            duration = end_time - start_time
            
            # 正確なシーン切り出しのためのコマンド（再エンコード）
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),  # 入力ファイルの前に-ss を配置（より正確なシーク）
                "-i", video_path,
                "-t", str(duration),    # 切り出す長さ
                "-c:v", "libx264",      # 再エンコードして正確なフレームから開始
                "-c:a", "aac",          # 音声も再エンコード
                "-preset", "fast",      # 処理速度と品質のバランス
                "-pix_fmt", "yuv420p",  # 互換性のあるピクセルフォーマット
                "-map_metadata", "-1",  # メタデータを取り除く
                output_path
            ]
            
            # プログレスダイアログ
            progress_window = tk.Toplevel(self.root)
            progress_window.title("シーン書き出し中")
            progress_window.geometry("400x150")
            progress_window.resizable(False, False)
            
            info_text = f"シーンを書き出し中...\n開始: {start_time:.2f}秒 - 終了: {end_time:.2f}秒 (長さ: {duration:.2f}秒)"
            ttk.Label(progress_window, text=info_text).pack(padx=10, pady=10)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
            progress_bar.pack(fill="x", padx=10, pady=10)
            
            # キャンセルボタン
            cancel_button = ttk.Button(
                progress_window, 
                text="キャンセル", 
                command=lambda: progress_window.after(0, progress_window.destroy)
            )
            cancel_button.pack(pady=10)
            
            # 処理実行
            def execute_command():
                try:
                    process = subprocess.Popen(
                        command, 
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        bufsize=1
                    )
                    
                    # FFmpegの進捗をパースして表示
                    progress_var.set(10)  # 初期進捗
                    
                    # プロセスが終了するか、ウィンドウが閉じられるまで待機
                    while process.poll() is None:
                        if not progress_window.winfo_exists():
                            # ウィンドウが閉じられた場合、プロセスを終了
                            process.terminate()
                            return
                        
                        # 一定間隔で進捗を更新
                        progress_var.set(min(progress_var.get() + 1, 95))
                        progress_window.update_idletasks()
                        time.sleep(0.1)
                    
                    # 正常終了した場合
                    if process.returncode == 0:
                        progress_var.set(100)
                        progress_window.after(500, progress_window.destroy)
                        messagebox.showinfo("完了", f"シーンを保存しました:\n{output_path}")
                    else:
                        error_output = process.stderr.read() if hasattr(process, 'stderr') else "不明なエラー"
                        logger.error(f"FFmpegエラー: {error_output}")
                        messagebox.showerror("エラー", f"シーンの書き出しに失敗しました")
                        progress_window.destroy()
                except Exception as e:
                    logger.error(f"シーン書き出しエラー: {str(e)}")
                    messagebox.showerror("エラー", f"シーンの書き出しに失敗しました: {str(e)}")
                    progress_window.destroy()
            
            # 別スレッドで実行
            threading.Thread(target=execute_command, daemon=True).start()
            
            logger.info(f"シーン書き出し: {video_path}, {start_time}秒-{end_time}秒 -> {output_path}")
        except Exception as e:
            logger.error(f"シーン書き出しエラー: {str(e)}")
            messagebox.showerror("エラー", f"シーンの書き出しに失敗しました: {str(e)}")

    def update_preview(self, event=None):
        """選択されたファイルのプレビューを表示"""
        selection = self.file_list.selection()
        if not selection:
            return
        
        item_id = selection[0]
        file_path = self.file_list.item(item_id, "values")[0]
        
        # プレビュー再生用の変数を初期化
        self.current_preview_path = file_path
        self.is_playing = False
        
        # 既に再生中なら停止
        if self.player:
            self.stop_preview()
        
        # プレビュー画像の更新
        try:
            # 動画の長さを取得
            media = self.vlc_instance.media_new(file_path)
            media.parse()
            self.current_duration = media.get_duration() / 1000.0  # ミリ秒を秒に変換
            
            # 時間表示を更新
            self.time_label.config(text=f"再生時間: 00:00 / {self.format_time(self.current_duration)}")
            
            # プレビューラベルを更新
            self.preview_label.config(text=f"選択中: {os.path.basename(file_path)}")
            
            # VLCプレーヤーに設定
            if self.vlc_instance and self.player:
                if os.name == 'nt':  # Windows
                    self.player.set_hwnd(self.video_frame.winfo_id())
                else:  # Linux/MacOS
                    self.player.set_xwindow(self.video_frame.winfo_id())
            
            # 詳細表示ボタンを有効化
            self.view_button.config(state="normal")
            
        except Exception as e:
            logger.error(f"プレビュー更新エラー: {str(e)}")
            self.preview_label.config(text="プレビューの読み込みに失敗しました")
            self.time_label.config(text="再生時間: --:-- / --:--")
            self.current_duration = 0
    
    def on_closing(self):
        """アプリケーション終了時の処理"""
        if self.processing:
            if messagebox.askyesno("確認", "処理中ですが、終了しますか？"):
                self.processing = False
                self.root.destroy()
        else:
            self.root.destroy()
    
    def generate_preview_clip(self, video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
        """低解像度のプレビュー動画を生成"""
        try:
            duration = end_time - start_time
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-vf", "scale=320:180",  # 180p（より小さく）
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "35",  # より高圧縮
                "-tune", "fastdecode",  # 再生速度優先
                "-profile:v", "baseline",  # 互換性重視
                "-level", "3.0",
                "-maxrate", "500k",  # ビットレート制限
                "-bufsize", "1000k",
                "-c:a", "aac",
                "-b:a", "32k",  # より低音質
                "-ac", "1",  # モノラル
                "-ar", "22050",  # サンプリングレート低下
                "-movflags", "+faststart",
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"プレビュー動画生成成功: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"プレビュー動画生成エラー: {str(e)}")
            return False

    def extract_keyframe(self, video_path: str, timestamp: float, output_path: str) -> bool:
        """指定時間のキーフレームを抽出"""
        try:
            command = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",  # JPEGの品質（2は高品質）
                "-vf", "scale=320:180",  # サムネイルサイズ
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"キーフレーム抽出成功: {output_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"キーフレーム抽出エラー: {str(e)}")
            return False

    def enhanced_video_processing(self, video_path: str) -> List[dict]:
        """
        OpenCVを使用したヒストグラム差分と適応型閾値による高度なシーン検出処理を行う。
        
        動画を解析して意味のあるシーン境界を検出し、各シーンのキーフレームと分析情報を生成する。
        この処理は2段階で実行される：
        1. 動画全体をサンプリングして差分データを収集し、適応型閾値を算出
        2. 算出した閾値を使用してシーン境界を検出
        
        Args:
            video_path: 処理する動画ファイルのパス
            
        Returns:
            List[dict]: 検出されたシーンのリスト。各シーンは開始・終了時間、キーフレームなどの情報を含む。
        """
        try:
            # 出力ディレクトリの準備
            base_name, output_dir, keyframes_dir = self._prepare_output_directories(video_path)
            
            # 動画ファイルを開いて基本情報を取得
            cap, fps, total_frames, total_duration, sample_rate = self._open_video_and_get_info(video_path)
            if not cap:
                return self._create_fallback_scene(total_duration=0)
            
            # 第1パス: 差分値を収集して適応型閾値を算出
            hist_threshold, pixel_threshold = self._calculate_adaptive_thresholds(cap, fps, sample_rate)
            
            # 第2パス: シーン境界を検出
            scene_boundaries = self._detect_scene_boundaries(
                video_path, hist_threshold, pixel_threshold, fps, total_duration, sample_rate
            )
            
            # シーンの選択処理（シーン数が多い場合は均等に選択）
            selected_indices = self._select_representative_scenes(scene_boundaries)
            
            # シーンごとにキーフレームを抽出し、データを生成
            scene_data = self._generate_scene_data(
                video_path, scene_boundaries, selected_indices, keyframes_dir
            )
            
            # 結果の検証と調整
            scene_data = self._validate_scene_data(scene_data, total_duration)
            
            return scene_data

        except Exception as e:
            logger.error(f"映像処理エラー: {str(e)}", exc_info=True)
            # エラー時は動画全体を1シーンとして返す
            return self._create_fallback_scene(
                total_duration=total_frames / fps if 'total_frames' in locals() and 'fps' in locals() else 240.0
            )

    def _prepare_output_directories(self, video_path: str) -> tuple:
        """
        出力ディレクトリを準備する
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            tuple: (base_name, output_dir, keyframes_dir)
        """
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
        keyframes_dir = os.path.join(output_dir, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)
        return base_name, output_dir, keyframes_dir

    def _open_video_and_get_info(self, video_path: str) -> tuple:
        """
        動画ファイルを開き、基本情報を取得する
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            tuple: (cap, fps, total_frames, total_duration, sample_rate)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"ビデオファイルを開けません: {video_path}")
            return None, 0, 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        sample_rate = max(1, int(fps / 4))  # より細かいサンプリング（fps/4）
        
        logger.info(f"動画情報: 総フレーム数={total_frames}, FPS={fps}, 推定時間={total_duration:.2f}秒")
        return cap, fps, total_frames, total_duration, sample_rate

    def _calculate_adaptive_thresholds(self, cap, fps, sample_rate):
        """
        動画をサンプリングして差分データを収集し、適応型閾値を算出する
        
        Args:
            cap: OpenCVのVideoCapture
            fps: フレームレート
            sample_rate: サンプリングレート
            
        Returns:
            tuple: (hist_threshold, pixel_threshold)
        """
        prev_frame = None
        prev_hist = None
        frame_count = 0
        hist_diffs = []
        pixel_diffs = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 先頭に移動
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # グレースケール変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # ヒストグラム計算
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_hist is not None and prev_frame is not None:
                    # ヒストグラム相関（1に近いほど類似）
                    hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    hist_diffs.append(hist_diff)
                    
                    # ピクセル差分（値が大きいほど差異が大きい）
                    pixel_diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(pixel_diff)
                    pixel_diffs.append(mean_diff)
                
                prev_hist = hist
                prev_frame = gray.copy()
            
            frame_count += 1
            
            # 処理が中断された場合
            if not self.processing:
                cap.release()
                return 0.9, 20.0  # デフォルト値
        
        # 適応型閾値の計算（より緩い閾値を設定）
        if len(hist_diffs) < 2 or len(pixel_diffs) < 2:
            logger.warning("差分データが不足しています。デフォルト閾値を使用します。")
            hist_threshold = 0.9  # デフォルト値（緩和）
            pixel_threshold = 20.0  # デフォルト値（緩和）
        else:
            hist_diffs = np.array(hist_diffs)
            pixel_diffs = np.array(pixel_diffs)
            
            hist_mean = np.mean(hist_diffs)
            hist_std = np.std(hist_diffs)
            pixel_mean = np.mean(pixel_diffs)
            pixel_std = np.std(pixel_diffs)
            
            # 閾値を緩和: ヒストグラム閾値を0.9以下、ピクセル差分閾値を低めに
            hist_threshold = min(0.95, hist_mean - 0.5 * hist_std)  # 標準偏差の0.5倍（さらに緩和）
            # ピクセル差分閾値を調整
            pixel_threshold = max(5.0, pixel_mean + 0.5 * pixel_std)  # 標準偏差の0.5倍（さらに緩和）
            
            logger.info(f"適応型閾値（より緩和版）: ヒストグラム={hist_threshold:.3f} (平均={hist_mean:.3f}, 標準偏差={hist_std:.3f})")
            logger.info(f"適応型閾値（より緩和版）: ピクセル差分={pixel_threshold:.3f} (平均={pixel_mean:.3f}, 標準偏差={pixel_std:.3f})")
        
        return hist_threshold, pixel_threshold

    def _detect_scene_boundaries(self, video_path, hist_threshold, pixel_threshold, fps, total_duration, sample_rate):
        """
        シーン境界を検出する
        
        Args:
            video_path: 動画ファイルのパス
            hist_threshold: ヒストグラム相関の閾値
            pixel_threshold: ピクセル差分の閾値
            fps: フレームレート
            total_duration: 動画の総時間
            sample_rate: サンプリングレート
            
        Returns:
            list: 検出されたシーン境界のリスト（秒単位）
        """
        scene_boundaries = [0.0]  # 最初のフレームは常に境界
        cap = cv2.VideoCapture(video_path)
        
        ret, prev_frame = cap.read()
        if not ret:
            logger.error("動画ファイルを読み込めませんでした")
            cap.release()
            return [0.0, total_duration]
        
        # グレースケールに変換
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
        
        frame_idx = 0
        skipped_frames = 0
        
        # 進捗状況の表示用
        progress_interval = max(1, int(total_duration * fps / sample_rate / 20))  # 5%ごと
        
        while True:
            # サンプリングレートに従ってフレームをスキップ
            for _ in range(sample_rate):
                ret = cap.grab()
                if not ret:
                    break
                skipped_frames += 1
            
            if not ret:
                break
                
            # デコード
            ret, frame = cap.retrieve()
            if not ret:
                break
            
            frame_idx += 1
            
            # 現在の時間（秒）
            current_time = (skipped_frames + frame_idx) / fps
            
            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ヒストグラム計算
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            
            # ヒストグラム相関を計算（1に近いほど類似）
            hist_corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            
            # ピクセル差分を計算
            frameDelta = cv2.absdiff(prev_gray, gray)
            pixel_diff = np.mean(frameDelta)
            
            # 境界条件の確認：ヒストグラム相関が低い、またはピクセル差分が高い
            if hist_corr < hist_threshold or pixel_diff > pixel_threshold:
                # 前のシーン境界から十分離れているか確認（短すぎるシーンを避ける）
                min_scene_duration = 1.0  # 最低1秒
                if len(scene_boundaries) == 0 or current_time - scene_boundaries[-1] >= min_scene_duration:
                    scene_boundaries.append(current_time)
                    logger.info(f"シーン境界検出: {current_time:.2f}秒 (相関: {hist_corr:.4f}, 差分: {pixel_diff:.2f})")
            
            # 前フレームを更新
            prev_gray = gray
            prev_hist = hist
            
            # 進捗状況の表示
            if frame_idx % progress_interval == 0:
                progress = frame_idx / (total_duration * fps / sample_rate) * 100
                logger.info(f"シーン検出進捗: {progress:.1f}% ({current_time:.1f}/{total_duration:.1f}秒)")
        
        cap.release()
        
        # 最後のフレームを境界として追加
        if scene_boundaries[-1] < total_duration:
            scene_boundaries.append(total_duration)
        
        logger.info(f"検出されたシーン境界: {len(scene_boundaries)-1}個のシーン")
        
        # シーン検出に失敗してもフォールバックを使用しない
        # 代わりに、音声分析により境界を取得する（_merge_scene_boundariesメソッド）
        
        return scene_boundaries

    def _filter_short_scenes(self, scene_boundaries, min_scene_duration=5.0, video_path=None):
        """
        短すぎるシーンを統合する
        
        Args:
            scene_boundaries: シーン境界のリスト
            min_scene_duration: 最小シーン長さ（秒）
            video_path: 動画ファイルのパス（フォールバック用）
            
        Returns:
            list: フィルタリング後のシーン境界リスト
        """
        filtered_boundaries = [scene_boundaries[0]]
        
        for i in range(1, len(scene_boundaries)):
            if scene_boundaries[i] - filtered_boundaries[-1] >= min_scene_duration:
                filtered_boundaries.append(scene_boundaries[i])
            else:
                logger.info(f"短すぎるシーンをスキップ: {filtered_boundaries[-1]:.2f} - {scene_boundaries[i]:.2f}秒")
        
        # 最後の境界を必ず含める
        if filtered_boundaries[-1] != scene_boundaries[-1]:
            filtered_boundaries.append(scene_boundaries[-1])
        
        return filtered_boundaries

    def _select_representative_scenes(self, scene_boundaries):
        """
        代表的なシーンを選択する（シーン数が多い場合は均等に選択）
        
        Args:
            scene_boundaries: シーン境界のリスト
            
        Returns:
            list: 選択されたシーンのインデックスリスト
        """
        total_scenes = len(scene_boundaries) - 1
        if total_scenes > 10:
            # 均等に10シーンを選択
            step = total_scenes / 10
            selected_indices = [int(i * step) for i in range(10)]
            selected_indices[-1] = min(selected_indices[-1], total_scenes - 1)  # 範囲を超えないように
            logger.info(f"シーン数制限: {total_scenes}から10シーンを選択")
        else:
            selected_indices = list(range(total_scenes))
        
        return selected_indices

    def _generate_scene_data(self, video_path, scene_boundaries, selected_indices, keyframes_dir):
        """
        各シーンのデータとキーフレームを生成する
        
        Args:
            video_path: 動画ファイルのパス
            scene_boundaries: シーン境界のリスト
            selected_indices: 選択されたシーンのインデックス
            keyframes_dir: キーフレーム保存ディレクトリ
            
        Returns:
            list: 生成されたシーンデータのリスト
        """
        scene_data = []
        keyframe_count = 0

        for i in selected_indices:
            start_time = scene_boundaries[i]
            end_time = scene_boundaries[i + 1]
            keyframe_time = (start_time + end_time) / 2  # シーン中間の時間
            
            logger.info(f"シーン {i+1}/{len(selected_indices)}: {start_time:.2f}-{end_time:.2f}秒 ({end_time-start_time:.2f}秒)")
            
            # キーフレーム抽出
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, keyframe_time * 1000)
            ret, frame = cap.read()
            if ret:
                keyframe_path = os.path.join(keyframes_dir, f"keyframe_{keyframe_count:04d}.jpg")
                cv2.imwrite(keyframe_path, frame)
                logger.info(f"キーフレーム保存: {keyframe_path}")

                # Geminiで映像分析
                analysis = self._analyze_keyframe(keyframe_path)
                
                scene_data.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "representative_frame": {"path": keyframe_path},
                    "ai_analysis": analysis
                })
                keyframe_count += 1
            cap.release()
        
        return scene_data

    def _analyze_keyframe(self, keyframe_path):
        """
        キーフレームをGeminiで分析する
        
        Args:
            keyframe_path: キーフレームのパス
            
        Returns:
            dict: 分析結果
        """
        try:
            analysis_result = self.gemini_client.analyze_image(keyframe_path)
            # 文字列の場合は辞書に変換
            if isinstance(analysis_result, str):
                analysis = {
                    "scene_type": "",
                    "time_of_day": "",
                    "weather": "",
                    "activities": [analysis_result.strip()]
                }
            else:
                analysis = analysis_result
        except Exception as e:
            logger.error(f"画像分析エラー: {str(e)}")
            analysis = {
                "scene_type": "",
                "time_of_day": "",
                "weather": "",
                "activities": []
            }
        
        return analysis

    def _validate_scene_data(self, scene_data, total_duration):
        """
        生成されたシーンデータを検証し、必要に応じて調整する
        
        Args:
            scene_data: 生成されたシーンデータ
            total_duration: 動画の総再生時間
            
        Returns:
            list: 調整後のシーンデータ
        """
        # 結果の検証
        if len(scene_data) == 0:
            logger.warning("シーンデータが生成されませんでした。デフォルトシーンを作成します。")
            return [self._create_fallback_scene(total_duration)]
        elif total_duration > 0 and scene_data[-1]["end_time"] < total_duration * 0.9:
            logger.warning(f"最終シーン終了時間({scene_data[-1]['end_time']:.2f})が全体時間({total_duration:.2f})より大幅に短いです")
            # 最終シーンの終了時間を全体時間に修正
            scene_data[-1]["end_time"] = total_duration

        return scene_data

    def _create_fallback_scene(self, total_duration):
        """
        フォールバック用のデフォルトシーンを作成する
        
        Args:
            total_duration: 動画の総再生時間
            
        Returns:
            dict: デフォルトシーンデータ
        """
        return {
            "start_time": 0, 
            "end_time": total_duration,
            "representative_frame": None, 
            "ai_analysis": {}
        }

    def run(self):
        """アプリケーション実行"""
        self.root.mainloop()

    def play_preview(self):
        """プレビュー動画を再生"""
        if not self.vlc_instance or not self.player:
            logger.warning("プレビュー再生できません：VLCプレーヤーが未初期化")
            messagebox.showwarning("再生エラー", "VLCプレーヤーが初期化されていません。\nVLCをインストールしてください。")
            return
            
        if not self.current_preview_path:
            logger.warning("プレビュー再生できません：ファイルが選択されていません")
            messagebox.showwarning("再生エラー", "ファイルが選択されていません。")
            return
            
        # ファイルの存在確認
        abs_path = os.path.abspath(self.current_preview_path)
        if not os.path.exists(abs_path):
            logger.error(f"ファイルが存在しません: {abs_path}")
            messagebox.showerror("再生エラー", f"ファイルが見つかりません:\n{abs_path}")
            return
        
        try:
            if not self.is_playing:
                logger.info(f"プレビュー再生開始: {abs_path}")
                
                media = self.vlc_instance.media_new(abs_path)
                self.player.set_media(media)
                result = self.player.play()
                
                if result == -1:
                    logger.error("VLCプレーヤーの再生開始に失敗しました")
                    messagebox.showerror("再生エラー", "VLCプレーヤーの再生開始に失敗しました")
                    return
                    
                self.is_playing = True
                # 再生時間の更新タイマーを開始
                self.update_playback_time()
        except Exception as e:
            logger.error(f"プレビュー再生エラー: {str(e)}")
            messagebox.showerror("再生エラー", f"プレビュー再生中にエラーが発生しました：\n{str(e)}")
            self.is_playing = False
    
    def format_time(self, seconds):
        """秒を MM:SS 形式に変換"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def stop_preview(self):
        """プレビュー動画を停止"""
        try:
            if self.player:
                logger.info("プレビュー再生停止")
                # 再生状態確認
                is_playing = self.player.is_playing()
                if is_playing:
                    self.player.stop()
                self.is_playing = False
                # タイマーを停止
                if self.update_timer:
                    self.root.after_cancel(self.update_timer)
                    self.update_timer = None
                # 時間表示をリセット
                self.time_label.config(text=f"再生時間: 00:00 / {self.format_time(self.current_duration)}")
                logger.info("プレビュー再生を停止しました")
        except Exception as e:
            logger.error(f"プレビュー停止エラー: {str(e)}")
            # エラーメッセージを表示
            messagebox.showerror("停止エラー", f"プレビューの停止中にエラーが発生しました:\n{str(e)}")
    
    def update_playback_time(self):
        """再生時間を更新"""
        if not self.player or not self.is_playing:
            return
        
        try:
            if self.player.is_playing():
                # ミリ秒を秒に変換
                current_time = self.player.get_time() / 1000.0
                self.time_label.config(text=f"再生時間: {self.format_time(current_time)} / {self.format_time(self.current_duration)}")
                # 100ミリ秒後に再度更新
                self.update_timer = self.root.after(100, self.update_playback_time)
            else:
                # 再生が終了した場合
                self.is_playing = False
                self.time_label.config(text=f"再生時間: {self.format_time(self.current_duration)} / {self.format_time(self.current_duration)}")
        except Exception as e:
            logger.error(f"再生時間更新エラー: {str(e)}")
            self.is_playing = False

if __name__ == "__main__":
    app = VideoEnhanced()
    app.run()
