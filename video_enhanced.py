import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
import threading
import json
import time
from datetime import datetime
import logging
from api_client import WhisperClient
from typing import List
import glob
import torch  # GPUメモリ管理用
import gc
from PIL import Image, ImageTk
import sys
import re

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/enhanced_{datetime.now():%Y%m%d_%H%M%S}.log", encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class VideoSummary:
    def __init__(self):
        self.title = ""  # 動画の端的なタイトル
        self.overview = ""  # 詳細な概要
        self.topics = []  # 主要トピックリスト
        self.filming_date = ""  # 撮影日
        self.location = ""  # 撮影場所
        self.weather = ""  # 天候
        self.purpose = ""  # 動画の目的
        self.transportation = ""  # 移動手段
        self.starting_point = ""  # 出発地点
        self.destination = ""  # 目的地
        self.scene_count = 0  # シーン数
        self.total_duration = 0.0  # 合計秒数
        self.gopro_start_time = ""  # 撮影開始時間
    
    def to_dict(self):
        return {
            "title": self.title,
            "overview": self.overview,
            "topics": self.topics,
            "filming_date": self.filming_date,
            "location": self.location,
            "weather": self.weather,
            "purpose": self.purpose,
            "transportation": self.transportation,
            "starting_point": self.starting_point,
            "destination": self.destination,
            "scene_count": self.scene_count,
            "total_duration": self.total_duration,
            "gopro_start_time": self.gopro_start_time
        }

class VideoNode:
    def __init__(self, time_in, time_out, transcript="", description="", keyframe_path="", preview_path=""):
        self.time_in = time_in
        self.time_out = time_out
        self.transcript = transcript
        self.description = description
        self.keyframe_path = keyframe_path
        self.preview_path = preview_path
        self.context_analysis = {
            "location_type": "",  # 場所の種類（屋内/屋外/交通機関内など）
            "estimated_time_of_day": "",  # 推定時刻（朝/昼/夕方/夜）
            "weather_conditions": "",  # 天候状態
            "key_activities": [],  # 活動リスト
            "emotional_tone": "",  # 話者の感情トーン
            "narrative_purpose": ""  # シーンの物語上の目的
        }
        self.editing_suggestions = {
            "highlight_worthy": False,  # ハイライト候補か
            "potential_cutpoint": False,  # カットポイントとして適切か
            "b_roll_opportunity": "",  # B-rollの提案
            "audio_considerations": ""  # 音声に関する注意点
        }
    
    def to_dict(self):
        return {
            "scene_id": id(self),  # オブジェクトのIDを一意のシーンIDとして使用
            "time_in": self.time_in,
            "time_out": self.time_out,
            "transcript": self.transcript,
            "description": self.description,
            "context_analysis": self.context_analysis,
            "editing_suggestions": self.editing_suggestions,
            "keyframe_path": self.keyframe_path,
            "preview_path": self.preview_path,
            "duration": self.time_out - self.time_in
        }

class VideoEnhanced:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ビデオノード生成ツール")
        self.root.geometry("1200x800")  # ウィンドウサイズを大きく
        
        # アイコンとスタイルの設定
        self.style = ttk.Style()
        self.style.configure("Custom.Treeview", rowheight=40)  # 行の高さを調整
        self.style.configure("Custom.Treeview.Heading", font=('Helvetica', 10, 'bold'))
        
        # 処理フラグ
        self.processing = False
        self.processing_thread = None
        self.selected_files = []
        self.processed_files = set()
        self.current_file_progress = 0
        
        # UIの構築
        self.build_ui()
    
    def build_ui(self):
        # メインフレーム（左右に分割）
        main_frame = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左側：ファイルリスト
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=2)
        
        # コントロールフレーム
        control_frame = ttk.LabelFrame(left_frame, text="コントロール")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # ボタンフレーム
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # ファイル選択ボタン
        self.select_button = ttk.Button(
            button_frame,
            text="📂 フォルダを選択",
            command=self.select_files
        )
        self.select_button.pack(side="left", padx=5)
        
        # 処理ボタン
        self.process_button = ttk.Button(
            button_frame,
            text="▶ 一括処理開始",
            command=self.confirm_start_processing,
            state="disabled"
        )
        self.process_button.pack(side="left", padx=5)
        
        # キャンセルボタン
        self.cancel_button = ttk.Button(
            button_frame,
            text="⏹ キャンセル",
            command=self.confirm_cancel_processing,
            state="disabled"
        )
        self.cancel_button.pack(side="left", padx=5)
        
        # オプションフレーム
        option_frame = ttk.Frame(control_frame)
        option_frame.pack(fill="x", padx=5, pady=5)
        
        # 処理済みスキップオプション
        self.resume_var = tk.BooleanVar(value=True)
        self.resume_check = ttk.Checkbutton(
            option_frame,
            text="処理済みをスキップ",
            variable=self.resume_var
        )
        self.resume_check.pack(side="left", padx=5)
        
        # ファイルリストフレーム
        list_frame = ttk.LabelFrame(left_frame, text="ファイルリスト")
        list_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # ツリービュー
        columns = ("ファイル名", "状態", "シーン数", "サムネイル")
        self.file_list = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            style="Custom.Treeview"
        )
        
        # カラム設定
        self.file_list.heading("ファイル名", text="ファイル名")
        self.file_list.heading("状態", text="状態")
        self.file_list.heading("シーン数", text="シーン数")
        self.file_list.heading("サムネイル", text="サムネイル")
        
        self.file_list.column("ファイル名", width=200)
        self.file_list.column("状態", width=100)
        self.file_list.column("シーン数", width=80)
        self.file_list.column("サムネイル", width=300)
        
        # スクロールバー
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        
        self.file_list.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # ツリービューにダブルクリックイベントを追加
        self.file_list.bind("<Double-1>", self.show_file_details)
        
        # 選択変更イベントを追加（プレビュー表示用）
        self.file_list.bind("<<TreeviewSelect>>", self.update_preview)
        
        # 右側：プレビューとログ
        right_frame = ttk.Frame(main_frame)
        main_frame.add(right_frame, weight=1)
        
        # プレビューフレーム
        preview_frame = ttk.LabelFrame(right_frame, text="プレビュー")
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # プレビュー画像
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # プログレスフレーム
        progress_frame = ttk.LabelFrame(right_frame, text="進捗状況")
        progress_frame.pack(fill="x", padx=5, pady=5)
        
        # 全体の進捗
        ttk.Label(progress_frame, text="全体の進捗:").pack(fill="x", padx=5, pady=2)
        self.total_progress_var = tk.DoubleVar()
        self.total_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.total_progress_var,
            maximum=100
        )
        self.total_progress_bar.pack(fill="x", padx=5, pady=2)
        
        # 現在のファイルの進捗
        ttk.Label(progress_frame, text="現在のファイル:").pack(fill="x", padx=5, pady=2)
        self.file_progress_var = tk.DoubleVar()
        self.file_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.file_progress_var,
            maximum=100
        )
        self.file_progress_bar.pack(fill="x", padx=5, pady=2)
        
        # ステータス表示
        self.status_label = ttk.Label(
            right_frame,
            text="フォルダを選択してください",
            wraplength=400
        )
        self.status_label.pack(fill="x", padx=5, pady=5)
    
    def select_files(self):
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
        self.resume_var.set(False)  # 再処理のためにスキップをオフに
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
                            status = "⚠ 未完了"
                except Exception as e:
                    logger.error(f"ノードファイル読み込みエラー: {str(e)}")
                    status = "⚠ エラー"

            # ツリービューにファイル情報を挿入
            item_id = self.file_list.insert("", tk.END, values=(
                os.path.basename(video_path),
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
        self.select_button.config(state="disabled")
        self.process_button.config(state="disabled")
        self.cancel_button.config(state="normal")
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
                if self.resume_var.get() and video_path in self.processed_files:
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
    
    def process_single_video(self, video_path: str):
        """個別のビデオファイルを処理"""
        try:
            # ノードリストの初期化
            nodes = []

            # 出力ディレクトリの作成
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
            keyframes_dir = os.path.join(output_dir, "keyframes")
            preview_dir = os.path.join(output_dir, "previews")
            os.makedirs(keyframes_dir, exist_ok=True)
            os.makedirs(preview_dir, exist_ok=True)

            # GoPro内部時間の取得（ファイル単位のメタデータ）
            try:
                gopro_metadata = self.extract_gopro_metadata(video_path)
            except Exception as e:
                logger.error(f"GoPro時間取得エラー: {str(e)}")
                gopro_metadata = None

            # 音声分析
            try:
                # GPUメモリをクリア
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                client = WhisperClient(
                    model_size="large",
                    compute_type="int8"
                )
                result = client.process_video(
                    video_path,
                    min_silence=0.3,  # 無音検出の感度をさらに上げる（0.3秒）
                    start_time=0.0  # ファイル単位の開始時間
                )
                
                # 音声分析の進捗を更新
                self.file_progress_var.set(30)  # 30%まで完了
                
            finally:
                # GPUメモリを解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                # WhisperClientのクリーンアップ
                if 'client' in locals():
                    del client
            
            if not self.processing:  # キャンセルチェック
                return

            # シーン分割処理
            enhanced_boundaries = []
            
            # 無音区間の検出（より短い無音も検出）
            if result.get("silent_regions"):
                for region in result["silent_regions"]:
                    if region["duration"] >= 0.5:  # 0.5秒以上の無音
                        enhanced_boundaries.extend([region["start"], region["end"]])
            
            # 音声認識結果からの境界も追加
            if result["scene_boundaries"]:
                for boundary in result["scene_boundaries"]:
                    if boundary not in enhanced_boundaries:
                        enhanced_boundaries.append(boundary)
            
            # 文の区切りからの境界も追加
            for t in result["transcripts"]:
                if t["text"].strip().endswith(("。", ".", "!", "?", "！", "？")):
                    if t["end"] not in enhanced_boundaries:
                        enhanced_boundaries.append(t["end"])
            
            # 境界の整理と重複除去
            enhanced_boundaries = sorted(list(set(enhanced_boundaries)))
            
            # シーンノード生成
            for i in range(len(enhanced_boundaries) - 1):
                start_time = enhanced_boundaries[i]
                end_time = enhanced_boundaries[i + 1]
                
                # シーンノード作成
                node = VideoNode(start_time, end_time)
                
                # シーン内の文字起こしを取得
                scene_transcripts = [
                    t["text"] for t in result["transcripts"]
                    if t["start"] >= start_time and t["end"] <= end_time
                ]
                node.transcript = " ".join(scene_transcripts)

                # シーンの説明を生成
                description_parts = []
                
                # 無音シーンの判定
                if not scene_transcripts:
                    description_parts.append("無音シーン")
                
                # 仮の説明文
                transcript_text = node.transcript.lower()
                if "山" in transcript_text or "登山" in transcript_text:
                    if "自撮り" in transcript_text or "撮影" in transcript_text:
                        description_parts.append("話者が登山中に自撮りをしている様子")
                    else:
                        description_parts.append("登山に関する会話シーン")
                elif "準備" in transcript_text or "装備" in transcript_text:
                    description_parts.append("登山準備に関するシーン")
                else:
                    description_parts.append("話者が自撮りをしている様子")
                
                node.description = " / ".join(description_parts)
                
                # キーフレーム抽出
                keyframe_path = os.path.join(
                    keyframes_dir,
                    f"keyframe_{i:04d}.jpg"
                )
                if self.extract_keyframe(video_path, start_time, keyframe_path):
                    node.keyframe_path = os.path.relpath(keyframe_path, os.path.dirname(video_path))
                    logger.info(f"キーフレーム保存: {node.keyframe_path}")
                
                # プレビュー動画の生成
                preview_path = os.path.join(
                    preview_dir,
                    f"preview_{i:04d}.mp4"
                )
                if self.generate_preview_clip(video_path, start_time, end_time, preview_path):
                    node.preview_path = os.path.relpath(preview_path, os.path.dirname(video_path))
                    logger.info(f"プレビュー保存: {node.preview_path}")
                
                # シーンの文脈分析と編集提案を生成
                node.context_analysis = self.analyze_scene_context(node, nodes, result["transcripts"])
                node.editing_suggestions = self.generate_editing_suggestions(node, nodes)
                
                nodes.append(node)

            # 動画全体のサマリー生成
            summary = self.generate_video_summary(result["transcripts"], nodes)
            
            # GoProの開始時間を追加
            if gopro_metadata and gopro_metadata.get("start_time"):
                summary["gopro_start_time"] = gopro_metadata["start_time"]
            
            # 結果を保存
            self.save_results(video_path, nodes, completed=True, summary=summary)
            
        except Exception as e:
            logger.error(f"ビデオ処理中にエラー: {str(e)}")
            raise

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

    def generate_video_summary(self, transcripts: list, nodes: list) -> dict:
        """動画全体のサマリーを生成"""
        all_text = " ".join([t["text"] for t in transcripts])
        summary = VideoSummary()
        
        # タイトルの生成
        if "大山" in all_text:
            summary.title = "大山登山記録"
            if "雪" in all_text or "寒" in all_text or "氷点下" in all_text or "冬" in all_text:
                summary.title = "冬の大山登山記録"
            summary.destination = "大山"
        elif "富士山" in all_text:
            summary.title = "富士山登山記録"
            summary.destination = "富士山"
        else:
            summary.title = "登山記録"
        
        # 日付情報の抽出
        for t in transcripts:
            if "月" in t["text"] and "日" in t["text"]:
                import re
                date_match = re.search(r'(\d+)月(\d+)日', t["text"])
                if date_match:
                    month, day = date_match.groups()
                    summary.filming_date = f"{month}月{day}日"
                    # 曜日情報も探す
                    day_of_week = ""
                    if "土曜" in t["text"] or "土曜日" in t["text"]:
                        day_of_week = "（土曜日）"
                    elif "日曜" in t["text"] or "日曜日" in t["text"]:
                        day_of_week = "（日曜日）"
                    elif "月曜" in t["text"] or "月曜日" in t["text"]:
                        day_of_week = "（月曜日）"
                    elif "火曜" in t["text"] or "火曜日" in t["text"]:
                        day_of_week = "（火曜日）"
                    elif "水曜" in t["text"] or "水曜日" in t["text"]:
                        day_of_week = "（水曜日）"
                    elif "木曜" in t["text"] or "木曜日" in t["text"]:
                        day_of_week = "（木曜日）"
                    elif "金曜" in t["text"] or "金曜日" in t["text"]:
                        day_of_week = "（金曜日）"
                    summary.filming_date += day_of_week
                    break
        
        # 目的の抽出
        if "健康" in all_text and ("維持" in all_text or "ため" in all_text):
            summary.purpose = "健康維持"
        
        # 天候情報の抽出
        weather_keywords = {
            "晴れ": "晴天",
            "曇り": "曇天",
            "雨": "雨天",
            "雪": "雪",
            "寒": "寒冷",
            "暑": "暑熱",
            "風": "強風",
            "氷点下": "氷点下"
        }
        weather_conditions = []
        for keyword, condition in weather_keywords.items():
            if keyword in all_text:
                weather_conditions.append(condition)
        summary.weather = "、".join(weather_conditions) if weather_conditions else "不明"
        
        # 移動手段の推測
        transport_keywords = {
            "電車": "電車",
            "バス": "バス",
            "車": "自動車",
            "タクシー": "タクシー",
            "自転車": "自転車"
        }
        transports = []
        for keyword, transport in transport_keywords.items():
            if keyword in all_text:
                transports.append(transport)
        summary.transportation = "、".join(transports) if transports else "不明"
        
        # 出発地点の推測
        location_keywords = ["駅", "バス停", "パーキング", "駐車場", "ロープウェイ"]
        for t in transcripts[:10]:  # 最初の10発言を確認
            for keyword in location_keywords:
                if keyword in t["text"]:
                    location_match = re.search(r'([^\s。、]+%s)' % keyword, t["text"])
                    if location_match:
                        summary.starting_point = location_match.group(1)
                        break
            if summary.starting_point:
                break
        
        # トピックの抽出
        summary.topics = []
        if "山" in all_text or "登山" in all_text:
            summary.topics.append("登山")
        if "健康" in all_text and ("維持" in all_text or "ため" in all_text):
            summary.topics.append("健康維持")
        if "雪" in all_text or "寒" in all_text or "冬" in all_text or "氷点下" in all_text:
            summary.topics.append("冬山")
        if "大山" in all_text:
            summary.topics.append("大山")
        elif "富士山" in all_text:
            summary.topics.append("富士山")
        
        # 基本情報の設定
        summary.scene_count = len(nodes)
        summary.total_duration = nodes[-1].time_out if nodes else 0
        
        # 概要文の生成
        overview_parts = []
        if summary.filming_date:
            overview_parts.append(f"{summary.filming_date}に撮影")
        if summary.destination:
            overview_parts.append(f"{summary.destination}への登山")
        if summary.purpose:
            overview_parts.append(f"{summary.purpose}が目的")
        if summary.weather != "不明":
            overview_parts.append(f"天候は{summary.weather}")
        if summary.transportation != "不明":
            overview_parts.append(f"{summary.transportation}で移動")
        
        summary.overview = "、".join(overview_parts) + "。"
        
        return summary.to_dict()

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
            self.select_button.config(state="normal")
            self.process_button.config(state="normal")
            self.cancel_button.config(state="disabled")
    
    def check_thread_status(self):
        """処理スレッドの状態を確認"""
        if self.processing and self.processing_thread:
            if self.processing_thread.is_alive():
                # まだ実行中なら再度チェック
                self.root.after(100, self.check_thread_status)
            else:
                # 処理が完了したらUIを更新
                self.processing = False
                self.select_button.config(state="normal")
                self.process_button.config(state="normal")
                self.cancel_button.config(state="disabled")
    
    def show_file_details(self, event=None):
        """選択されたファイルの詳細情報を表示"""
        selection = self.file_list.selection()
        if not selection:
            return
        
        item = selection[0]
        file_name = self.file_list.item(item)["values"][0]
        video_path = next(
            (path for path in self.selected_files if os.path.basename(path) == file_name),
            None
        )
        
        if not video_path:
            return
        
        # 詳細情報ウィンドウを作成
        details_window = tk.Toplevel(self.root)
        details_window.title(f"詳細情報 - {file_name}")
        details_window.geometry("1200x800")  # ウィンドウを大きくする
        
        # ノードファイルのパスを取得
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
        nodes_file = os.path.join(output_dir, "nodes.json")
        
        if not os.path.exists(nodes_file):
            ttk.Label(details_window, text="処理結果が見つかりません").pack(padx=10, pady=10)
            return
        
        try:
            with open(nodes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # スクロール可能なフレーム
            main_frame = ttk.Frame(details_window)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # キャンバスとスクロールバー
            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # 基本情報
            info_frame = ttk.LabelFrame(scrollable_frame, text="基本情報")
            info_frame.pack(fill="x", padx=5, pady=5)
            
            # 新フォーマットと旧フォーマットの両方に対応
            summary = data.get("summary", {})
            scene_nodes = data.get("scenes", data.get("nodes", []))
            
            ttk.Label(info_frame, text=f"ファイル名: {file_name}").pack(anchor="w", padx=5, pady=2)
            
            if isinstance(summary, dict):
                if "title" in summary:
                    ttk.Label(info_frame, text=f"タイトル: {summary.get('title', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "overview" in summary:
                    overview_text = tk.Text(info_frame, height=3, width=80, wrap="word")
                    overview_text.insert("1.0", f"概要: {summary.get('overview', '不明')}")
                    overview_text.config(state="disabled")
                    overview_text.pack(anchor="w", padx=5, pady=2)
                
                if "topics" in summary:
                    ttk.Label(info_frame, text=f"主要トピック: {', '.join(summary.get('topics', []))}").pack(anchor="w", padx=5, pady=2)
                
                if "filming_date" in summary:
                    ttk.Label(info_frame, text=f"撮影日: {summary.get('filming_date', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "location" in summary:
                    ttk.Label(info_frame, text=f"撮影場所: {summary.get('location', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "weather" in summary:
                    ttk.Label(info_frame, text=f"天候: {summary.get('weather', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "purpose" in summary:
                    ttk.Label(info_frame, text=f"目的: {summary.get('purpose', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "transportation" in summary:
                    ttk.Label(info_frame, text=f"移動手段: {summary.get('transportation', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "starting_point" in summary:
                    ttk.Label(info_frame, text=f"出発地点: {summary.get('starting_point', '不明')}").pack(anchor="w", padx=5, pady=2)
                
                if "destination" in summary:
                    ttk.Label(info_frame, text=f"目的地: {summary.get('destination', '不明')}").pack(anchor="w", padx=5, pady=2)
            
            ttk.Label(info_frame, text=f"シーン数: {len(scene_nodes)}").pack(anchor="w", padx=5, pady=2)
            ttk.Label(info_frame, text=f"処理状態: {'完了' if data.get('completed') else '未完了'}").pack(anchor="w", padx=5, pady=2)
            ttk.Label(info_frame, text=f"最終更新: {data.get('last_update', '不明')}").pack(anchor="w", padx=5, pady=2)
            
            if isinstance(summary, dict) and "gopro_start_time" in summary:
                ttk.Label(info_frame, text=f"撮影開始時間: {summary.get('gopro_start_time', '不明')}").pack(anchor="w", padx=5, pady=2)
            
            # シーン情報
            scenes_frame = ttk.LabelFrame(scrollable_frame, text="シーン情報")
            scenes_frame.pack(fill="x", padx=5, pady=5)
            
            for i, node in enumerate(scene_nodes):
                scene_frame = ttk.Frame(scenes_frame)
                scene_frame.pack(fill="x", padx=5, pady=5)
                
                # 左側：テキスト情報
                text_frame = ttk.Frame(scene_frame)
                text_frame.pack(side="left", fill="x", expand=True)
                
                # シーンIDを取得（新フォーマットでは明示的に含まれる）
                scene_id = node.get("scene_id", i)
                
                header = f"シーン {scene_id + 1}"
                if node.get("keyframe_path"):
                    header += f" (キーフレーム: {node['keyframe_path']})"
                ttk.Label(text_frame, text=header, font=("", 10, "bold")).pack(anchor="w")
                
                time_info = f"開始: {node['time_in']:.1f}秒 - 終了: {node['time_out']:.1f}秒 (長さ: {node.get('duration', node['time_out'] - node['time_in']):.1f}秒)"
                ttk.Label(text_frame, text=time_info).pack(anchor="w")
                
                # 会話の有無を判定
                has_speech = bool(node.get("transcript", "").strip())
                speech_label = f"会話: {'あり' if has_speech else 'なし'}"
                ttk.Label(text_frame, text=speech_label, foreground='black' if has_speech else 'green').pack(anchor="w")
                
                if node.get("description"):
                    description_text = tk.Text(text_frame, height=2, width=60, wrap="word")
                    description_text.insert("1.0", node['description'])
                    description_text.config(state="disabled")
                    description_text.pack(anchor="w", pady=2)
                
                if node.get("transcript"):
                    transcript_text = tk.Text(text_frame, height=2, width=60, wrap="word")
                    transcript_text.insert("1.0", node['transcript'])
                    transcript_text.config(state="disabled")
                    transcript_text.pack(anchor="w", pady=2)
                
                # 文脈分析情報
                if node.get("context_analysis"):
                    context_frame = ttk.LabelFrame(text_frame, text="文脈分析")
                    context_frame.pack(fill="x", padx=5, pady=5)
                    
                    context = node["context_analysis"]
                    ttk.Label(context_frame, text=f"場所の種類: {context.get('location_type', '不明')}").pack(anchor="w")
                    ttk.Label(context_frame, text=f"推定時刻: {context.get('estimated_time_of_day', '不明')}").pack(anchor="w")
                    ttk.Label(context_frame, text=f"天候状態: {context.get('weather_conditions', '不明')}").pack(anchor="w")
                    ttk.Label(context_frame, text=f"主要な活動: {', '.join(context.get('key_activities', []))}").pack(anchor="w")
                    ttk.Label(context_frame, text=f"感情トーン: {context.get('emotional_tone', '不明')}").pack(anchor="w")
                    ttk.Label(context_frame, text=f"物語上の目的: {context.get('narrative_purpose', '不明')}").pack(anchor="w")
                
                # 編集提案情報
                if node.get("editing_suggestions"):
                    edit_frame = ttk.LabelFrame(text_frame, text="編集提案")
                    edit_frame.pack(fill="x", padx=5, pady=5)
                    
                    suggestions = node["editing_suggestions"]
                    ttk.Label(edit_frame, text=f"ハイライト候補: {'はい' if suggestions.get('highlight_worthy') else 'いいえ'}").pack(anchor="w")
                    ttk.Label(edit_frame, text=f"カットポイント: {'はい' if suggestions.get('potential_cutpoint') else 'いいえ'}").pack(anchor="w")
                    if suggestions.get("b_roll_opportunity"):
                        ttk.Label(edit_frame, text=f"B-roll提案: {suggestions['b_roll_opportunity']}").pack(anchor="w")
                    if suggestions.get("audio_considerations"):
                        ttk.Label(edit_frame, text=f"音声の注意点: {suggestions['audio_considerations']}").pack(anchor="w")
                
                # アクションボタンフレーム
                action_frame = ttk.Frame(text_frame)
                action_frame.pack(anchor="w", pady=5)
                
                # 再生ボタン
                play_button = ttk.Button(
                    action_frame, 
                    text="▶ 再生", 
                    command=lambda v=video_path, t=node["time_in"]: self.play_video_segment(v, t)
                )
                play_button.pack(side="left", padx=5)
                
                # シーン書き出しボタン
                extract_button = ttk.Button(
                    action_frame, 
                    text="📋 シーン保存", 
                    command=lambda v=video_path, s=node["time_in"], e=node["time_out"]: self.extract_video_segment(v, s, e)
                )
                extract_button.pack(side="left", padx=5)
                
                # 右側：サムネイル画像（クリックで再生）
                if node.get("keyframe_path"):
                    try:
                        keyframe_path = os.path.join(os.path.dirname(video_path), node["keyframe_path"])
                        if os.path.exists(keyframe_path):
                            # PILでイメージを読み込み
                            image = Image.open(keyframe_path)
                            # アスペクト比を保持しながらリサイズ
                            image.thumbnail((200, 150))
                            photo = ImageTk.PhotoImage(image)
                            
                            # 画像を表示（クリック可能）
                            image_label = ttk.Label(scene_frame)
                            image_label.image = photo  # 参照を保持
                            image_label.configure(image=photo)
                            image_label.pack(side="right", padx=5)
                            
                            # クリックイベントを追加
                            image_label.bind("<Button-1>", lambda e, v=video_path, t=node["time_in"]: self.play_video_segment(v, t))
                            
                            # ホバー時のカーソル変更
                            image_label.bind("<Enter>", lambda e: e.widget.configure(cursor="hand2"))
                            image_label.bind("<Leave>", lambda e: e.widget.configure(cursor=""))
                    except Exception as e:
                        logger.error(f"サムネイル表示エラー: {str(e)}")
                
                # 区切り線
                ttk.Separator(scene_frame, orient="horizontal").pack(fill="x", pady=5)
            
            # スクロールバーとキャンバスを配置
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
        except Exception as e:
            ttk.Label(details_window, text=f"エラー: {str(e)}").pack(padx=10, pady=10)
            logger.error(f"詳細表示中にエラー: {str(e)}")
    
    def play_video_segment(self, video_path, start_time):
        """指定時間から動画を再生"""
        try:
            # プレビュー動画のパスを取得
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
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
                                preview_path = os.path.join(os.path.dirname(video_path), scene["preview_path"])
                                break
            
            if preview_path and os.path.exists(preview_path):
                # プレビュー動画を再生
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
            if os.name == 'nt':  # Windows
                # VLCがインストールされている場合の時間指定
                try:
                    vlc_path = self.find_vlc_path()
                    if vlc_path:
                        # VLCでタイムスタンプ指定して再生
                        command = [vlc_path, "--start-time", str(int(start_time)), video_path]
                        subprocess.Popen(command)
                        logger.info(f"VLCで動画再生: {video_path}, 開始時間: {start_time}秒")
                        return
                except Exception as e:
                    logger.warning(f"VLCでの再生に失敗しました: {str(e)}")
                
                # VLCが使えない場合はデフォルトプレーヤーで開く
                subprocess.Popen(['start', '', video_path], shell=True)
            elif os.name == 'posix':  # macOS または Linux
                if sys.platform == 'darwin':  # macOS
                    subprocess.Popen(['open', video_path])
                else:  # Linux
                    # VLCが利用可能か確認
                    try:
                        subprocess.run(["which", "vlc"], check=True, capture_output=True)
                        subprocess.Popen(['vlc', '--start-time', str(int(start_time)), video_path])
                        return
                    except subprocess.CalledProcessError:
                        # VLCがない場合はデフォルトプレーヤーで開く
                        subprocess.Popen(['xdg-open', video_path])
            
            logger.info(f"動画再生: {video_path}, 開始時間: {start_time}秒")
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
        
        item = selection[0]
        values = self.file_list.item(item)["values"]
        
        # サムネイル情報を取得
        thumbnail_path = values[3] if len(values) > 3 else None
        if thumbnail_path and thumbnail_path != "-":
            # サムネイル表示
            try:
                file_name = values[0]
                video_path = next(
                    (path for path in self.selected_files if os.path.basename(path) == file_name),
                    None
                )
                
                if video_path:
                    full_thumbnail_path = os.path.join(os.path.dirname(video_path), thumbnail_path)
                    if os.path.exists(full_thumbnail_path):
                        # 画像を読み込んで表示
                        image = Image.open(full_thumbnail_path)
                        # アスペクト比を保持しながらリサイズ
                        image.thumbnail((400, 300))
                        photo = ImageTk.PhotoImage(image)
                        
                        # プレビュー更新
                        self.preview_label.configure(image=photo)
                        self.preview_label.image = photo  # 参照を保持
                        
                        # ファイル情報表示
                        file_info = f"{file_name}\n{thumbnail_path}"
                        self.preview_label.configure(compound="bottom", text=file_info)
                        return
            except Exception as e:
                logger.error(f"プレビュー表示エラー: {str(e)}")
        
        # サムネイルがない場合はデフォルト表示
        self.preview_label.configure(image="")
        self.preview_label.configure(text="サムネイルがありません")
    
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

    def analyze_scene_context(self, node: VideoNode, all_nodes: list, transcripts: list) -> dict:
        """シーンの文脈を分析"""
        scene_text = node.transcript.lower()
        
        # 場所の種類を推定
        location_type = "屋外"  # デフォルト
        indoor_keywords = ["室内", "家", "建物", "店", "駅"]
        transport_keywords = ["電車", "バス", "車", "タクシー"]
        
        for keyword in indoor_keywords:
            if keyword in scene_text:
                location_type = "屋内"
                break
        
        for keyword in transport_keywords:
            if keyword in scene_text:
                location_type = "交通機関内"
                break
        
        # 時刻の推定
        time_of_day = "不明"
        if "朝" in scene_text or "早" in scene_text:
            time_of_day = "朝"
        elif "昼" in scene_text or "午後" in scene_text:
            time_of_day = "昼"
        elif "夕" in scene_text or "夜" in scene_text:
            time_of_day = "夕方/夜"
        
        # 天候状態の推定
        weather_conditions = []
        weather_keywords = {
            "晴れ": "晴天",
            "曇り": "曇天",
            "雨": "雨天",
            "雪": "雪",
            "寒": "寒冷",
            "暑": "暑熱",
            "風": "強風"
        }
        
        for keyword, condition in weather_keywords.items():
            if keyword in scene_text:
                weather_conditions.append(condition)
        
        # 主要な活動の特定
        activities = []
        activity_keywords = {
            "歩": "歩行/移動",
            "登": "登山",
            "休": "休憩",
            "食": "飲食",
            "準備": "準備",
            "装備": "装備確認",
            "撮影": "撮影"
        }
        
        for keyword, activity in activity_keywords.items():
            if keyword in scene_text:
                activities.append(activity)
        
        # 感情トーンの分析
        emotional_tone = "中立"
        positive_keywords = ["楽しい", "嬉しい", "良い", "素晴らしい", "快適"]
        negative_keywords = ["不安", "怖い", "辛い", "疲れ", "心配"]
        
        positive_count = sum(1 for word in positive_keywords if word in scene_text)
        negative_count = sum(1 for word in negative_keywords if word in scene_text)
        
        if positive_count > negative_count:
            emotional_tone = "ポジティブ"
        elif negative_count > positive_count:
            emotional_tone = "ネガティブ"
        
        # 物語上の目的を推定
        narrative_purpose = "情報提供"
        if node == all_nodes[0]:
            narrative_purpose = "導入/状況説明"
        elif node == all_nodes[-1]:
            narrative_purpose = "まとめ/結論"
        elif "準備" in scene_text or "装備" in scene_text:
            narrative_purpose = "準備/計画"
        elif "休憩" in scene_text or "疲れ" in scene_text:
            narrative_purpose = "休息/リフレッシュ"
        
        return {
            "location_type": location_type,
            "estimated_time_of_day": time_of_day,
            "weather_conditions": "、".join(weather_conditions) if weather_conditions else "不明",
            "key_activities": activities,
            "emotional_tone": emotional_tone,
            "narrative_purpose": narrative_purpose
        }

    def generate_editing_suggestions(self, node: VideoNode, all_nodes: list) -> dict:
        """編集提案を生成"""
        scene_text = node.transcript.lower()
        duration = node.time_out - node.time_in
        
        # ハイライト候補の判定
        highlight_worthy = False
        highlight_keywords = ["すごい", "きれい", "素晴らしい", "頂上", "山頂", "達成"]
        if any(keyword in scene_text for keyword in highlight_keywords):
            highlight_worthy = True
        elif duration > 30:  # 長いシーンは重要かもしれない
            highlight_worthy = True
        
        # カットポイントの提案
        potential_cutpoint = False
        if duration < 3:  # 短すぎるシーン
            potential_cutpoint = True
        elif "えーと" in scene_text or "あのー" in scene_text:  # 言い淀み
            potential_cutpoint = True
        
        # B-rollの提案
        b_roll_suggestions = []
        if "景色" in scene_text or "眺め" in scene_text:
            b_roll_suggestions.append("風景のワイドショット")
        if "天気" in scene_text or "空" in scene_text:
            b_roll_suggestions.append("空の様子")
        if "道" in scene_text or "歩" in scene_text:
            b_roll_suggestions.append("歩道/山道の様子")
        
        # 音声に関する注意点
        audio_considerations = []
        if "風" in scene_text:
            audio_considerations.append("風切り音に注意")
        if duration < 2:
            audio_considerations.append("音声が短すぎる可能性")
        
        return {
            "highlight_worthy": highlight_worthy,
            "potential_cutpoint": potential_cutpoint,
            "b_roll_opportunity": "、".join(b_roll_suggestions) if b_roll_suggestions else "",
            "audio_considerations": "、".join(audio_considerations) if audio_considerations else ""
        }

    def run(self):
        """アプリケーション実行"""
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoEnhanced()
    app.run()
