import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
import logging
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class ScenePreviewGUI:
    """
    シーンプレビュー用のGUIコンポーネント
    サムネイルギャラリーとプレビュー再生機能を提供
    """
    
    def __init__(self, parent, vlc_instance=None, player=None):
        """
        初期化
        
        Args:
            parent: 親ウィジェット
            vlc_instance: VLCインスタンス（オプション）
            player: VLCプレーヤー（オプション）
        """
        self.parent = parent
        self.vlc_instance = vlc_instance
        self.player = player
        
        # 状態変数
        self.current_video_path = None
        self.scenes = []
        self.current_scene_index = -1
        self.is_playing = False
        self.update_timer = None
        self.current_duration = 0
        
        # コールバック関数
        self.on_scene_selected = None
        self.on_scene_play = None
        
        # GUIコンポーネントの作成
        self._create_widgets()
    
    def _create_widgets(self):
        """GUIコンポーネントを作成"""
        # メインフレーム
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 上下に分割
        self.paned = ttk.PanedWindow(self.frame, orient=tk.VERTICAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        
        # 上部：サムネイルギャラリー
        self.gallery_frame = ttk.LabelFrame(self.paned, text="シーンギャラリー")
        
        # サムネイルキャンバスとスクロールバー
        self.canvas_frame = ttk.Frame(self.gallery_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#f0f0f0", height=150)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # サムネイルを表示するフレーム
        self.thumbnails_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.thumbnails_frame, anchor=tk.NW)
        
        # 下部：プレビュー再生エリア
        self.preview_frame = ttk.LabelFrame(self.paned, text="シーンプレビュー")
        
        # 動画表示エリア
        self.video_container = ttk.Frame(self.preview_frame, relief="solid", borderwidth=1)
        self.video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 実際の動画表示領域
        self.video_frame = tk.Frame(self.video_container, width=640, height=360, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        # VLCの代わりにデフォルトで表示するラベル
        self.video_label = ttk.Label(self.video_frame, text="シーンを選択してください", 
                                   anchor="center", background="black", foreground="white")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # コントロールパネル
        self.controls_frame = ttk.Frame(self.preview_frame)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 再生/停止ボタン
        self.play_button = ttk.Button(self.controls_frame, text="▶ 再生", command=self.play_preview)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.controls_frame, text="⏹ 停止", command=self.stop_preview)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 前後のシーンに移動するボタン
        self.prev_button = ttk.Button(self.controls_frame, text="◀ 前のシーン", command=self.prev_scene)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(self.controls_frame, text="次のシーン ▶", command=self.next_scene)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # シーン情報
        self.info_frame = ttk.Frame(self.preview_frame)
        self.info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # シーン番号と時間情報
        self.scene_label = ttk.Label(self.info_frame, text="シーン: -")
        self.scene_label.pack(side=tk.LEFT, padx=5)
        
        self.time_label = ttk.Label(self.info_frame, text="時間: --:-- / --:--")
        self.time_label.pack(side=tk.RIGHT, padx=5)
        
        # シーン説明
        self.description_frame = ttk.LabelFrame(self.preview_frame, text="シーン情報")
        self.description_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.description_text = tk.Text(self.description_frame, height=3, wrap=tk.WORD)
        self.description_text.pack(fill=tk.X, padx=5, pady=5)
        self.description_text.insert(tk.END, "シーンの説明がここに表示されます。")
        self.description_text.config(state=tk.DISABLED)
        
        # パネル分割の設定
        self.paned.add(self.gallery_frame, weight=1)
        self.paned.add(self.preview_frame, weight=2)
        
        # VLCプレーヤーの設定
        if self.vlc_instance and self.player:
            # プレイヤーが初期化できたらラベルを削除
            self.video_label.pack_forget()
            if os.name == 'nt':  # Windows
                self.player.set_hwnd(self.video_frame.winfo_id())
            else:  # Linux/MacOS
                self.player.set_xwindow(self.video_frame.winfo_id())
    
    def set_scenes(self, video_path: str, scenes: List[Dict]):
        """
        シーンデータを設定し、サムネイルギャラリーを更新
        
        Args:
            video_path: 動画ファイルのパス
            scenes: シーンデータのリスト
        """
        self.current_video_path = video_path
        self.scenes = scenes
        self.current_scene_index = -1
        
        # サムネイルギャラリーを更新
        self._update_thumbnail_gallery()
        
        # 最初のシーンを選択
        if scenes:
            self.select_scene(0)
    
    def _update_thumbnail_gallery(self):
        """サムネイルギャラリーを更新"""
        # 既存のサムネイルをクリア
        for widget in self.thumbnails_frame.winfo_children():
            widget.destroy()
        
        # サムネイルを追加
        thumbnail_width = 160
        thumbnail_height = 90
        padding = 5
        
        for i, scene in enumerate(self.scenes):
            # サムネイルフレーム
            thumb_frame = ttk.Frame(self.thumbnails_frame)
            thumb_frame.pack(side=tk.LEFT, padx=padding, pady=padding)
            
            # キーフレームパスがあれば表示
            if "keyframe_path" in scene and scene["keyframe_path"] and os.path.exists(scene["keyframe_path"]):
                try:
                    # PILでイメージを読み込み
                    image = Image.open(scene["keyframe_path"])
                    # アスペクト比を保持しながらリサイズ
                    image.thumbnail((thumbnail_width, thumbnail_height))
                    photo = ImageTk.PhotoImage(image)
                    
                    # 画像を表示（クリック可能）
                    image_label = ttk.Label(thumb_frame)
                    image_label.image = photo  # 参照を保持
                    image_label.configure(image=photo)
                    image_label.pack(padx=2, pady=2)
                    
                    # クリックイベントを追加
                    image_label.bind("<Button-1>", lambda e, idx=i: self.select_scene(idx))
                except Exception as e:
                    logger.error(f"サムネイル表示エラー: {str(e)}")
                    # エラー時はテキストラベルを表示
                    ttk.Label(thumb_frame, text=f"シーン {i+1}").pack(padx=2, pady=2)
            else:
                # キーフレームがない場合はテキストラベルを表示
                ttk.Label(thumb_frame, text=f"シーン {i+1}").pack(padx=2, pady=2)
            
            # シーン番号ラベル
            ttk.Label(thumb_frame, text=f"シーン {i+1}").pack(padx=2, pady=2)
            
            # 時間情報ラベル
            start_time = scene.get("start_time", 0)
            end_time = scene.get("end_time", 0)
            duration = end_time - start_time
            time_text = f"{self._format_time(start_time)} - {self._format_time(end_time)}"
            ttk.Label(thumb_frame, text=time_text).pack(padx=2, pady=2)
        
        # キャンバスのスクロール領域を更新
        self.thumbnails_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def select_scene(self, index: int):
        """
        シーンを選択
        
        Args:
            index: シーンのインデックス
        """
        if index < 0 or index >= len(self.scenes):
            return
        
        # 再生中なら停止
        if self.is_playing:
            self.stop_preview()
        
        self.current_scene_index = index
        scene = self.scenes[index]
        
        # シーン情報を表示
        self.scene_label.config(text=f"シーン: {index+1}/{len(self.scenes)}")
        
        start_time = scene.get("start_time", 0)
        end_time = scene.get("end_time", 0)
        duration = end_time - start_time
        self.current_duration = duration
        
        self.time_label.config(text=f"時間: {self._format_time(start_time)} - {self._format_time(end_time)} ({self._format_time(duration)})")
        
        # シーン説明を更新
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        
        # 説明テキストを生成
        description = scene.get("description", "")
        if not description and "ai_analysis" in scene:
            ai_analysis = scene["ai_analysis"]
            activities = ai_analysis.get("activities", [])
            if activities:
                description = f"活動: {', '.join(activities)}"
            else:
                description = "説明なし"
        
        self.description_text.insert(tk.END, description)
        self.description_text.config(state=tk.DISABLED)
        
        # コールバック関数を呼び出し
        if self.on_scene_selected:
            self.on_scene_selected(index, scene)
    
    def play_preview(self):
        """現在選択されているシーンを再生"""
        if self.current_scene_index < 0 or not self.vlc_instance or not self.player:
            return
        
        scene = self.scenes[self.current_scene_index]
        start_time = scene.get("start_time", 0)
        
        try:
            if not self.is_playing:
                logger.info(f"シーン再生開始: {self.current_video_path}, 開始時間: {start_time}秒")
                
                # VLCメディアを設定
                media = self.vlc_instance.media_new(self.current_video_path)
                self.player.set_media(media)
                
                # 開始位置を設定
                self.player.play()
                time.sleep(0.1)  # 少し待機
                self.player.set_time(int(start_time * 1000))  # ミリ秒単位
                
                self.is_playing = True
                # 再生時間の更新タイマーを開始
                self.update_playback_time()
                
                # コールバック関数を呼び出し
                if self.on_scene_play:
                    self.on_scene_play(self.current_scene_index, scene, True)
        except Exception as e:
            logger.error(f"プレビュー再生エラー: {str(e)}")
            self.is_playing = False
    
    def stop_preview(self):
        """再生を停止"""
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
                    self.parent.after_cancel(self.update_timer)
                    self.update_timer = None
                
                # コールバック関数を呼び出し
                if self.on_scene_play and self.current_scene_index >= 0:
                    self.on_scene_play(self.current_scene_index, self.scenes[self.current_scene_index], False)
        except Exception as e:
            logger.error(f"プレビュー停止エラー: {str(e)}")
    
    def prev_scene(self):
        """前のシーンを選択"""
        if self.current_scene_index > 0:
            self.select_scene(self.current_scene_index - 1)
    
    def next_scene(self):
        """次のシーンを選択"""
        if self.current_scene_index < len(self.scenes) - 1:
            self.select_scene(self.current_scene_index + 1)
    
    def update_playback_time(self):
        """再生時間を更新"""
        if not self.player or not self.is_playing:
            return
        
        try:
            if self.player.is_playing():
                # ミリ秒を秒に変換
                current_time = self.player.get_time() / 1000.0
                scene = self.scenes[self.current_scene_index]
                start_time = scene.get("start_time", 0)
                end_time = scene.get("end_time", 0)
                
                # シーン内の相対時間
                relative_time = current_time - start_time
                if relative_time < 0:
                    relative_time = 0
                
                # シーン終了時間を超えたら次のシーンへ
                if current_time >= end_time:
                    self.stop_preview()
                    # 自動的に次のシーンへ
                    if self.current_scene_index < len(self.scenes) - 1:
                        self.next_scene()
                        self.play_preview()
                    return
                
                # 時間表示を更新
                self.time_label.config(text=f"時間: {self._format_time(relative_time)} / {self._format_time(end_time - start_time)}")
                
                # 100ミリ秒後に再度更新
                self.update_timer = self.parent.after(100, self.update_playback_time)
            else:
                # 再生が終了した場合
                self.is_playing = False
                scene = self.scenes[self.current_scene_index]
                start_time = scene.get("start_time", 0)
                end_time = scene.get("end_time", 0)
                self.time_label.config(text=f"時間: {self._format_time(end_time - start_time)} / {self._format_time(end_time - start_time)}")
        except Exception as e:
            logger.error(f"再生時間更新エラー: {str(e)}")
            self.is_playing = False
    
    def _format_time(self, seconds):
        """秒を MM:SS 形式に変換"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def set_on_scene_selected_callback(self, callback: Callable):
        """シーン選択時のコールバック関数を設定"""
        self.on_scene_selected = callback
    
    def set_on_scene_play_callback(self, callback: Callable):
        """シーン再生時のコールバック関数を設定"""
        self.on_scene_play = callback
    
    def get_frame(self):
        """フレームを取得"""
        return self.frame
