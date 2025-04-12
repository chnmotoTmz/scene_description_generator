import cv2
import numpy as np
import logging
import os
import time
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class EnhancedSceneDetector:
    """
    強化されたシーン検出クラス
    音声ベースと映像ベースのシーン検出を組み合わせて高精度なシーン分割を実現
    """
    
    def __init__(self, 
                 min_scene_duration: float = 3.0,
                 hist_threshold: float = 0.9,
                 pixel_threshold: float = 20.0,
                 sample_rate: int = 20):
        """
        初期化
        
        Args:
            min_scene_duration: 最小シーン長さ（秒）
            hist_threshold: ヒストグラム相関閾値（低いほど敏感）
            pixel_threshold: ピクセル差分閾値（高いほど敏感）
            sample_rate: サンプリングレート（フレーム数）- 高いほど処理が速くなるが精度が低下
        """
        self.min_scene_duration = min_scene_duration
        self.hist_threshold = hist_threshold
        self.pixel_threshold = pixel_threshold
        self.sample_rate = sample_rate
        
        # 進捗コールバック
        self.progress_callback = None
        
        # 検出結果
        self.scene_boundaries = []
        self.keyframes = []
        
        logger.info(f"EnhancedSceneDetector初期化: sample_rate={sample_rate} (高速モード), min_scene_duration={min_scene_duration}")
        
    def set_progress_callback(self, callback):
        """進捗報告用コールバックを設定"""
        self.progress_callback = callback
        
    def report_progress(self, progress: float, message: str):
        """進捗を報告"""
        if self.progress_callback:
            self.progress_callback(progress, message)
        
    def detect_scenes(self, video_path: str, output_dir: str = None, max_duration: float = None) -> List[Dict]:
        """
        動画からシーンを検出
        
        Args:
            video_path: 動画ファイルのパス
            output_dir: 出力ディレクトリ（指定がなければ動画と同じディレクトリ）
            max_duration: 処理する最大時間（秒）。Noneの場合は動画全体を処理
            
        Returns:
            List[Dict]: 検出されたシーンのリスト
        """
        try:
            # 出力ディレクトリの準備
            if not output_dir:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.join(os.path.dirname(video_path), f"video_nodes_{base_name}")
            
            keyframes_dir = os.path.join(output_dir, "keyframes")
            os.makedirs(keyframes_dir, exist_ok=True)
            
            # 動画ファイルを開いて基本情報を取得
            cap, fps, total_frames, total_duration = self._open_video_and_get_info(video_path)
            if not cap:
                return self._create_fallback_scene(total_duration=0)
            
            # 最大時間が指定されている場合、処理時間を制限
            if max_duration is not None and max_duration > 0:
                total_duration = min(total_duration, max_duration)
                total_frames = min(total_frames, int(max_duration * fps))
                logger.info(f"最大時間を{max_duration}秒に制限して処理します（実際の処理時間: {total_duration:.2f}秒）")
            
            self.report_progress(5, "動画情報を取得しました")
            
            # 第1パス: 差分値を収集して適応型閾値を算出
            hist_threshold, pixel_threshold = self._calculate_adaptive_thresholds(
                cap, fps, self.sample_rate, max_frames=total_frames
            )
            
            self.report_progress(30, "適応型閾値を計算しました")
            
            # 第2パス: シーン境界を検出
            self.scene_boundaries = self._detect_scene_boundaries(
                video_path, hist_threshold, pixel_threshold, fps, total_duration, max_frames=total_frames
            )
            
            self.report_progress(60, f"{len(self.scene_boundaries)-1}個のシーン候補を検出しました")
            
            # 短いシーンをフィルタリング
            self.scene_boundaries = self._filter_short_scenes(
                self.scene_boundaries, self.min_scene_duration
            )
            
            self.report_progress(70, f"{len(self.scene_boundaries)-1}個のシーンに絞り込みました")
            
            # シーンごとにキーフレームを抽出し、データを生成
            scene_data = self._generate_scene_data(
                video_path, self.scene_boundaries, keyframes_dir
            )
            
            self.report_progress(90, "シーンデータを生成しました")
            
            # 結果の検証と調整
            scene_data = self._validate_scene_data(scene_data, total_duration)
            
            # 時間制限が指定されている場合、最終シーンの終了時間を調整
            if max_duration is not None and scene_data:
                for scene in scene_data:
                    if scene["end_time"] > max_duration:
                        scene["end_time"] = max_duration
                        scene["duration"] = scene["end_time"] - scene["start_time"]
            
            self.report_progress(100, "シーン検出が完了しました")
            
            return scene_data

        except Exception as e:
            logger.error(f"シーン検出エラー: {str(e)}", exc_info=True)
            # エラー時は動画全体を1シーンとして返す
            return [self._create_fallback_scene(
                total_duration=max_duration if max_duration is not None else 
                (total_frames / fps if 'total_frames' in locals() and 'fps' in locals() else 0.0)
            )]
    
    def _open_video_and_get_info(self, video_path: str) -> Tuple:
        """
        動画ファイルを開き、基本情報を取得する
        
        Args:
            video_path: 動画ファイルのパス
            
        Returns:
            tuple: (cap, fps, total_frames, total_duration)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"ビデオファイルを開けません: {video_path}")
            return None, 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps
        
        logger.info(f"動画情報: 総フレーム数={total_frames}, FPS={fps}, 推定時間={total_duration:.2f}秒")
        return cap, fps, total_frames, total_duration
    
    def _calculate_adaptive_thresholds(self, cap, fps, sample_rate, max_frames=None):
        """
        動画をサンプリングして差分データを収集し、適応型閾値を算出する
        
        Args:
            cap: OpenCVのVideoCapture
            fps: フレームレート
            sample_rate: サンプリングレート
            max_frames: 処理する最大フレーム数（Noneの場合は全フレーム処理）
            
        Returns:
            tuple: (hist_threshold, pixel_threshold)
        """
        prev_frame = None
        prev_hist = None
        frame_count = 0
        hist_diffs = []
        pixel_diffs = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 先頭に移動
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
            
        progress_interval = max(1, total_frames // 100)  # 進捗報告間隔
        
        while cap.isOpened() and (max_frames is None or frame_count < max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # 進捗報告
                if frame_count % progress_interval == 0:
                    progress = (frame_count / total_frames) * 25  # 25%までの進捗
                    self.report_progress(progress, f"閾値計算中... ({frame_count}/{total_frames}フレーム)")
                
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
            
            # 最大フレーム数に達したら終了
            if max_frames is not None and frame_count >= max_frames:
                break
        
        # 適応型閾値の計算
        if len(hist_diffs) < 2 or len(pixel_diffs) < 2:
            logger.warning("差分データが不足しています。デフォルト閾値を使用します。")
            hist_threshold = 0.9  # デフォルト値
            pixel_threshold = 20.0  # デフォルト値
        else:
            hist_diffs = np.array(hist_diffs)
            pixel_diffs = np.array(pixel_diffs)
            
            hist_mean = np.mean(hist_diffs)
            hist_std = np.std(hist_diffs)
            pixel_mean = np.mean(pixel_diffs)
            pixel_std = np.std(pixel_diffs)
            
            # 閾値を調整: ヒストグラム閾値を0.9以下、ピクセル差分閾値を適度に
            hist_threshold = min(0.92, hist_mean - 0.7 * hist_std)
            pixel_threshold = max(10.0, pixel_mean + 1.0 * pixel_std)
            
            logger.info(f"適応型閾値: ヒストグラム={hist_threshold:.3f} (平均={hist_mean:.3f}, 標準偏差={hist_std:.3f})")
            logger.info(f"適応型閾値: ピクセル差分={pixel_threshold:.3f} (平均={pixel_mean:.3f}, 標準偏差={pixel_std:.3f})")
        
        return hist_threshold, pixel_threshold
    
    def _detect_scene_boundaries(self, video_path, hist_threshold, pixel_threshold, fps, total_duration, max_frames=None):
        """
        シーン境界を検出する
        
        Args:
            video_path: 動画ファイルのパス
            hist_threshold: ヒストグラム閾値
            pixel_threshold: ピクセル閾値
            fps: フレームレート
            total_duration: 動画の長さ（秒）
            max_frames: 処理する最大フレーム数（Noneの場合は全フレーム処理）
            
        Returns:
            list: シーン境界（秒）のリスト
        """
        # 動画を開く
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"ビデオファイルを開けません: {video_path}")
            return [0.0, total_duration]

        # 基本情報
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total_video_frames = min(total_video_frames, max_frames)
        logger.info(f"シーン検出: 閾値=[ヒストグラム:{hist_threshold:.3f}, ピクセル:{pixel_threshold:.1f}]")
        
        # 検出に使用するパラメータ
        sample_rate = self.sample_rate
        min_scene_len = int(fps * 0.5)  # 0.5秒未満のシーンは除外
        boundaries = [0.0]  # 常に0秒から開始
        
        # フレーム処理用の変数
        prev_frame = None
        prev_hist = None
        frame_count = 0
        scene_frames = []
        progress_interval = max(1, total_video_frames // 200)  # 進捗報告間隔
        
        # 先頭に移動
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # フレームを順に処理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 最大フレーム数に達したら終了
            if max_frames is not None and frame_count > max_frames:
                break
            
            # 進捗報告
            if frame_count % progress_interval == 0:
                progress = 30 + (frame_count / total_video_frames) * 30  # 30%-60%の進捗
                self.report_progress(int(progress), f"シーン検出中... ({frame_count}/{total_video_frames}フレーム)")
            
            # サンプリングレートに基づいてフレームを処理
            if frame_count % sample_rate == 0:
                # グレースケール変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # ヒストグラム計算
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if prev_hist is not None and prev_frame is not None:
                    # ヒストグラム相関（1に近いほど類似）
                    hist_corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # ピクセル差分
                    pixel_diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = np.mean(pixel_diff)
                    
                    # シーン変化の判定
                    is_new_scene = False
                    
                    # 条件1: ヒストグラム相関が閾値より低い（大きな変化）
                    if hist_corr < hist_threshold:
                        is_new_scene = True
                        logger.debug(f"ヒストグラム変化を検出: {frame_count}, 相関値={hist_corr:.4f}")
                    
                    # 条件2: ピクセル差分が閾値より高い（大きな変化）
                    elif mean_diff > pixel_threshold:
                        is_new_scene = True
                        logger.debug(f"ピクセル変化を検出: {frame_count}, 差分値={mean_diff:.2f}")
                    
                    # 新しいシーンの開始
                    if is_new_scene:
                        # 直前のシーンが短すぎないかチェック
                        if not scene_frames or frame_count - scene_frames[-1] > min_scene_len:
                            scene_frames.append(frame_count)
                            timestamp = frame_count / fps
                            boundaries.append(timestamp)
                            logger.debug(f"シーン境界検出: {timestamp:.2f}秒")
                
                # 現在のフレームを保存
                prev_hist = hist
                prev_frame = gray.copy()
        
        # 動画の最後の時間を追加
        if max_frames is not None:
            # 最大フレーム数に制限されている場合
            end_time = min(total_duration, max_frames / fps)
        else:
            # 動画全体を処理した場合
            end_time = total_duration
            
        if not boundaries or boundaries[-1] < end_time:
            boundaries.append(end_time)
        
        # リソース開放
        cap.release()
        
        logger.info(f"検出されたシーン境界: {len(boundaries)-1}個のシーン")
        return boundaries
    
    def _filter_short_scenes(self, scene_boundaries, min_scene_duration=3.0):
        """
        短すぎるシーンを統合する
        
        Args:
            scene_boundaries: シーン境界のリスト
            min_scene_duration: 最小シーン長さ（秒）
            
        Returns:
            list: フィルタリング後のシーン境界リスト
        """
        if len(scene_boundaries) <= 2:
            return scene_boundaries
            
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
    
    def _generate_scene_data(self, video_path, scene_boundaries, keyframes_dir):
        """
        各シーンのデータとキーフレームを生成する
        
        Args:
            video_path: 動画ファイルのパス
            scene_boundaries: シーン境界のリスト
            keyframes_dir: キーフレーム保存ディレクトリ
            
        Returns:
            list: 生成されたシーンデータのリスト
        """
        scene_data = []
        start_time = time.time()
        
        # 各シーンについて処理
        for i in range(len(scene_boundaries) - 1):
            start_time_scene = scene_boundaries[i]
            end_time_scene = scene_boundaries[i + 1]
            
            # 進捗報告
            progress = 70 + (i / (len(scene_boundaries) - 1)) * 20  # 70%～90%の進捗
            self.report_progress(progress, f"シーン {i+1}/{len(scene_boundaries)-1} を処理中...")
            
            # シーンの中間点でキーフレームを抽出
            keyframe_time = (start_time_scene + end_time_scene) / 2
            keyframe_filename = f"keyframe_{i:04d}.jpg"
            keyframe_path = os.path.join(keyframes_dir, keyframe_filename)
            
            # キーフレーム抽出
            extraction_success = self._extract_keyframe(video_path, keyframe_time, keyframe_path)
            if not extraction_success:
                logger.warning(f"シーン {i+1} のキーフレーム抽出に失敗しました。再試行します。")
                # 異なる時間で再試行
                retry_time = start_time_scene + (end_time_scene - start_time_scene) * 0.25
                extraction_success = self._extract_keyframe(video_path, retry_time, keyframe_path)
                if not extraction_success:
                    logger.error(f"シーン {i+1} のキーフレーム再抽出にも失敗しました。")
            
            # キーフレームパスを絶対パスに変換（JSONに保存するため）
            abs_keyframe_path = os.path.abspath(keyframe_path) if extraction_success else None
            
            # シーンデータを作成
            scene_data.append({
                "scene_id": i,
                "start_time": start_time_scene,
                "end_time": end_time_scene,
                "duration": end_time_scene - start_time_scene,
                "keyframe_path": abs_keyframe_path,
                "keyframe_filename": keyframe_filename,  # ファイル名も保存
                "keyframe_time": keyframe_time,
                "extraction_success": extraction_success
            })
        
        end_time = time.time()
        logger.info(f"シーンデータ生成完了: {len(scene_data)}個のシーン, 処理時間: {end_time - start_time:.2f}秒")
        
        return scene_data
    
    def _extract_keyframe(self, video_path, timestamp, output_path):
        """
        指定時間のキーフレームを抽出
        
        Args:
            video_path: 動画ファイルのパス
            timestamp: 抽出する時間（秒）
            output_path: 出力ファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            cap = cv2.VideoCapture(video_path)
            # ミリ秒単位で指定
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if ret:
                cv2.imwrite(output_path, frame)
                cap.release()
                return True
            else:
                logger.error(f"キーフレーム抽出失敗: {timestamp}秒")
                cap.release()
                return False
                
        except Exception as e:
            logger.error(f"キーフレーム抽出エラー: {str(e)}")
            return False
    
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
            scene_data[-1]["duration"] = total_duration - scene_data[-1]["start_time"]

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
            "scene_id": 0,
            "start_time": 0, 
            "end_time": total_duration,
            "duration": total_duration,
            "keyframe_path": None,
            "keyframe_filename": None,
            "keyframe_time": 0,
            "extraction_success": False
        }
    
    def merge_with_audio_boundaries(self, audio_boundaries):
        """
        音声ベースのシーン境界と映像ベースのシーン境界を統合
        
        Args:
            audio_boundaries: 音声ベースのシーン境界リスト
            
        Returns:
            list: 統合されたシーン境界リスト
        """
        if not self.scene_boundaries:
            return audio_boundaries
            
        if not audio_boundaries:
            return self.scene_boundaries
            
        # 両方の境界をマージして重複を排除
        merged_boundaries = sorted(list(set(self.scene_boundaries + audio_boundaries)))
        
        # 短いシーンをフィルタリング
        return self._filter_short_scenes(merged_boundaries, self.min_scene_duration)
