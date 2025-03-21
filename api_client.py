import os
import json
import base64
import requests
import logging
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from PIL import Image
from faster_whisper import WhisperModel
import torch
import gc

logger = logging.getLogger(__name__)

# 環境変数のロード
load_dotenv()

# faster-whisperのサポートを追加
try:
    FASTER_WHISPER_AVAILABLE = True
    logger.info("faster-whisperライブラリが利用可能です。高速モードが使用できます。")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.info("faster-whisperライブラリが見つかりません。標準モードで実行します。")
    logger.info("高速モードを使用するには次のコマンドを実行してください: pip install faster-whisper")

class WhisperClient:
    def __init__(self, model_size="large", compute_type="int8", device="cuda", batch_size=8):
        try:
            from faster_whisper import WhisperModel
            
            # GPUメモリ設定
            if device == "cuda" and torch.cuda.is_available():
                # 使用可能なGPUメモリの計算
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)  # バイトからGBに変換
                
                # メモリに基づいて設定を調整（常にint8を使用）
                compute_type = "int8"  # float16をサポートしていないGPUのため
                if gpu_memory_gb < 6:
                    batch_size = 4
                elif gpu_memory_gb < 8:
                    batch_size = 6
                
                logger.info(f"GPU Memory: {gpu_memory_gb:.2f}GB")
                logger.info(f"Using compute_type: {compute_type}, batch_size: {batch_size}")
                
                # GPUメモリをクリア
                torch.cuda.empty_cache()
                gc.collect()
            else:
                device = "cpu"
                compute_type = "int8"
                batch_size = 4
            
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            logger.info(f"faster-whisper {model_size}モデルを初期化中... (device: {device}, compute_type: {compute_type})")
            
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=1,  # ワーカー数を減らしてメモリ使用量を削減
                download_root="models"
            )
            
            self.batch_size = batch_size
            logger.info(f"faster-whisper {model_size}モデルを{device}モードで初期化しました")
            
        except Exception as e:
            logger.error(f"WhisperClient初期化エラー: {str(e)}")
            raise

    def process_video(self, video_path: str, min_silence: float = 1.0, start_time: float = 0.0) -> dict:
        """ビデオを処理し、文字起こしとシーン境界を返す"""
        try:
            # 音声ファイルの一時パス
            audio_file = os.path.join(
                os.path.dirname(video_path),
                f"temp_audio_{os.path.splitext(os.path.basename(video_path))[0]}.wav"
            )
            
            try:
                # 音声抽出
                self.extract_audio(video_path, audio_file)
                
                # 文字起こし
                segments, _ = self.model.transcribe(
                    audio_file,
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=min_silence * 1000
                    )
                )
                
                # セグメントをリストに変換
                segments = list(segments)
                
                # シーン境界を検出
                boundaries = self.detect_silence_boundaries(segments, min_silence)
                
                return {
                    "transcripts": [
                        {
                            "start": s.start,
                            "end": s.end,
                            "text": s.text
                        }
                        for s in segments
                    ],
                    "scene_boundaries": boundaries
                }
                
            finally:
                # 一時ファイルを削除
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                
                # GPUメモリを解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
            
        except Exception as e:
            logger.error(f"ビデオ処理エラー: {str(e)}")
            raise

    def extract_audio(self, video_path: str, audio_file: str):
        """動画から音声を抽出（1回だけ）"""
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            audio_file
        ]
        
        subprocess.run(command, check=True, capture_output=True)

    def detect_silence_boundaries(self, segments: List, min_silence: float = 1.0) -> List[float]:
        """無音部分（セグメント間のギャップ）を検出してシーン区切りとして返す"""
        boundaries = [0.0]  # 最初は常に0秒から始める
        
        # 最低2つのセグメントが必要
        if len(segments) < 2:
            if segments:
                end_time = segments[0].end
                boundaries.append(end_time)
            else:
                boundaries.append(0.0)
            return boundaries
        
        # セグメント間のギャップを探す
        for i in range(1, len(segments)):
            prev_end = segments[i-1].end
            curr_start = segments[i].start
            
            # セグメント間のギャップがmin_silence以上なら境界とみなす
            if curr_start - prev_end >= min_silence:
                # 境界は前のセグメントの終了時間とする
                boundaries.append(prev_end)
                logger.info(f"無音検出: {prev_end:.2f}秒 - {curr_start:.2f}秒 ({curr_start-prev_end:.2f}秒)")
        
        # 最後のセグメントの終了時間も追加
        if segments:
            boundaries.append(segments[-1].end)
        
        # 重複を排除して並べ替え
        boundaries = sorted(list(set(boundaries)))
        
        logger.info(f"無音による境界検出: {len(boundaries)-1}個のシーンに分割")
        return boundaries

    def get_text_for_timerange(self, result: Dict, start_time: float, end_time: float) -> str:
        """特定の時間範囲のテキストを抽出"""
        if not result or "segments" not in result:
            return ""
        
        segments = result.get("segments", [])
        matched_text = []
        
        for segment in segments:
            segment_start = segment.get("start", 0)
            segment_end = segment.get("end", 0)
            
            # 時間範囲が重なるセグメントを抽出
            if segment_end >= start_time and segment_start <= end_time:
                matched_text.append(segment.get("text", ""))
        
        return " ".join(matched_text)

    def transcribe(self, video_path: str, min_silence: float = 1.0) -> Dict:
        """動画ファイルから音声を抽出し、文字起こしを行う"""
        try:
            # 音声ファイルの一時パス
            audio_file = os.path.join(
                os.path.dirname(video_path),
                f"temp_audio_{os.path.splitext(os.path.basename(video_path))[0]}.wav"
            )
            
            try:
                # 音声抽出
                self.extract_audio(video_path, audio_file)
                
                # Whisperで文字起こし
                segments, _ = self.model.transcribe(
                    audio_file,
                    language="ja",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=int(min_silence * 1000)
                    )
                )
                
                # セグメントから無音区間と文字起こしを抽出
                scene_boundaries = []
                transcripts = []
                
                for i in range(len(segments) - 1):
                    current_seg = segments[i]
                    next_seg = segments[i + 1]
                    
                    transcripts.append({
                        "start": current_seg.start,
                        "end": current_seg.end,
                        "text": current_seg.text
                    })
                    
                    gap = next_seg.start - current_seg.end
                    if gap >= self.chunk_duration:
                        scene_boundaries.append(current_seg.end)
                
                # 最後のセグメントを追加
                if segments:
                    last_seg = segments[-1]
                    transcripts.append({
                        "start": last_seg.start,
                        "end": last_seg.end,
                        "text": last_seg.text
                    })
                
                return {
                    "scene_boundaries": scene_boundaries,
                    "transcripts": transcripts
                }
            
            finally:
                # 一時ファイルを削除
                if os.path.exists(audio_file):
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                
                # GPUメモリを解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        except Exception as e:
            logger.error(f"動画処理中にエラー: {str(e)}")
            raise

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEYが設定されていません")
            raise ValueError("GEMINI_API_KEYが設定されていません。.envファイルを確認してください。")
        
        # APIクライアントを初期化
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            # モデル名を新しいものに更新
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini 1.5 Flashモデルを初期化しました")
        except ImportError:
            logger.error("google-generativeaiライブラリがインストールされていません")
            raise ImportError("google-generativeaiパッケージをインストールしてください: pip install google-generativeai")
    
    def analyze_image(self, image_path: str) -> str:
        """画像をGemini APIで分析"""
        if not os.path.exists(image_path):
            logger.error(f"画像ファイルが見つかりません: {image_path}")
            return "画像ファイルが見つかりません"
        
        logger.info(f"画像分析開始: {image_path}")
        
        try:
            # PILで画像を読み込み
            img = Image.open(image_path)
            
            # 分析実行
            prompt = "この画像を詳細に説明してください。映像のシーンとして何が映っているか、視覚的特徴を具体的に説明してください。"
            
            response = self.model.generate_content([prompt, img])
            
            if response.text:
                logger.info("画像分析完了")
                return response.text
            else:
                logger.warning("画像分析結果が空です")
                return "説明を生成できませんでした"
            
        except Exception as e:
            error_msg = f"画像分析中にエラーが発生: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return f"画像分析エラー: {type(e).__name__}"
    
    def analyze_scene(self, image_path, transcript):
        """シーンの詳細分析を行う"""
        try:
            prompt = f"""
            以下の情報から、シーンの詳細な分析を行ってください。
            
            画像とトランスクリプトを分析し、以下の要素を特定してください：
            1. シーンタイプ（urban_street, indoor_room, nature, etc.）
            2. トピック（主な話題や行動）
            3. 感情（シーンの全体的な感情）
            4. 追加のラベル（関連するキーワード）
            
            トランスクリプト: {transcript}
            
            結果をJSON形式で返してください。
            """
            
            response = self.model.generate_content([prompt, Image.open(image_path)])
            
            # レスポンスをパースしてJSON形式に変換
            analysis = json.loads(response.text)
            return {
                "scene_type": analysis.get("scene_type"),
                "topic": analysis.get("topic"),
                "emotion": analysis.get("emotion"),
                "labels": analysis.get("labels", [])
            }
            
        except Exception as e:
            logger.error(f"シーン分析中にエラーが発生: {str(e)}")
            return None
