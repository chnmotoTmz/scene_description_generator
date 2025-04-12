import os
import json
import base64
import requests
import logging
import subprocess
import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Any, Callable, TypeVar, Union
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from PIL import Image
from faster_whisper import WhisperModel
import torch
import gc
import traceback
from datetime import datetime

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

T = TypeVar('T')

class WhisperClient:
    def __init__(self, model_size="large-v3", compute_type="int8", device="cuda", batch_size=8):
        try:
            from faster_whisper import WhisperModel
            
            # large-v3モデルを使用（最高精度、GPUで高速処理）
            # GPUメモリ設定
            if device == "cuda" and torch.cuda.is_available():
                # 使用可能なGPUメモリの計算
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)  # バイトからGBに変換
                
                # メモリに基づいて設定を調整
                compute_type = "int8"  # float16をサポートしていないGPUのため
                if gpu_memory_gb < 4:
                    # 小さいGPUメモリでは使用不可
                    logger.warning(f"GPUメモリが不足しています({gpu_memory_gb:.2f}GB)。CPUモードに切り替えます。")
                    device = "cpu"
                    model_size = "small"  # 小さいモデルに切り替え
                elif gpu_memory_gb < 6:
                    batch_size = 1
                    if model_size == "large-v3":
                        model_size = "medium"
                        logger.info(f"GPUメモリが制限されています({gpu_memory_gb:.2f}GB)。mediumモデルに自動調整しました")
                elif gpu_memory_gb < 8:
                    batch_size = 4
                    # large-v3を維持
                else:
                    # 十分なGPUメモリがある場合、バッチサイズを増やす
                    batch_size = 16
                    logger.info(f"十分なGPUメモリ({gpu_memory_gb:.2f}GB)を検出。バッチサイズを{batch_size}に増加")
                
                logger.info(f"GPU Memory: {gpu_memory_gb:.2f}GB")
                logger.info(f"Using compute_type: {compute_type}, batch_size: {batch_size}")
                
                # GPUメモリをクリア
                torch.cuda.empty_cache()
                gc.collect()
            else:
                device = "cpu"
                compute_type = "int8"
                batch_size = 1
                # CPUモードでは、中型モデルを使用
                if model_size == "large-v3":
                    model_size = "small"
                    logger.info("CPUモードではlargeモデルは処理が重いため、smallに自動調整しました")
            
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            logger.info(f"faster-whisper {model_size}モデルを初期化中... (device: {device}, compute_type: {compute_type})")
            
            # 最適化パラメータ
            download_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(download_root, exist_ok=True)
            
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=os.cpu_count() or 4,  # 使用可能なCPUコア数を自動検出
                num_workers=2,  # GPUモードではワーカー数を増やす
                download_root=download_root
            )
            
            self.model_size = model_size
            self.device = device
            self.batch_size = batch_size
            # 無音区間を判定する基準となる時間（秒）
            self.chunk_duration = 1.0
            logger.info(f"faster-whisper {model_size}モデルを{device}モードで初期化しました (高精度モード)")
            
        except Exception as e:
            logger.error(f"WhisperClient初期化エラー: {str(e)}")
            raise

    def process_video(self, video_path: str, min_silence: float = 1.0, start_time: float = 0.0, max_duration: float = None) -> dict:
        """ビデオを処理し、文字起こしとシーン境界を返す"""
        try:
            # 音声ファイルの一時パス
            audio_file = os.path.join(
                os.path.dirname(video_path),
                f"temp_audio_{os.path.splitext(os.path.basename(video_path))[0]}.wav"
            )
            
            try:
                # 音声抽出（最大時間制限がある場合）
                duration_param = []
                if max_duration is not None:
                    logger.info(f"最大時間を{max_duration}秒に制限して処理します")
                    duration_param = ["-t", str(max_duration)]
                    
                # 音声抽出コマンド
                command = [
                    "ffmpeg", "-y",
                    "-i", video_path
                ]
                
                # 最大時間制限を追加（ある場合）
                if duration_param:
                    command.extend(duration_param)
                    
                # 残りのパラメータ
                command.extend([
                    "-vn",
                    "-ar", "16000",
                    "-ac", "1",
                    "-b:a", "32k",
                    audio_file
                ])
                
                subprocess.run(command, check=True, capture_output=True)
                
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
                
                # 最大時間内のセグメントだけに制限（念のため）
                if max_duration is not None:
                    segments = [s for s in segments if s.start <= max_duration]
                
                # シーン境界を検出
                boundaries = self.detect_silence_boundaries(segments, min_silence)
                
                # 最大時間で区切る（必要な場合）
                if max_duration is not None and (not boundaries or boundaries[-1] < max_duration):
                    if boundaries and boundaries[-1] < max_duration:
                        boundaries.append(max_duration)
                    elif not boundaries:
                        boundaries = [0.0, max_duration]
                
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

    def transcribe(self, audio_file: str, language="ja") -> List[Dict]:
        """音声ファイルを文字起こし"""
        start_time = time.time()
        logger.info(f"文字起こし開始: {audio_file} (高精度モード)")
        
        try:
            # VADパラメータ（精度重視の設定）
            vad_params = {
                "min_silence_duration_ms": 300,  # 短い無音区間（0.3秒）も検出
                "threshold": 0.3,  # 無音検出の閾値を低く設定
                "min_speech_duration_ms": 100  # より短い発話も検出（0.1秒）
            }
            
            # 高精度モードでの文字起こし設定
            accuracy_params = {
                "beam_size": 5,  # より広い探索（精度向上）
                "best_of": 5,
                "patience": 1.5,  # より正確な探索
                "temperature": 0.0,  # ランダム性を排除
                "initial_prompt": None,
                "condition_on_previous_text": True,  # 前のテキストに条件付け（文脈の一貫性向上）
                "compression_ratio_threshold": 1.35,  # より厳密な制約
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.4,  # 音声がない場合の閾値を適度に
                "word_timestamps": True,  # 単語単位のタイムスタンプを有効化
                "vad_filter": True,  # 音声区間検出を有効化
                "vad_parameters": vad_params
            }
            
            # GPU最適化（chunk_lengthを増やして並列処理を向上）
            if self.device == "cuda" and torch.cuda.is_available():
                # GPUモードでは長いチャンクで効率化
                self.model.feature_extractor.hop_length = 160
                self.model.feature_extractor.chunk_length = 30
            else:
                # CPUモードでは短いチャンクを使用
                self.model.feature_extractor.hop_length = 160
                self.model.feature_extractor.chunk_length = 15
            
            # トランスクリプション実行
            segments, info = self.model.transcribe(
                audio_file,
                language=language,
                **accuracy_params
            )
            
            # 結果をリストに変換（イテレータからリストへ）
            segments_list = list(segments)
            
            # 検出情報のログ
            logger.info(f"文字起こし完了: {len(segments_list)}個のセグメントを検出 (言語: {info.language}, 確率: {info.language_probability:.2f})")
            logger.info(f"処理時間: {time.time() - start_time:.2f}秒")
            
            # セグメントの長さと数をログ出力
            total_text_length = sum(len(s.text) for s in segments_list)
            logger.info(f"総テキスト長: {total_text_length}文字, セグメント数: {len(segments_list)}")
            
            return segments_list
            
        except Exception as e:
            logger.error(f"文字起こしエラー: {str(e)}")
            logger.error(traceback.format_exc())
            # エラー時は空のリストを返す
            return []
        finally:
            # GPUメモリを解放
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

class GeminiClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        # APIの最新エンドポイントとモデル名
        self.base_url = "https://generativelanguage.googleapis.com/v1/models"
        self.model = "gemini-1.5-flash"
        self.vision_model = "gemini-1.5-flash-vision"
        
        # リトライ設定
        self.max_retries = 3
        self.base_delay = 2  # 基本遅延（秒）
        self.max_delay = 10  # 最大遅延（秒）
        
        # API接続テスト
        self._test_api_connection()
    
    def _test_api_connection(self) -> bool:
        """APIキーが有効かテストする"""
        if not self.api_key:
            logger.error("Gemini APIキーが設定されていません。デモモードで実行します。")
            return False
        
        try:
            logger.info("Gemini API接続テスト実行中...")
            logger.info(f"使用するAPIベースURL: {self.base_url}")
            logger.info(f"使用するモデル: {self.model}")
            
            # シンプルなプロンプトでAPIをテスト
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                    json={"contents": [{"parts": [{"text": "Hello"}]}]}
                )
            )
            
            if response and response.status_code == 200:
                logger.info("Gemini API接続テスト成功")
                return True
            else:
                status = response.status_code if response else "不明"
                error_text = response.text if response else "応答なし"
                logger.error(f"Gemini API接続テスト失敗: ステータス {status}, エラー: {error_text}")
                
                # エラーの詳細を追加
                if response and response.status_code == 404:
                    logger.error("404エラー: 指定されたモデルまたはAPIバージョンが見つかりません。APIベースURLとモデル名を確認してください。")
                    logger.error(f"現在の設定: base_url={self.base_url}, model={self.model}")
                    # 代替モデル名を提案
                    logger.error("代替設定を試みます。model=gemini-pro に変更...")
                    # gemini-proモデルを試す
                    self.model = "gemini-pro"
                    self.vision_model = "gemini-pro-vision"
                    logger.info(f"モデルを変更しました: {self.model}")
                    
                    # 再試行
                    retry_response = self._retry_operation(
                        lambda: requests.post(
                            f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                            json={"contents": [{"parts": [{"text": "Hello"}]}]}
                        )
                    )
                    
                    if retry_response and retry_response.status_code == 200:
                        logger.info(f"代替モデル {self.model} でAPI接続テスト成功")
                        return True
                    else:
                        logger.error(f"代替モデルでもAPI接続テスト失敗")
                        return False
                        
                elif response and response.status_code == 401:
                    logger.error("401エラー: APIキーが無効です。有効なAPIキーを設定してください。")
                return False
                
        except Exception as e:
            logger.error(f"Gemini API接続テスト中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _retry_operation(self, operation, max_retries=3, is_vision=False):
        """操作の再試行を処理する"""
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                model_info = "Vision API" if is_vision else "Text API"
                logger.info(f"Gemini {model_info} リクエスト実行 (試行 {retries+1}/{max_retries})")
                
                response = operation()
                
                # レスポンスのステータスコードをログに記録
                if response:
                    logger.info(f"Gemini API レスポンス: ステータスコード={response.status_code}")
                    
                    # エラーが発生した場合、詳細をログに記録
                    if response.status_code != 200:
                        error_detail = ""
                        try:
                            error_json = response.json()
                            error_detail = f", エラー詳細: {error_json}"
                        except:
                            error_detail = f", レスポンス: {response.text[:200]}..."
                            
                        logger.error(f"Gemini API エラー: ステータスコード={response.status_code}{error_detail}")
                        
                        # レート制限エラーの場合は長めに待機
                        if response.status_code == 429:
                            wait_time = min(2 ** retries * 2, 30)  # 指数バックオフ（最大30秒）
                            logger.warning(f"レート制限に達しました。{wait_time}秒待機します...")
                            time.sleep(wait_time)
                    
                # 成功した場合はレスポンスを返す
                if response and response.status_code == 200:
                    return response
                
                # 特定のエラーに対する処理
                if response and response.status_code == 404:
                    logger.error("モデルが見つかりません (404)。別のモデルを試すか設定を確認してください。")
                elif response and response.status_code == 401:
                    logger.error("認証エラー (401)。APIキーが正しいか確認してください。")
                elif response and response.status_code == 400:
                    logger.error("不正なリクエスト (400)。リクエスト形式を確認してください。")
                
                # 再試行する前に待機
                wait_time = 1 * (retries + 1)  # 徐々に待機時間を長くする
                logger.info(f"リトライ前に{wait_time}秒待機...")
                time.sleep(wait_time)
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"接続エラー: {str(e)}")
                time.sleep(2)
            except Exception as e:
                logger.error(f"API呼び出し中に例外発生: {str(e)}")
                logger.error(traceback.format_exc())
                last_error = e
                time.sleep(1)
            
            retries += 1
        
        # すべての再試行が失敗した場合
        logger.error(f"Gemini API呼び出しが{max_retries}回失敗しました。最終エラー: {str(last_error) if last_error else '不明'}")
        return None
    
    def generate_content(self, prompt, temperature=0.7, retry_on_empty=True):
        """
        Gemini APIを使用してテキストを生成する
        """
        if not self.api_key:
            logger.warning("API_KEY未設定のためデモモードでレスポンスを返します")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            demo_result = {
                "choices": [{
                    "message": {
                        "content": f"デモレスポンス（{current_time}）: {prompt[:30]}...",
                    }
                }],
                "created": "1970-01-01T00:00:00.000000Z"  # デモモード識別用のタイムスタンプ
            }
            logger.info(f"デモモードレスポンス生成: timestamp=1970-01-01T00:00:00.000000Z")
            return AttrDict(demo_result)
        
        try:
            logger.info(f"Gemini APIコンテンツ生成: プロンプト長={len(prompt)}")
            
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                logger.error("Gemini APIからのレスポンスが無効です")
                logger.error(f"エラー詳細: {response.status_code if response else 'なし'} - {response.text if response else 'レスポンスなし'}")
                return {"api_error": True, "error": "APIレスポンスエラー"}
                
        except Exception as e:
            logger.error(f"コンテンツ生成中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e)}
    
    def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Gemini Vision APIを使用して画像を分析する"""
        try:
            logger.info(f"Gemini Vision API画像分析: {image_path}")
            
            # 画像をbase64エンコード
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.vision_model}:generateContent?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [
                                {"text": prompt},
                                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                            ]
                        }]
                    }
                ),
                is_vision=True
            )
            
            if response and response.status_code == 200:
                return response.json()
            else:
                logger.error("Gemini Vision APIからのレスポンスが無効です")
                logger.error(f"エラー詳細: {response.status_code if response else 'なし'} - {response.text if response else 'レスポンスなし'}")
                return {"api_error": True, "error": "画像分析APIエラー"}
                
        except Exception as e:
            logger.error(f"画像分析中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e)}
    
    def analyze_scene(self, image_path: str) -> Dict[str, Any]:
        """シーンを分析してメタデータを生成する"""
        try:
            prompt = """
            この画像について詳細に分析し、以下の情報をJSON形式で提供してください:
            {
              "scene_type": "シーンのタイプ（屋内/屋外/都市/自然など）",
              "location": "撮影場所の推測",
              "objects": ["画像内の主要な物体のリスト"],
              "activities": ["画像内で行われている活動"],
              "people_count": 人の数（整数）,
              "time_of_day": "昼/夜/朝/夕方など",
              "weather": "天気の状態（晴れ/雨/曇りなど）",
              "keywords": ["シーンを表す重要なキーワード（5-10個）"]
            }
            JSONフォーマットのみを返してください。説明は不要です。
            """
            
            response = self.analyze_image(image_path, prompt)
            
            if "api_error" in response:
                return response
            
            # レスポンスからJSONを抽出
            try:
                text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                # テキストからJSONブロックを抽出
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                
                return json.loads(json_str)
            except Exception as e:
                logger.error(f"シーン分析のJSON解析に失敗: {str(e)}")
                logger.error(f"原文: {text if 'text' in locals() else 'テキストなし'}")
                logger.error(traceback.format_exc())
                return {"api_error": True, "error": "JSON解析エラー", "scene_type": "不明", "keywords": ["エラー"]}
                
        except Exception as e:
            logger.error(f"シーン分析中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e), "scene_type": "不明", "keywords": ["エラー"]}
    
    def analyze_scene_context(self, images: List[str], transcript: str = "") -> Dict[str, Any]:
        """複数の画像とトランスクリプトから文脈を分析する"""
        try:
            # 最大3枚の画像を選択
            selected_images = images[:3]
            logger.info(f"シーン文脈分析: {len(selected_images)}枚の画像, トランスクリプト長={len(transcript)}")
            
            # 最初の画像を分析
            if selected_images:
                result = self.analyze_scene(selected_images[0])
                
                # エラーチェック
                if "api_error" in result and result.get("api_error") == True:
                    logger.error("シーン文脈分析のための最初の画像分析に失敗")
                    return {"api_error": True, "error": result.get("error", "シーン分析エラー")}
                
                # 追加のコンテキスト分析（複数画像がある場合）
                if len(selected_images) > 1 and transcript:
                    # 複数の画像を使って文脈を強化...（実装はAPIの機能によって異なる）
                    pass
                
                return result
            else:
                logger.error("シーン文脈分析に画像が提供されていません")
                return {"api_error": True, "error": "画像なしエラー", "scene_type": "不明", "keywords": ["エラー"]}
                
        except Exception as e:
            logger.error(f"シーン文脈分析中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e), "scene_type": "不明", "keywords": ["エラー"]}
    
    def generate_editing_suggestions(self, scenes_data: List[Dict], full_transcript: str) -> Dict[str, Any]:
        """シーンデータとトランスクリプトからビデオ編集の提案を生成する"""
        try:
            scenes_str = json.dumps(scenes_data, ensure_ascii=False)
            # トランスクリプトが長すぎる場合は切り詰める
            max_transcript_len = 4000
            if len(full_transcript) > max_transcript_len:
                logger.warning(f"トランスクリプトが長すぎるため切り詰めます ({len(full_transcript)} -> {max_transcript_len}文字)")
                transcript_summary = full_transcript[:max_transcript_len] + "..."
            else:
                transcript_summary = full_transcript
            
            prompt = f"""
            以下のビデオシーンデータとトランスクリプトに基づいて、ビデオの編集方法の提案を行ってください。
            結果はJSON形式で提供してください:
            
            シーンデータ: {scenes_str}
            
            トランスクリプト: {transcript_summary}
            
            以下のJSON形式で返してください:
            {{
              "title": "ビデオのタイトル案",
              "overview": "ビデオの概要（200文字以内）",
              "highlights": ["重要なハイライトシーン（最大5つ）"],
              "suggested_clips": [
                {{
                  "start_time": "開始時間（秒）",
                  "end_time": "終了時間（秒）",
                  "description": "クリップの説明"
                }}
              ],
              "editing_notes": ["編集上の注意点"]
            }}
            """
            
            response = self.generate_content(prompt)
            
            if "api_error" in response:
                logger.error("編集提案の生成に失敗")
                return {"api_error": True, "error": response.get("error", "生成エラー"), "title": "API接続エラー", "overview": "Gemini APIとの接続に問題が発生しました。APIキーとネットワーク接続を確認してください。"}
            
            # レスポンスからJSONを抽出
            try:
                text = response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                # テキストからJSONブロックを抽出
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_str)
                # タイムスタンプを追加
                result["filming_date"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                return result
                
            except Exception as e:
                logger.error(f"編集提案のJSON解析に失敗: {str(e)}")
                logger.error(f"原文: {text if 'text' in locals() else 'テキストなし'}")
                logger.error(traceback.format_exc())
                return {"api_error": True, "error": "JSON解析エラー", "title": "解析エラー", "overview": "レスポンスの解析中にエラーが発生しました。", "filming_date": "1970-01-01T00:00:00.000000Z"}
                
        except Exception as e:
            logger.error(f"編集提案生成中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            # デモモードで使用するフォールバック
            return {
                "api_error": True, 
                "error": str(e), 
                "title": "処理エラー", 
                "overview": f"ビデオ分析中にエラーが発生しました: {str(e)}", 
                "filming_date": "1970-01-01T00:00:00.000000Z"
            }
    
    def generate_video_summary(self, scenes: List[Dict], transcript: str) -> Dict[str, Any]:
        """ビデオシーンとトランスクリプトから要約を生成する"""
        try:
            # シーンとトランスクリプトをJSON文字列に変換
            scenes_str = json.dumps(scenes, ensure_ascii=False)
            
            # トランスクリプトが長すぎる場合は切り詰める
            max_transcript_len = 4000
            if len(transcript) > max_transcript_len:
                logger.warning(f"トランスクリプトが長すぎるため切り詰めます ({len(transcript)} -> {max_transcript_len}文字)")
                transcript_summary = transcript[:max_transcript_len] + "..."
            else:
                transcript_summary = transcript
            
            prompt = f"""
            以下のビデオシーンデータとトランスクリプトに基づいて、ビデオの要約を行ってください。
            
            シーンデータ: {scenes_str}
            
            トランスクリプト: {transcript_summary}
            
            以下のJSON形式で返してください:
            {{
              "title": "ビデオのタイトル",
              "overview": "ビデオの概要（200文字以内）",
              "concepts": ["ビデオのコンセプト/テーマ（最大5つ）"],
              "key_moments": [
                {{
                  "time": "時間（秒）",
                  "description": "重要な瞬間の説明"
                }}
              ],
              "filming_date": "撮影日時の推測（ISO 8601形式）",
              "filming_location": "撮影場所の推測"
            }}
            """
            
            response = self._retry_operation(
                lambda: requests.post(
                    f"{self.base_url}/{self.model}:generateContent?key={self.api_key}",
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
            )
            
            if not response or response.status_code != 200:
                logger.error("ビデオ要約の生成に失敗")
                logger.error(f"エラー詳細: {response.status_code if response else 'なし'} - {response.text if response else 'レスポンスなし'}")
                return {"api_error": True, "title": "API接続エラー", "overview": "Gemini APIとの接続に問題が発生しました。APIキーとネットワーク接続を確認してください。", "filming_date": "1970-01-01T00:00:00.000000Z"}
            
            # レスポンスからJSONを抽出
            try:
                data = response.json()
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                
                # テキストからJSONブロックを抽出
                json_str = text
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    json_str = text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(json_str)
                logger.info(f"ビデオ要約生成成功: {result.get('title', 'タイトルなし')}")
                return result
                
            except Exception as e:
                logger.error(f"ビデオ要約のJSON解析に失敗: {str(e)}")
                logger.error(f"原文: {text if 'text' in locals() else 'テキストなし'}")
                logger.error(traceback.format_exc())
                return {"api_error": True, "error": "JSON解析エラー", "title": "解析エラー", "overview": "レスポンスの解析中にエラーが発生しました。", "filming_date": "1970-01-01T00:00:00.000000Z"}
                
        except Exception as e:
            logger.error(f"ビデオ要約生成中に例外発生: {str(e)}")
            logger.error(traceback.format_exc())
            return {"api_error": True, "error": str(e), "title": "エラー", "overview": "処理中にエラーが発生しました。", "filming_date": "1970-01-01T00:00:00.000000Z"}
