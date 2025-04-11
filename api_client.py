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
            # 無音区間を判定する基準となる時間（秒）
            self.chunk_duration = 1.0
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
                
                # セグメントをリストに変換
                segments_list = list(segments)
                
                # セグメントから無音区間と文字起こしを抽出
                scene_boundaries = []
                transcripts = []
                
                for i, segment in enumerate(segments_list):
                    # 各セグメントをトランスクリプトに追加
                    transcripts.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
                    
                    # 次のセグメントとの間に無音区間があるか確認
                    if i < len(segments_list) - 1:
                        next_seg = segments_list[i + 1]
                        gap = next_seg.start - segment.end
                        if gap >= self.chunk_duration:
                            scene_boundaries.append(segment.end)
                
                logger.info(f"文字起こし完了: {len(transcripts)}個のセグメント、{len(scene_boundaries)}個の無音区間")
                
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
            import traceback
            logger.error(traceback.format_exc())
            # エラー時は空の結果を返す
            return {"transcripts": [], "scene_boundaries": []}

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
    
    def generate_content(self, prompt: str) -> str:
        """テキストプロンプトからコンテンツを生成"""
        try:
            logger.info("Geminiコンテンツ生成開始")
            response = self.model.generate_content(prompt)
            
            if response.text:
                logger.info("Geminiコンテンツ生成完了")
                return response.text
            else:
                logger.warning("Geminiコンテンツ生成結果が空です")
                return ""
                
        except Exception as e:
            error_msg = f"Geminiコンテンツ生成中にエラーが発生: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            return f"エラー: {type(e).__name__}"
    
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
            prompt = """
            この画像を簡潔に説明してください。
            映像のシーンとして何が映っているか、最も重要な視覚的特徴だけを50単語以内で説明してください。
            詳細な説明は避け、主要な活動や場面の特徴だけを端的に記述してください。
            例: 「日本の都市部を歩いている様子」「オフィスでの会議の様子」など
            """
            
            response = self.model.generate_content([prompt, img])
            
            if response.text:
                # 返答が長すぎる場合は切り詰める
                description = response.text.strip()
                if len(description) > 100:
                    description = description[:97] + "..."
                logger.info("画像分析完了: 簡潔な説明を生成")
                return description
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
            
    def analyze_scene_context(self, transcript: str, keyframe_path: str = None) -> dict:
        """シーンの文脈をAIを用いて分析する"""
        try:
            prompt = f"""
            以下のトランスクリプトを分析し、シーンについて次の情報を提供してください：
            - location_type（屋内/屋外/交通機関内など）
            - estimated_time_of_day（朝/昼/夜など）
            - weather_conditions（天候の状態）
            - key_activities（主要な活動）
            - emotional_tone（感情トーン）
            - narrative_purpose（シーンの物語上の目的）
            
            トランスクリプト: {transcript}
            
            JSONで回答してください。
            """
            
            if keyframe_path and os.path.exists(keyframe_path):
                response = self.model.generate_content([prompt, Image.open(keyframe_path)])
            else:
                response = self.model.generate_content(prompt)
                
            # レスポンスをパースしてJSON形式に変換
            try:
                analysis = json.loads(response.text)
                return {
                    "location_type": analysis.get("location_type", "不明"),
                    "estimated_time_of_day": analysis.get("estimated_time_of_day", "不明"),
                    "weather_conditions": analysis.get("weather_conditions", "不明"),
                    "key_activities": analysis.get("key_activities", []),
                    "emotional_tone": analysis.get("emotional_tone", "中立"),
                    "narrative_purpose": analysis.get("narrative_purpose", "情報提供")
                }
            except json.JSONDecodeError:
                logger.error("AIからの応答をJSONに変換できませんでした。テキスト形式で返します。")
                return {
                    "location_type": "不明",
                    "estimated_time_of_day": "不明",
                    "weather_conditions": "不明",
                    "key_activities": [],
                    "emotional_tone": "中立",
                    "narrative_purpose": "情報提供",
                    "ai_response": response.text
                }
                
        except Exception as e:
            logger.error(f"シーン文脈分析中にエラーが発生: {str(e)}")
            return {
                "location_type": "不明",
                "estimated_time_of_day": "不明",
                "weather_conditions": "不明",
                "key_activities": [],
                "emotional_tone": "中立",
                "narrative_purpose": "情報提供"
            }
    
    def generate_editing_suggestions(self, node_data: dict) -> dict:
        """編集提案をAIを用いて生成する"""
        try:
            prompt = f"""
            以下の情報からビデオ編集の提案を生成してください：
            
            トランスクリプト: {node_data.get('transcript', '')}
            時間: {node_data.get('time_in', 0)}秒 から {node_data.get('time_out', 0)}秒
            シーン分析: {json.dumps(node_data.get('context_analysis', {}), ensure_ascii=False)}
            
            以下の項目についてJSONで回答してください：
            - highlight_worthy（このシーンがハイライトに値するか、true/false）
            - potential_cutpoint（このシーンをカットすべきか、true/false）
            - b_roll_opportunity（B-rollとして追加すべき映像の提案）
            - audio_considerations（音声に関する考慮事項）
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                suggestions = json.loads(response.text)
                return {
                    "highlight_worthy": suggestions.get("highlight_worthy", False),
                    "potential_cutpoint": suggestions.get("potential_cutpoint", False),
                    "b_roll_opportunity": suggestions.get("b_roll_opportunity", ""),
                    "audio_considerations": suggestions.get("audio_considerations", "")
                }
            except json.JSONDecodeError:
                logger.error("AIからの応答をJSONに変換できませんでした")
                return {
                    "highlight_worthy": False,
                    "potential_cutpoint": False,
                    "b_roll_opportunity": "",
                    "audio_considerations": ""
                }
                
        except Exception as e:
            logger.error(f"編集提案生成中にエラーが発生: {str(e)}")
            return {
                "highlight_worthy": False,
                "potential_cutpoint": False,
                "b_roll_opportunity": "",
                "audio_considerations": ""
            }
    
    def generate_video_summary(self, transcripts: list, nodes: list) -> dict:
        """動画全体の要約をAIを用いて生成する"""
        try:
            all_text = " ".join([t.get("text", "") for t in transcripts])
            
            # ノードから追加情報を抽出
            activities = set()
            emotions = set()
            locations = set()
            
            for node in nodes:
                if hasattr(node, 'context_analysis'):
                    activities.update(node.context_analysis.get("key_activities", []))
                    emotions.add(node.context_analysis.get("emotional_tone", ""))
                    locations.add(node.context_analysis.get("location_type", ""))
            
            prompt = f"""
            以下の情報から動画の要約を生成してください：
            
            トランスクリプト全文: {all_text}
            
            シーン数: {len(nodes)}
            主な活動: {", ".join(activities)}
            場所の種類: {", ".join(locations)}
            感情トーン: {", ".join(emotions)}
            
            以下の項目についてJSONで回答してください：
            - title（動画の端的なタイトル）
            - overview（詳細な概要）
            - topics（主要トピックリスト、配列）
            - filming_date（撮影日、検出できる場合）
            - location（撮影場所）
            - weather（天候）
            - purpose（動画の目的）
            - transportation（移動手段、検出できる場合）
            - starting_point（出発地点、検出できる場合）
            - destination（目的地、検出できる場合）
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                summary = json.loads(response.text)
                
                # VideoSummaryの形式に合わせる
                return {
                    "title": summary.get("title", "無題"),
                    "overview": summary.get("overview", ""),
                    "topics": summary.get("topics", []),
                    "filming_date": summary.get("filming_date", ""),
                    "location": summary.get("location", ""),
                    "weather": summary.get("weather", "不明"),
                    "purpose": summary.get("purpose", ""),
                    "transportation": summary.get("transportation", "不明"),
                    "starting_point": summary.get("starting_point", ""),
                    "destination": summary.get("destination", ""),
                    "scene_count": len(nodes),
                    "total_duration": nodes[-1].time_out if nodes else 0,
                    "gopro_start_time": ""  # このデータはAIでは生成できない
                }
            except json.JSONDecodeError:
                logger.error("AIからの応答をJSONに変換できませんでした")
                return {
                    "title": "要約生成エラー",
                    "overview": "AI要約の生成中にエラーが発生しました",
                    "topics": [],
                    "scene_count": len(nodes),
                    "total_duration": nodes[-1].time_out if nodes else 0
                }
                
        except Exception as e:
            logger.error(f"動画要約生成中にエラーが発生: {str(e)}")
            return {
                "title": "要約生成エラー",
                "overview": f"エラー: {str(e)}",
                "topics": [],
                "scene_count": len(nodes),
                "total_duration": nodes[-1].time_out if nodes else 0
            }
