import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import json
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

# scene_description_generator.pyからの関数インポート
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.scene_description_generator import generate_captions_from_scenario, add_scene_descriptions_to_srt

# 多言語対応のための翻訳データ
TRANSLATIONS = {
    "en": {  # 英語
        "title": "Scene Description Generator",
        "input_file": "Input SRT File:",
        "output_file": "Output SRT File:",
        "scenario_file": "Scenario File (Optional):",
        "browse": "Browse...",
        "api_key_gemini": "Gemini API Key:",
        "api_key_serper": "Serper API Key:",
        "scene_duration": "Scene Duration (seconds):",
        "advanced_options": "Advanced Options",
        "generate_button": "Generate Scene Descriptions",
        "cancel": "Cancel",
        "language": "Interface Language:",
        "output_language": "Output Languages:",
        "processing": "Processing...",
        "completed": "Generation completed!",
        "output_files": "Output files are saved to:",
        "error": "Error",
        "no_input_file": "No input SRT file specified",
        "input_file_not_exist": "Input SRT file does not exist",
        "confirm_title": "Confirm",
        "confirm_message": "Are you sure you want to start the scene description generation?",
        "yes": "Yes",
        "no": "No",
        "scenario_text": "Scenario Text (If no file is selected):",
        "use_scenario_file": "Use Scenario File",
        "use_scenario_text": "Use Scenario Text",
        "disable_web_search": "Disable Web Search (Use only scenario and SRT)",
        "scenario_only_mode": "Scenario & SRT Mode (No web search)",
        "select_languages": "Select output languages:",
        "japanese": "Japanese",
        "english": "English",
        "chinese": "Chinese",
        "korean": "Korean"
    },
    "ja": {  # 日本語
        "title": "シーン説明生成ツール",
        "input_file": "入力SRTファイル:",
        "output_file": "出力SRTファイル:",
        "scenario_file": "シナリオファイル（オプション）:",
        "browse": "参照...",
        "api_key_gemini": "Gemini APIキー:",
        "api_key_serper": "Serper APIキー:",
        "scene_duration": "シーン時間（秒）:",
        "advanced_options": "詳細オプション",
        "generate_button": "シーン説明を生成",
        "cancel": "キャンセル",
        "language": "インターフェース言語:",
        "output_language": "出力言語:",
        "processing": "処理中...",
        "completed": "生成完了！",
        "output_files": "出力ファイルの保存先:",
        "error": "エラー",
        "no_input_file": "入力SRTファイルが指定されていません",
        "input_file_not_exist": "入力SRTファイルが存在しません",
        "confirm_title": "確認",
        "confirm_message": "シーン説明の生成を開始しますか？",
        "yes": "はい",
        "no": "いいえ",
        "scenario_text": "シナリオテキスト（ファイルが選択されていない場合）:",
        "use_scenario_file": "シナリオファイルを使用",
        "use_scenario_text": "シナリオテキストを使用",
        "disable_web_search": "ウェブ検索を無効化（シナリオとSRTのみ使用）",
        "scenario_only_mode": "シナリオ＆SRTモード（ウェブ検索なし）",
        "select_languages": "出力する言語を選択:",
        "japanese": "日本語",
        "english": "英語",
        "chinese": "中国語",
        "korean": "韓国語"
    }
}


class SceneDescriptionGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.language = tk.StringVar(value="ja")  # デフォルト言語は日本語
        
        # 出力言語の選択状態を保持する変数
        self.output_languages = {
            'ja': tk.BooleanVar(value=True),  # 日本語
            'en': tk.BooleanVar(value=False),  # 英語
            'zh': tk.BooleanVar(value=False),  # 中国語
            'ko': tk.BooleanVar(value=False)   # 韓国語
        }
        
        # 言語設定を読み込む（もし以前に保存されていれば）
        self.settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_gen_settings.json")
        self.load_settings()
        
        self.root.title(self.get_text("title"))
        self.root.geometry("800x900")
        self.root.resizable(True, True)
        
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.scenario_file = tk.StringVar()
        self.gemini_api_key = tk.StringVar(value=os.getenv("GEMINI_API_KEY", ""))
        self.serper_api_key = tk.StringVar(value=os.getenv("SERPER_API_KEY", ""))
        self.scene_duration = tk.IntVar(value=10)
        self.use_scenario_file = tk.BooleanVar(value=True)
        self.scenario_text = tk.StringVar(value="")
        self.disable_web_search = tk.BooleanVar(value=True)  # デフォルトでウェブ検索を無効化
        
        self.create_widgets()
        
    def get_text(self, key):
        """現在の言語に基づいてテキストを取得"""
        lang = self.language.get()
        if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
            return TRANSLATIONS[lang][key]
        # フォールバック: 英語または最初に見つかったキー
        if "en" in TRANSLATIONS and key in TRANSLATIONS["en"]:
            return TRANSLATIONS["en"][key]
        for lang in TRANSLATIONS:
            if key in TRANSLATIONS[lang]:
                return TRANSLATIONS[lang][key]
        return key  # キーが見つからない場合はキー自体を返す
        
    def save_settings(self):
        """設定をJSONファイルに保存"""
        try:
            settings = {
                "language": self.language.get(),
                "gemini_api_key": self.gemini_api_key.get(),
                "serper_api_key": self.serper_api_key.get(),
                "scene_duration": self.scene_duration.get(),
                "last_input_file": self.input_file.get(),
                "last_output_file": self.output_file.get(),
                "last_scenario_file": self.scenario_file.get(),
                "disable_web_search": self.disable_web_search.get(),
                "output_languages": {lang: var.get() for lang, var in self.output_languages.items()}
            }
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"設定の保存中にエラーが発生しました: {e}")
    
    def load_settings(self):
        """設定をJSONファイルから読み込み"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                    if "language" in settings:
                        self.language.set(settings["language"])
                    if "gemini_api_key" in settings:
                        self.gemini_api_key = tk.StringVar(value=settings["gemini_api_key"])
                    if "serper_api_key" in settings:
                        self.serper_api_key = tk.StringVar(value=settings["serper_api_key"])
                    if "scene_duration" in settings:
                        self.scene_duration = tk.IntVar(value=settings["scene_duration"])
                    if "last_input_file" in settings and os.path.exists(settings["last_input_file"]):
                        self.input_file = tk.StringVar(value=settings["last_input_file"])
                    if "last_output_file" in settings:
                        self.output_file = tk.StringVar(value=settings["last_output_file"])
                    if "last_scenario_file" in settings:
                        self.scenario_file = tk.StringVar(value=settings["last_scenario_file"])
                    if "disable_web_search" in settings:
                        self.disable_web_search = tk.BooleanVar(value=settings["disable_web_search"])
                    if "output_languages" in settings:
                        for lang, value in settings["output_languages"].items():
                            if lang in self.output_languages:
                                self.output_languages[lang].set(value)
        except Exception as e:
            print(f"設定の読み込み中にエラーが発生しました: {e}")
    
    def change_language(self, *args):
        """言語変更時の処理"""
        # 設定を保存
        self.save_settings()
        # GUIを再構築
        for widget in self.root.winfo_children():
            widget.destroy()
        self.create_widgets()
    
    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 言語選択（右上に配置）
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=(0, 10), anchor=tk.NE)
        
        ttk.Label(lang_frame, text=self.get_text("language")).pack(side=tk.LEFT, padx=(0, 5))
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.language, state="readonly", width=10)
        lang_combo["values"] = ["en", "ja"]
        lang_combo.pack(side=tk.LEFT)
        # 言語変更時のイベント
        self.language.trace_add("write", self.change_language)
        
        # タイトル
        title_label = ttk.Label(main_frame, text=self.get_text("title"), font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # 入力SRTファイル選択
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text=self.get_text("input_file")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text=self.get_text("browse"), command=self.browse_input_file).pack(side=tk.LEFT, padx=(10, 0))
        
        # 出力SRTファイル選択
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text=self.get_text("output_file")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text=self.get_text("browse"), command=self.browse_output_file).pack(side=tk.LEFT, padx=(10, 0))
        
        # シナリオ選択フレーム
        scenario_option_frame = ttk.Frame(main_frame)
        scenario_option_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(scenario_option_frame, text=self.get_text("use_scenario_file"), 
                         variable=self.use_scenario_file, value=True,
                         command=self.toggle_scenario_mode).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(scenario_option_frame, text=self.get_text("use_scenario_text"), 
                         variable=self.use_scenario_file, value=False,
                         command=self.toggle_scenario_mode).pack(side=tk.LEFT)
        
        # シナリオファイル選択
        self.scenario_file_frame = ttk.Frame(main_frame)
        self.scenario_file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.scenario_file_frame, text=self.get_text("scenario_file")).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(self.scenario_file_frame, textvariable=self.scenario_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(self.scenario_file_frame, text=self.get_text("browse"), command=self.browse_scenario_file).pack(side=tk.LEFT, padx=(10, 0))
        
        # シナリオテキスト入力
        self.scenario_text_frame = ttk.Frame(main_frame)
        ttk.Label(self.scenario_text_frame, text=self.get_text("scenario_text")).pack(anchor=tk.W, pady=(0, 5))
        self.scenario_text_entry = tk.Text(self.scenario_text_frame, height=10, width=50, wrap=tk.WORD)
        self.scenario_text_entry.pack(fill=tk.BOTH, expand=True)
        
        # 初期状態の設定
        self.toggle_scenario_mode()
        
        # ウェブ検索の無効化オプション
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=10)
        
        ttk.Checkbutton(search_frame, text=self.get_text("disable_web_search"), 
                        variable=self.disable_web_search, 
                        command=self.toggle_web_search_mode).pack(anchor=tk.W)
        
        # モード説明ラベル
        self.mode_description = ttk.Label(search_frame, 
                                         text=self.get_text("scenario_only_mode"),
                                         font=("Arial", 9, "italic"),
                                         foreground="blue")
        self.mode_description.pack(anchor=tk.W, padx=20)
        
        # 詳細オプション
        self.options_frame = ttk.LabelFrame(main_frame, text=self.get_text("advanced_options"))
        self.options_frame.pack(fill=tk.X, pady=10)
        
        # APIキー設定
        api_frame = ttk.Frame(self.options_frame)
        self.api_frame = api_frame
        
        ttk.Label(api_frame, text=self.get_text("api_key_gemini")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(api_frame, textvariable=self.gemini_api_key, width=40).pack(side=tk.LEFT, expand=True)
        
        self.api_frame2 = ttk.Frame(self.options_frame)
        
        ttk.Label(self.api_frame2, text=self.get_text("api_key_serper")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(self.api_frame2, textvariable=self.serper_api_key, width=40).pack(side=tk.LEFT, expand=True)
        
        # シーン時間設定
        scene_frame = ttk.Frame(self.options_frame)
        scene_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(scene_frame, text=self.get_text("scene_duration")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(scene_frame, from_=5, to=60, increment=5, textvariable=self.scene_duration, width=5).pack(side=tk.LEFT)
        
        # ウェブ検索モードの初期表示を設定
        self.toggle_web_search_mode()
        
        # 出力言語選択フレーム
        language_select_frame = ttk.LabelFrame(main_frame, text=self.get_text("select_languages"))
        language_select_frame.pack(fill=tk.X, pady=10)
        
        # 言語選択チェックボックス
        ttk.Checkbutton(language_select_frame, text=self.get_text("japanese"),
                       variable=self.output_languages['ja']).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(language_select_frame, text=self.get_text("english"),
                       variable=self.output_languages['en']).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(language_select_frame, text=self.get_text("chinese"),
                       variable=self.output_languages['zh']).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(language_select_frame, text=self.get_text("korean"),
                       variable=self.output_languages['ko']).pack(side=tk.LEFT, padx=10)
        
        # ログエリア
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
        # ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text=self.get_text("cancel"), command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text=self.get_text("generate_button"), command=self.start_generation).pack(side=tk.RIGHT)

    def toggle_scenario_mode(self):
        """シナリオモードの切り替え（ファイルかテキスト入力か）"""
        if self.use_scenario_file.get():
            self.scenario_file_frame.pack(fill=tk.X, pady=5)
            self.scenario_text_frame.pack_forget()
        else:
            self.scenario_file_frame.pack_forget()
            self.scenario_text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def toggle_web_search_mode(self):
        """ウェブ検索モードの切り替え"""
        if self.disable_web_search.get():
            # ウェブ検索を無効化する場合
            if hasattr(self, 'api_frame2') and self.api_frame2.winfo_ismapped():
                self.api_frame2.pack_forget()
            self.mode_description.config(foreground="blue")
        else:
            # ウェブ検索を有効化する場合
            self.api_frame.pack(fill=tk.X, padx=10, pady=5)
            self.api_frame2.pack(fill=tk.X, padx=10, pady=5)
            self.mode_description.config(foreground="gray")
        
        # 設定を保存
        self.save_settings()

    def browse_input_file(self):
        file = filedialog.askopenfilename(
            title=self.get_text("input_file"),
            filetypes=[("SRT Files", "*.srt"), ("All Files", "*.*")]
        )
        if file:
            self.input_file.set(file)
            # 出力ファイル名を自動設定
            input_name = os.path.basename(file)
            input_dir = os.path.dirname(file)
            base_name, ext = os.path.splitext(input_name)
            output_name = f"{base_name}_with_scenes{ext}"
            self.output_file.set(os.path.join(input_dir, output_name))
            self.save_settings()

    def browse_output_file(self):
        file = filedialog.asksaveasfilename(
            title=self.get_text("output_file"),
            defaultextension=".srt",
            filetypes=[("SRT Files", "*.srt"), ("All Files", "*.*")]
        )
        if file:
            self.output_file.set(file)
            self.save_settings()

    def browse_scenario_file(self):
        file = filedialog.askopenfilename(
            title=self.get_text("scenario_file"),
            filetypes=[("Text Files", "*.txt *.md"), ("All Files", "*.*")]
        )
        if file:
            self.scenario_file.set(file)
            self.save_settings()

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_generation(self):
        input_file = self.input_file.get()
        output_file = self.output_file.get()
        
        # 入力ファイルの存在確認
        if not input_file:
            messagebox.showerror(self.get_text("error"), self.get_text("no_input_file"))
            return
        
        if not os.path.exists(input_file):
            messagebox.showerror(self.get_text("error"), self.get_text("input_file_not_exist"))
            return
        
        # 確認ダイアログ
        if not messagebox.askyesno(self.get_text("confirm_title"), self.get_text("confirm_message")):
            return
        
        # シナリオテキストかファイルかを取得
        scenario_content = ""
        if self.use_scenario_file.get():
            scenario_file = self.scenario_file.get()
            # シナリオファイルがあれば読み込む
            if scenario_file and os.path.exists(scenario_file):
                try:
                    with open(scenario_file, 'r', encoding='utf-8') as f:
                        scenario_content = f.read()
                except Exception as e:
                    self.log(f"シナリオファイルの読み込みエラー: {str(e)}")
        else:
            # テキストエリアからシナリオを取得
            scenario_content = self.scenario_text_entry.get("1.0", tk.END)
        
        # 処理開始のログ
        self.log_text.delete(1.0, tk.END)
        self.log(f"{self.get_text('processing')}")
        
        # APIキーを環境変数に設定
        os.environ["GEMINI_API_KEY"] = self.gemini_api_key.get()
        os.environ["SERPER_API_KEY"] = "" if self.disable_web_search.get() else self.serper_api_key.get()
        os.environ["DISABLE_WEB_SEARCH"] = "1" if self.disable_web_search.get() else "0"
        
        # 別スレッドで処理を実行
        threading.Thread(
            target=self.run_generation,
            args=(input_file, output_file, scenario_content, self.scene_duration.get()),
            daemon=True
        ).start()

    def run_generation(self, input_file, output_file, scenario_text, scene_duration):
        """シーン説明の生成を実行"""
        try:
            # 選択された言語を取得
            selected_languages = [lang for lang, var in self.output_languages.items() if var.get()]
            
            if not selected_languages:
                messagebox.showerror(self.get_text("error"), "少なくとも1つの出力言語を選択してください。")
                return
            
            # 各言語用の出力ファイルパスを生成
            output_files = {}
            base_name, ext = os.path.splitext(output_file)
            for lang in selected_languages:
                output_files[lang] = f"{base_name}_{lang}{ext}"
            
            # リダイレクト用のクラスは既存のまま使用
            class StdoutRedirector:
                def __init__(self, text_widget):
                    self.text_widget = text_widget
                    self.buffer = ""
                
                def write(self, string):
                    self.buffer += string
                    if "\n" in self.buffer:
                        lines = self.buffer.split("\n")
                        self.buffer = lines[-1]
                        for line in lines[:-1]:
                            self.text_widget.insert(tk.END, line + "\n")
                            self.text_widget.see(tk.END)
                            self.text_widget.update()
                
                def flush(self):
                    if self.buffer:
                        self.text_widget.insert(tk.END, self.buffer)
                        self.text_widget.see(tk.END)
                        self.text_widget.update()
                        self.buffer = ""
            
            # 進捗表示用のテキストウィジェット
            progress_window = tk.Toplevel(self.root)
            progress_window.title(self.get_text("processing"))
            progress_window.geometry("600x400")
            
            text_widget = tk.Text(progress_window, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # 標準出力をリダイレクト
            old_stdout = sys.stdout
            sys.stdout = StdoutRedirector(text_widget)
            
            try:
                # 各言語でシーン説明を生成
                for lang in selected_languages:
                    lang_display = TRANSLATIONS[self.language.get()].get(lang.lower(), lang.upper())
                    print(f"\n=== {lang_display}での生成を開始 ===")
                    if self.use_scenario_file.get() and os.path.exists(self.scenario_file.get()):
                        generate_captions_from_scenario(
                            input_file,
                            output_files[lang],
                            self.scenario_file.get(),
                            scene_duration,
                            self.gemini_api_key.get(),
                            self.serper_api_key.get(),
                            self.disable_web_search.get(),
                            language=lang
                        )
                    else:
                        add_scene_descriptions_to_srt(
                            input_file,
                            output_files[lang],
                            scene_duration,
                            self.gemini_api_key.get(),
                            self.serper_api_key.get(),
                            self.disable_web_search.get(),
                            language=lang
                        )
                
                # 生成完了後の処理
                self.root.after(0, lambda: self.generation_completed(output_files))
            
            finally:
                # 標準出力を元に戻す
                sys.stdout = old_stdout
        
        except Exception as e:
            messagebox.showerror(self.get_text("error"), str(e))
            raise

    def generation_completed(self, output_files):
        """生成完了時の処理"""
        message = self.get_text("output_files") + "\n"
        for lang, file_path in output_files.items():
            lang_display = TRANSLATIONS[self.language.get()].get(lang.lower(), lang.upper())
            message += f"\n{lang_display}: {file_path}"
        messagebox.showinfo(self.get_text("completed"), message)


def main():
    root = tk.Tk()
    app = SceneDescriptionGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main() 