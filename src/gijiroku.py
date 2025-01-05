#Whisper対応してね
from reazonspeech.k2.asr import load_model, transcribe, audio_from_path
import ffmpeg
import json,os
from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
import glob
import shutil
gpu_use = False
model = None

class voice_check:
    def __init__(self, path):
        self.path = path

    def info(self):
        video_info = ffmpeg.probe(self.path)
        print(json.dumps(video_info,indent=2))

    def get_total_time(self)->float:
        return float(ffmpeg.probe(self.path)["format"]["duration"])

    def cat(self, time: int=0, save_path: str = "./tmp"):
        """
        Time(s)ごとにオーディオをカットする save_pathに保存
        """
        if time <= 10:
            print("timeが10以下か設定されてません")
            return
        format = os.path.splitext(os.path.basename(self.path))[1]
        audio = AudioSegment.from_file(self.path)
        total_time = self.get_total_time()
        print(format)
        last_time = 0
        start = True
        while start:
            if total_time >= time + last_time:
                audio_cut = audio[last_time*1000:(last_time + time)*1000]  # time秒分のオーディオを切り出す
            else:
                audio_cut = audio[last_time*1000:]  # 最後のオーディオを切り出す
                start = False

            formatted_time = self.time_seconds_to_hhmmss(last_time)
            try:
                audio_cut.export(f"{save_path}/cut_{formatted_time}{format}", format=format[1:])
                last_time += time  # 開始位置を更新する
            except Exception as e:
                print("ファイルのフォーマットが適していません")
                print(e)
                exit(1)

    def noise_cancel(self, out_path: str = "src/tmp_no2ise"):
        base_dir_pair = os.path.split(self.path)
        save_dir = os.path.join(base_dir_pair[0], out_path, base_dir_pair[1])
        # オーディオの読み込み
        audio, sr = librosa.load(self.path)
        # ノイズキャンセリング
        audio = nr.reduce_noise(y=audio, sr=sr)
        # オーディオの保存
        sf.write(save_dir, audio, sr)
        # 保存したパスを更新
        self.path = save_dir

    @staticmethod
    def time_seconds_to_hhmmss(seconds:float)->str:
        h = int(seconds / 3600)
        m = int(seconds / 60) % 60
        s = int(seconds % 60)
        f = str(seconds % 60 - s) + "0000"
        return f"{h:02d}-{m:02d}-{s:02d}.{f[2:6]}"
        # ミリ秒を秒に変換
    def get_tmp_filenames(self)->str:
        return glob.glob("./tmp/*")
    def del_tmp_files(self, target_dir=None):
        if target_dir is None:
            target_dir = ["./tmp", "./tmp_no2ise"]
        for f in target_dir:
            try:
                shutil.rmtree(f)
            except:
                pass
            os.mkdir(f)

class Reazon:
    last_end_time = 0  # 最後の終了時間を保持する静的変数
    last_text = ""     # 最後のテキストを保持する静的変数

    @staticmethod
    def model_load():
        global model
        model = load_model()

    @staticmethod
    def execution(audio_file:str, time_threshold=3.0)->str:
        global model
        if model is None:
            print("Model is not loaded.")
            return

        audio = audio_from_path(audio_file)
        ret = transcribe(model, audio)

        # 新しいファイルの開始時に、前回の終了時間を基準時間として設定
        Reazon.file_start_time = Reazon.last_end_time

        previous_end_time = Reazon.last_end_time
        current_text = ""
        ago_start_time = Reazon.last_end_time
        result_text = ""

        for subword in ret.subwords:
            # ファイルごとの相対時間を絶対時間に変換
            actual_time = Reazon.file_start_time + subword.seconds

            if actual_time - ago_start_time > time_threshold:
                if current_text and current_text != Reazon.last_text:
                    start_time_str = voice_check.time_seconds_to_hhmmss(previous_end_time).replace("-", ":")[:8]
                    end_time_str = voice_check.time_seconds_to_hhmmss(ago_start_time).replace("-", ":")[:8]
                    result_text += f"{current_text}"
                    Reazon.last_text = current_text
                current_text = subword.token
                previous_end_time = actual_time
            else:
                current_text += subword.token
            ago_start_time = actual_time

        if current_text and current_text != Reazon.last_text:
            start_time_str = voice_check.time_seconds_to_hhmmss(previous_end_time).replace("-", ":")[:8]
            end_time_str = voice_check.time_seconds_to_hhmmss(ago_start_time).replace("-", ":")[:8]
            result_text += f"{current_text}"
            Reazon.last_text = current_text

        Reazon.last_end_time = ago_start_time
        return result_text

