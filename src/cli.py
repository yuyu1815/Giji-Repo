import os

import toml
import argparse
from gijiroku import voice_check, Reazon
import tqdm
from separation import AI_Chat

config = toml.load(open("config.toml", encoding="utf-8"))
prompt = config["PROMPT"]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=str, help='カットする時間')
    parser.add_argument('--model', type=str, help='モデル名')
    parser.add_argument('--out', type=str, help='保存する名前')
    parser.add_argument('file', type=str, help='音声ファイル')
    args = parser.parse_args()
    if args.model is None:
        model = AI_Chat()
    else:
        model = AI_Chat(model_name=args.model)
    cat_time = args.time or config["CAT_TIME"] or 30
    load_file = voice_check(args.file)
    load_file.del_tmp_files()
    Reazon.model_load()
    load_file.cat(cat_time)
    msg = ''
    ai_msg = ''
    for i in tqdm.tqdm(load_file.get_tmp_filenames()):
        tmp_msg = Reazon.execution(i)
        if len(msg) + len(tmp_msg) >= 7000:
            ai_msg += model.prompt_Chat(prompt,msg)
            msg = ""
        else:
            msg += tmp_msg
    ai_msg += model.prompt_Chat(prompt,msg)
    load_file.del_tmp_files()
    ai_msg = ai_msg.replace("。", "。\n")
    print(ai_msg)
    file_name = os.path.splitext(os.path.basename(args.file))[0]
    save_file_path = args.out or f"{file_name}.txt"
    with open(save_file_path, 'w') as f:
        f.write(ai_msg)
    print(f"{save_file_path} を保存しました")

if __name__ == '__main__':
    main()