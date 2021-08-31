import os
import sys

def process():
    folder_path = "./libritts/"
    speakers = os.listdir(folder_path)
    for per_speaker in speakers:
        print("cur speaker is ", per_speaker)

        cur_speaker_path = folder_path + per_speaker + "/"
        sub_folders = os.listdir(cur_speaker_path)
        for per_folder in sub_folders:
            per_folder_path = cur_speaker_path + per_folder + "/*"

            cp_cmd_str = ("cp -r %s %s" % (per_folder_path, cur_speaker_path))
            print("cmd_str is : ", cp_cmd_str)
            os.system(r'%s'%(cp_cmd_str))

            rm_folder_path = cur_speaker_path + per_folder
            rm_cmd_str = ("rm -rf %s" % (rm_folder_path))
            os.system(r'%s'%(rm_cmd_str))

def write_data(file_path, str_data):
    with open(file_path, 'a') as file_object:
        file_object.write(str_data) 

def get_text(file_path):
    text = ""
    f = open(file_path)
    readlines = f.readlines()
    for line in readlines:
        text = line

    text = text.replace("\"", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace(".", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    
    text = text.upper()
    return text

def lab_process():
    folder_path = "./data/align/"
    speakers = os.listdir(folder_path)
    for per_speaker in speakers:
        print("cur speaker is ", per_speaker)
        cur_speaker_path = folder_path + per_speaker + "/"
        files = os.listdir(cur_speaker_path)

        file_list = []
        for per_file in files:
            if ".tsv" in per_file:
                continue

            cur_file_name = per_file.split(".")[0]
            if cur_file_name not in file_list:
                file_list.append(cur_file_name)

        # process per file
        for per_file in file_list:
            print("per file is :", per_file)
            originin_text_file_path = cur_speaker_path + per_file + ".original.txt"
            normal_text_file_path = cur_speaker_path + per_file + ".normalized.txt"

            new_text_path = cur_speaker_path + per_file + ".lab"

            text_content = get_text(normal_text_file_path)

            print("text_content is ", text_content)
            write_data(new_text_path, text_content)

            origin_cmd_str = ("rm -rf %s" % (originin_text_file_path))
            normal_cmd_str = ("rm -rf %s" % (normal_text_file_path))
            os.system(r'%s'%(origin_cmd_str))
            os.system(r'%s'%(normal_cmd_str))


if __name__ == "__main__":
    #process()
    lab_process()

