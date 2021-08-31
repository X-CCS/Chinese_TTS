import os
import sys

def rename_files(folder_path):
    file_list = []
    files = os.listdir(folder_path)

    for per_file in files:
        pre = per_file[0:2]
        #print("per is :", pre)
        if "00" in str(pre):
            print("file name is ", per_file)
            
            # rename
            old_file_name = folder_path + per_file
            new_file_name = folder_path + "aishu-" + per_file

            cmd_str = ("mv %s %s" % (old_file_name, new_file_name))
            os.system(r'%s'%(cmd_str))

def rename_files_SS(folder_path):
    file_list = []
    files = os.listdir(folder_path)

    for per_file in files:
        pre = per_file[0:2]
        if "SS" in str(pre):
            speaker_name = per_file[0:7]

            old_file_name = folder_path + per_file
            new_file_name = folder_path + speaker_name + "-"  + per_file

            print("new name is :", new_file_name)

            cmd_str = ("mv %s %s" % (old_file_name, new_file_name))
            os.system(r'%s'%(cmd_str))

def rename_files_dudu(folder_path):
    files = os.listdir(folder_path)

    for per_file in files:
        if "ibook" in str(per_file):
            print("file name is : ", per_file)

            old_file_name = folder_path + per_file
            new_file_name = folder_path + "dudu" + "-"  + per_file

            print("new name is :", new_file_name)

            cmd_str = ("mv %s %s" % (old_file_name, new_file_name))
            os.system(r'%s'%(cmd_str))


def rename():
    #file_name_list = get_files()

    durations_path = "./durations/"
    rename_files(durations_path)
    rename_files_SS(durations_path)
    #rename_files_dudu(durations_path)

    id_path = "./ids/"
    rename_files(id_path)
    rename_files_SS(id_path)
    #rename_files_dudu(id_path)

    raw_feat_path = "./raw-feats/"
    rename_files(raw_feat_path)
    rename_files_SS(raw_feat_path)
    #rename_files_dudu(raw_feat_path)

    nor_feat_path = "./norm-feats/"
    rename_files(nor_feat_path)
    rename_files_SS(nor_feat_path)
    #rename_files_dudu(nor_feat_path)

    energy_path = "./raw-energies/"
    rename_files(energy_path)
    rename_files_SS(energy_path)
    #rename_files_dudu(energy_path)

    pitch_path = "./raw-f0/"
    rename_files(pitch_path)
    rename_files_SS(pitch_path)
    #rename_files_dudu(pitch_path)

    wav_path = "./wavs/"
    rename_files(wav_path)
    rename_files_SS(wav_path)
    #rename_files_dudu(wav_path)


if __name__ == '__main__':
    rename()
