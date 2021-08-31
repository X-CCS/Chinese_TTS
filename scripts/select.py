import os
import sys

def get_files():
    file_list = []
    folder_path = "./durations"
    files = os.listdir(folder_path)

    for per_file in files:
        file_name = per_file.replace("-durations.npy", "")
        #print("file name is : ", file_name)

        file_list.append(file_name)
    return file_list

def cmd_exe(old_path, new_path):
    cmd_str = ("cp -r %s %s" % (old_path, new_path))
    os.system(r'%s'%(cmd_str))

def cp_files():
    file_name_list = get_files()

    id_path = "../old_train/ids/"
    raw_feat_path = "../old_train/raw-feats/"
    nor_feat_path = "../old_train/norm-feats/"
    energy_path = "../old_train/raw-energies/"
    pitch_path = "../old_train/raw-f0/"
    wav_path = "../old_train/wavs/"

    for per_name in file_name_list:
        print("per name is : ", per_name)
        old_id_path = id_path + per_name + "-ids.npy"
        new_id_path = "./ids/" + per_name + "-ids.npy"

        if os.path.exists(old_id_path) is True:
            cmd_exe(old_id_path, new_id_path)

        old_raw_feats_path = raw_feat_path + per_name + "-raw-feats.npy"
        new_raw_feats_path = "./raw-feats/" + per_name + "-raw-feats.npy"

        if os.path.exists(old_raw_feats_path) is True:
            cmd_exe(old_raw_feats_path, new_raw_feats_path)

        old_nor_feat_path = nor_feat_path + per_name + "-norm-feats.npy"
        new_nor_feat_path = "./norm-feats/" + per_name + "-norm-feats.npy"

        if os.path.exists(old_nor_feat_path) is True:
            cmd_exe(old_nor_feat_path, new_nor_feat_path)

        old_energy_path = energy_path + per_name + "-raw-energy.npy"
        new_energy_path = "./raw-energies/" + per_name + "-raw-energy.npy"

        if os.path.exists(old_energy_path) is True:
            cmd_exe(old_energy_path, new_energy_path)

        old_pitch_path = pitch_path + per_name + "-raw-f0.npy"
        new_pitch_path = "./raw-f0/" + per_name + "-raw-f0.npy"

        if os.path.exists(old_pitch_path) is True:
            cmd_exe(old_pitch_path, new_pitch_path)

        old_wav_path = wav_path + per_name + "-wave.npy"
        new_wav_path = "./wavs/" + per_name + "-wave.npy"

        if os.path.exists(old_wav_path) is True:
            cmd_exe(old_wav_path, new_wav_path)


if __name__ == '__main__':
    #get_duration()
    #get_files()
    cp_files()
