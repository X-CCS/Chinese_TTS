import os
import sys

def get_duration():
    all_duration_path = "../../data/align/trimmed-durations/"

    energy_path = "./raw-energies"
    files = os.listdir(energy_path)
    for per_file in files:
        #print("per file is ", per_file)
        file_name = per_file.replace("-raw-energy.npy", "")
        old_dur_file_path = all_duration_path + file_name + "-durations.npy"
        #old_dur_file_path = old_dur_file_path.replace("_", "-")

        new_dur_file_path = "./durations/" + file_name + "-durations.npy"
        
        if os.path.exists(old_dur_file_path) is True:
            print("old path is ", old_dur_file_path)
            print("new path is ", new_dur_file_path)
            
            cmd_str = ("cp -r %s %s" % (old_dur_file_path, new_dur_file_path))
            os.system(r'%s'%(cmd_str))


if __name__ == '__main__':
    get_duration()

