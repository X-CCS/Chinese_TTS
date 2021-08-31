import os
import sys

def write_data(init_phone_list, next_phone_list):
    file_path = "new_pinyin.py"
    with open(file_path, 'a') as file_object:
        file_object.write("initials = [\n")
        for per_phone in init_phone_list:
            str_data = "    \"" + per_phone + "\",\n"
            file_object.write(str_data)
        file_object.write("]\n")

        file_object.write("\n")

        file_object.write("finals = [\n")
        for per_phone in next_phone_list:
            str_data = "    \"" + per_phone + "\",\n"
            file_object.write(str_data)
        
        file_object.write("]\n")
        file_object.write("\n")

        # write rr
        file_object.write("valid_symbols = initials + finals + [\"rr\"]")

def summary_phones(file_path):
    init_phones_list = []
    next_phones_list = []

    f = open(file_path)

    readlines = f.readlines()
    for line in readlines:
        line = line.rstrip("\n")

        data = line.split()
        if len(data) < 2:
            continue

        phones = data[1:]
        for per_phone in phones:
            if per_phone is not "rr":
                if "1" in per_phone:
                    if per_phone not in next_phones_list:
                        next_phones_list.append(per_phone)
                elif "2" in per_phone:
                    if per_phone not in next_phones_list:
                        next_phones_list.append(per_phone)
                elif "3" in per_phone:
                    if per_phone not in next_phones_list:
                        next_phones_list.append(per_phone)
                elif "4" in per_phone:
                    if per_phone not in next_phones_list:
                        next_phones_list.append(per_phone)
                elif "5" in per_phone:
                    if per_phone not in next_phones_list:
                        next_phones_list.append(per_phone)
                elif "6" in per_phone:
                    if per_phone not in next_phones_list:
                        next_phones_list.append(per_phone)
                else:
                    if per_phone not in init_phones_list:
                        init_phones_list.append(per_phone)

    init_phones_list.sort()
    next_phones_list.sort()
    write_data(init_phones_list, next_phones_list)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("example: python make_pinyin.py ./lexicon/pinyin-lexicon-r.txt")
        exit()

    file_path = sys.argv[1]
    summary_phones(file_path)
