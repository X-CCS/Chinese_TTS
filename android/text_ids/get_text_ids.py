import os
import re
import numpy as np

from baker import BakerProcessor


if __name__ == '__main__':
    chn_char = "我希望每个人都能够尊重我们的隐私"
    pinyin_char = "wo3 xi1 wang4 mei3 ge4 ren2 dou1 neng2 gou4 zun1 zhong4 wo3 men5 de5 yin3 si1"
    pinyin_char_list = pinyin_char.split()

    processor = BakerProcessor()

    phonemes = processor.get_phoneme_from_char_and_pinyin(chn_char, pinyin_char_list)
    text = " ".join(phonemes)

    print("text is :", text)

    #text_ids = np.asarray(processor.text_to_sequence(text), np.int32)

    #print("text ids is :", text_ids)

