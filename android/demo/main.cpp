#include <iostream>
#include <iterator>
#include <vector>
#include "VoxCommon.h"
#include "TTSFrontend.h"
#include "TTSBackend.h"

using namespace std;

typedef struct 
{
    const char* mapperJson;
    unsigned int sampleRate;
} Processor;

int main(int argc, char* argv[])
{
    //if (argc != 2) 
    //{
    //    fprintf(stderr, "demo wavfile\n");
    //    return 1;
    //}

    //std::string text = argv[1];
    std::string wav_path = "aa.wav";

    Processor proc;
    proc.mapperJson = "../config/baker_mapper.json";
    proc.sampleRate = 24000;

    const char* melgenfile  = "../models/fastspeech2_quan.tflite";
    const char* vocoderfile = "../models/mb_melgan.tflite";

    // Init load model
    TTSBackend ttsbackend(melgenfile, vocoderfile);

    // Process text (text -> phone ids)
    //std::vector<int32_t> phonesIds = ttsfrontend.getPhoneIds();

    // text : 我希望每个人都能够尊重我们的隐私
    // phone ids : wo3 xi1 wang4 mei3 ge4 ren2 dou1 neng2 gou4 zun1 zhong4 wo3 men5 de5 yin3 si1
    // sil ^ uo3 #0 x i1 #0 ^ uang4 #0 m ei3 #0 g e4 #0 r en2 #0 d ou1 #0 n eng2 #0 g ou4 #0 z uen1 #0 zh ong4 #0 ^ uo3 #0 m en5 #0 d e5 #0 ^ in3 #0 s ii1 sil
    // 1 6 195 2 25 78 2 6 176 2 17 60 2 12 56 2 21 64 2 10 148 2 18 69 2 12 151 2 26 183 2 27 146 2 6 195 2 17 67 2 10 57 2 6 120 2 22 108 1
    std::vector<int> list = {1, 6, 195, 2, 25, 78, 2, 6, 176, 2, 17, 60, 2, 12, 56, 2, 21, 64, 2, 10, 148, 2, 18, 69, 2, 12, 151, 2, 26, 183, 2, 27, 146, 2, 6, 195, 2, 17, 67, 2, 10, 57, 2, 6, 120, 2, 22, 108, 1};
    
    std::vector<int32_t> phonesIds;

    for(int i=0; i<list.size(); i++)
        phonesIds.push_back(list.at(i));

    ttsbackend.inference(phonesIds);
    MelGenData mel = ttsbackend.getMel();
    std::vector<float> audio = ttsbackend.getAudio();

    std::cout << "********* Phones' ID *********" << std::endl;

    for (auto iter: phonesIds)
    {
        std::cout << iter << " ";
    }
    std::cout << std::endl;

    std::cout << "********* MEL SHAPE **********" << std::endl;
    for (auto index : mel.melShape)
    {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    std::cout << "********* AUDIO LEN **********" << std::endl;
    std::cout << audio.size() << std::endl;

    VoxUtil::ExportWAV(wav_path, audio, proc.sampleRate);
    std::cout << "Wavfile: " << wav_path << " creats." << std::endl;

    return 0;
}
