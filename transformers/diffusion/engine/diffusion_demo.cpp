#include <iostream>
#include <cstdlib>
#include "diffusion/diffusion.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
using namespace MNN::DIFFUSION;

int main(int argc, const char* argv[]) {
    if (argc < 9) {
        MNN_PRINT("=====================================================================================================================\n");
        MNN_PRINT("Usage: ./diffusion_demo <resource_path> <model_type> <memory_mode> <backend_type> <iteration_num> <random_seed> <output_image_name> [image_size] [cfg] [gpu_mem_mode] [precision_mode] [te_on_cpu] <prompt_text>\n");
        MNN_PRINT("  gpu_mem_mode: 0=auto, 1=buffer, 2=image (only for OpenCL)\n");
        MNN_PRINT("  precision_mode: 0=auto, 1=low(FP16), 2=normal(FP32), 3=high(FP32)\n");
        MNN_PRINT("  te_on_cpu: 0=same as unet, 1=text_encoder on CPU (recommended for large models)\n");
        MNN_PRINT("=====================================================================================================================\n");
        return 0;
    }

    auto resource_path = argv[1];
    auto model_type = (DiffusionModelType)atoi(argv[2]);
    auto memory_mode = atoi(argv[3]);
    auto backend_type = (MNNForwardType)atoi(argv[4]);
    auto iteration_num = atoi(argv[5]);
    auto random_seed = atoi(argv[6]);
    auto img_name = argv[7];

    int image_size = 0;
    float cfgScale = 7.5f;
    DiffusionGpuMemoryMode gpuMemoryMode = GPU_MEMORY_BUFFER;
    DiffusionPrecisionMode precisionMode = PRECISION_HIGH;
    int prompt_start = 8;
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            image_size = (int)v;
            prompt_start += 1;
        }
    }
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        float v = strtof(argv[prompt_start], &endptr);
        if (endptr != nullptr && *endptr == '\0') {
            cfgScale = v;
            prompt_start += 1;
        }
    }
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            if (v < 0) v = 0;
            if (v > 2) v = 2;
            gpuMemoryMode = (DiffusionGpuMemoryMode)v;
            prompt_start += 1;
        }
    }
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            if (v < 0) v = 0;
            if (v > 3) v = 3;
            precisionMode = (DiffusionPrecisionMode)v;
            prompt_start += 1;
        }
    }
    bool textEncoderOnCPU = false;
    if (argc > prompt_start + 1) {
        char* endptr = nullptr;
        long v = strtol(argv[prompt_start], &endptr, 10);
        if (endptr != nullptr && *endptr == '\0') {
            textEncoderOnCPU = (v != 0);
            prompt_start += 1;
        }
    }

    std::string input_text;
    for (int i = prompt_start; i < argc; ++i) {
        input_text += argv[i];
        if (i < argc - 1) {
            input_text += " ";
        }
    }
    
    MNN_PRINT("Model resource path: %s\n", resource_path);
    if (model_type == STABLE_DIFFUSION_1_5) {
        MNN_PRINT("Model type is stable diffusion 1.5\n");
    } else if (model_type == STABLE_DIFFUSION_TAIYI_CHINESE) {
        MNN_PRINT("Model type is stable diffusion taiyi chinese version\n");
    } else if (model_type == STABLE_DIFFUSION_ZIMAGE) {
        MNN_PRINT("Model type is ZImage diffusion model\n");
    } else {
        MNN_PRINT("Error: Model type %d not supported, please check\n", (int)model_type);
        return 0;
    }

    if(memory_mode == 1) {
        MNN_PRINT("(Memory Enough) All Diffusion models will be initialized when application enter. with fast initialization\n");
    } else {
        MNN_PRINT("(Memory Lack) Each diffusion model will be initialized when using, freed after using. with slow initialization\n");
    }
    MNN_PRINT("Backend type: %d\n", (int)backend_type);
    MNN_PRINT("Output image name: %s\n", img_name);
    MNN_PRINT("CFG scale: %f\n", cfgScale);
    MNN_PRINT("GPU memory mode: %d\n", (int)gpuMemoryMode);
    MNN_PRINT("Precision mode: %d\n", (int)precisionMode);
    MNN_PRINT("Text encoder on CPU: %s\n", textEncoderOnCPU ? "true" : "false");
    MNN_PRINT("Prompt text: %s\n", input_text.c_str());

    
    std::unique_ptr<Diffusion> diffusion;
    const int numThreads = 4;
    diffusion.reset(Diffusion::createDiffusion(resource_path, model_type, backend_type, memory_mode, image_size,
                                                textEncoderOnCPU, gpuMemoryMode, precisionMode, numThreads));

    diffusion->load();
    
    // callback to show progress
    auto progressDisplay = [](int progress) {
        std::cout << "Progress: " << progress << "%" << std::endl;
    };
    diffusion->run(input_text, img_name, iteration_num, random_seed, cfgScale, progressDisplay);
     
    /*
     when need multi text-generation-image:
     if you choose memory lack mode, need diffusion load with each diffusion run.
     if you choose memory enough mode,  just start another diffusion run, only need diffusion load in first time.
     */
    while(0) {
        if(memory_mode != 1) {
            diffusion->load();
        }
        
        diffusion->run("a big horse", "demo_2.jpg", 20, 42, cfgScale, progressDisplay);
    }
    return 0;
}
