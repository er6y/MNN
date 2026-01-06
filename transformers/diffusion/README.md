# MNN Diffusion Module

MNN Diffusion 模块支持多种扩散模型的推理，包括 Stable Diffusion 1.5、Taiyi Chinese、ZImage 和 LongCat Image Edit。

## 支持的模型类型

| 模型类型 | 枚举值 | 说明 |
|---------|--------|------|
| `STABLE_DIFFUSION_1_5` | 0 | Stable Diffusion 1.5 |
| `STABLE_DIFFUSION_TAIYI_CHINESE` | 1 | Taiyi 中文模型 |
| `STABLE_DIFFUSION_ZIMAGE` | 2 | ZImage 模型 |
| `LONGCAT_IMAGE_EDIT` | 3 | LongCat 图像编辑模型 |

## 编译

```bash
mkdir build && cd build
cmake .. -DMNN_BUILD_DIFFUSION=ON -DMNN_BUILD_LLM=ON -DMNN_BUILD_OPENCV=ON -DMNN_OPENCL=ON
make diffusion_demo -j8
```

**注意**: LongCat 模型需要 `MNN_BUILD_LLM=ON` 来启用 LLM text encoder。

## 使用方法

### 基本用法

```cpp
#include "diffusion/diffusion.hpp"

using namespace MNN::DIFFUSION;

// 创建 Diffusion 实例
auto diffusion = Diffusion::createDiffusion(
    modelPath,           // 模型目录路径
    LONGCAT_IMAGE_EDIT,  // 模型类型
    MNN_FORWARD_OPENCL,  // 后端类型
    1,                   // 内存模式 (0=省内存, 1=高性能, 2=平衡)
    512, 512,            // 输出图像尺寸 (宽, 高)
    true,                // text encoder 在 CPU 运行
    GPU_MEMORY_BUFFER,   // GPU 内存模式
    PRECISION_HIGH,      // 精度模式
    4                    // CPU 线程数
);

// 加载模型
diffusion->load();

// 运行推理
diffusion->run(
    "Make the cat wear a hat",  // 提示词
    "output.png",               // 输出路径
    20,                         // 迭代步数
    42,                         // 随机种子
    4.5f,                       // CFG scale
    nullptr,                    // 进度回调
    "input.png"                 // 输入图像路径 (LongCat 必需)
);
```

### 完整工厂方法 (含 VAE on CPU 和 CFG 模式)

```cpp
auto diffusion = Diffusion::createDiffusion(
    modelPath,
    LONGCAT_IMAGE_EDIT,
    MNN_FORWARD_OPENCL,
    1,                   // 内存模式
    512, 512,            // 输出尺寸
    true,                // text encoder on CPU
    false,               // VAE on CPU
    GPU_MEMORY_BUFFER,   // GPU 内存模式
    PRECISION_HIGH,      // 精度模式
    CFG_MODE_STANDARD,   // CFG 档位
    4                    // 线程数
);
```

## 配置选项

### GPU 内存模式 (`DiffusionGpuMemoryMode`)

| 模式 | 说明 |
|------|------|
| `GPU_MEMORY_AUTO` | 自动选择 |
| `GPU_MEMORY_BUFFER` | OpenCL Buffer 模式 |
| `GPU_MEMORY_IMAGE` | OpenCL Image 模式 (Adreno 推荐) |

### 精度模式 (`DiffusionPrecisionMode`)

| 模式 | 说明 |
|------|------|
| `PRECISION_AUTO` | 自动选择 (ZImage/LongCat 使用 FP32) |
| `PRECISION_LOW` | FP16 精度 |
| `PRECISION_NORMAL` | FP32 精度 |
| `PRECISION_HIGH` | FP32 高精度 |

### CFG 档位模式 (`DiffusionCFGMode`)

CFG 档位用于控制 LongCat 等双 UNet 模型的 Limited Interval CFG 范围。

| 模式 | Sigma 范围 | 说明 |
|------|-----------|------|
| `CFG_MODE_AUTO` | 模型默认 | LongCat: 0.1~0.8 |
| `CFG_MODE_WIDE` | 0.1~0.9 | 最宽，最强引导 |
| `CFG_MODE_STANDARD` | 0.1~0.8 | 标准 |
| `CFG_MODE_MEDIUM` | 0.15~0.7 | 中等 |
| `CFG_MODE_NARROW` | 0.2~0.6 | 窄 |
| `CFG_MODE_MINIMAL` | 0.25~0.5 | 最小引导 |

## LongCat 模型特性

LongCat Image Edit 是一个多模态图像编辑模型，具有以下特点：

1. **LLM Text Encoder**: 使用 Qwen2-VL 作为 text encoder，支持图像+文本的多模态输入
2. **Dual UNet**: 使用双 UNet 架构，支持 Limited Interval CFG
3. **FlowMatch Euler Scheduler**: 使用 FlowMatch Euler 调度器

### 模型目录结构

```
model_dir/
├── config.json              # 主配置文件
├── scheduler_config.json    # 调度器配置
├── unet.mnn                 # UNet 模型
├── vae_decoder.mnn          # VAE 解码器
├── vae_encoder.mnn          # VAE 编码器
└── text_encoder/            # LLM text encoder
    ├── config.json
    ├── tokenizer.txt
    └── llm.mnn
```

## 命令行工具

```bash
./diffusion_demo <model_dir> <output_path> [options]

选项:
  -t <type>      模型类型 (0=SD1.5, 1=Taiyi, 2=ZImage, 3=LongCat)
  -s <steps>     推理步数 (默认 20)
  -r <seed>      随机种子 (默认 -1 随机)
  -c <scale>     CFG scale (默认 7.5)
  -p <prompt>    提示词
  -i <image>     输入图像路径 (LongCat 必需)
  -w <width>     输出宽度
  -h <height>    输出高度
```

## 更新日志

### v2.0 (2025-01)

- **重构**: 移除调试代码 (`MNN_SAVE_INTERMEDIATES`)
- **新增**: 集成 LLM text encoder 到 diffusion.cpp，不再需要外部 `text_encoder_demo`
- **新增**: CFG 档位配置 (`DiffusionCFGMode`)
- **新增**: `vae_on_cpu` 参数
- **新增**: 完整工厂方法支持所有配置选项
- **优化**: 代码复用，确保 z-image 兼容性
