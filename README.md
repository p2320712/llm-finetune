# LLM Fine-tuning with QLoRA

基于 QLoRA 4-bit 量化微调 Qwen2.5-7B-Instruct 的中文指令微调项目。

## 项目亮点

- QLoRA 4-bit 量化微调：仅 5.2GB 显存即可微调 7B 模型
- 自定义中文数据集：覆盖知识问答、代码生成、文本创作、逻辑推理
- 完整训练流程：数据生成 - 模型下载 - QLoRA 微调 - 效果测试

## 技术栈

- 基础模型：Qwen2.5-7B-Instruct
- 微调方法：QLoRA (4-bit NF4 量化 + LoRA)
- 训练框架：Transformers + PEFT + BitsAndBytes
- 硬件：NVIDIA RTX 4060 Laptop (8GB VRAM)

## 项目结构

`
llm-finetune/
  data/
    gen_dataset.py    # 数据生成脚本
    train.json        # 训练集 (55条)
    val.json          # 验证集 (7条)
  train_qlora.py      # QLoRA 训练脚本
  test_model.py       # 模型测试脚本
  saves/              # 微调产物（gitignore）
`

## QLoRA 配置

| 参数 | 值 |
|------|-----|
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Learning Rate | 1e-4 |
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Target Modules | q/k/v/o_proj, gate/up/down_proj |

## 训练结果

| 指标 | 值 |
|------|-----|
| 训练时间 | ~1 分钟 |
| 显存占用 | 5.2 GB |
| 可训练参数占比 | 0.26%% |
| Train Loss | 1.765 -> 1.074 |
| Eval Loss | 1.274 |

## License

MIT
