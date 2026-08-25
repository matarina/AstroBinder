# Protein binder 多轮设计 Pipeline

本项目将 RIFgen/RIFdock、RFdiffusion、ProteinMPNN、AlphaFold 3 和 Rosetta 串成“种子对接一次 + 多轮设计筛选”的真实计算流程。

完整的模型细节、代码调用路径、配置说明、运行目录、生产测试统计和已知版本差异见：

[`docs/run_pipeline_model_technical_documentation.md`](../docs/run_pipeline_model_technical_documentation.md)

## 主流程

```text
RIFdock
  → RFdiffusion motif scaffolding
  → ProteinMPNN
  → AF3 单链折叠 + RIF graft
  → Rosetta 界面打分与筛选
  → 下一轮 RFdiffusion
```

## 运行

```bash
cd /mnt/data2/wtk/pipeline
python run_pipeline.py \
  --rifdock-input input \
  --config pipeline/config.yaml
```

主要入口和实现：

- `run_pipeline.py`：命令行入口。
- `pipeline/orchestrator.py`：轮次编排、数据流和步骤级续跑。
- `pipeline/steps/`：将统一上下文翻译成各计算脚本参数。
- `pipeline/scripts/`：真实模型调用、结构转换、graft、Rosetta 打分和筛选。
- `pipeline/config.yaml`：生产参数。
- `pipeline_runs/`：运行产物和日志。

步骤完成后会写 `.step_complete.json`。复用相同 `run_name` 时默认跳过已完成步骤；修改模型、配置或代码后应使用新运行名，或显式设置对应步骤的 `force: true`。
