# qj5090 单机全参数微调 benchmark

这里是 Figure 9 的 qj5090 复现实验入口：两种模型、8 张 GPU、序列长度 32 至 4096，比较 KT-FT、PyTorch FSDP2 + CPU oneDNN、DeepSpeed ZeRO-3 CPU offload 和 MegaTrain。APTMoE 没有在本轮重测。历史实验资料在 [references](references/)；当前图只取 qj5090 的实测数据。

## 登录与实验位置

~~~bash
ssh -J sap1 qj5090
cd /mnt/hurry-bench
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
awk '/MemAvailable:/ {print $2 " KiB"}' /proc/meminfo
~~~

所有启动命令都在 qj5090 上执行。每个 runner 使用 GPU 0–7，启动前检查资源；同一时间只运行一个八卡 case。模型和环境已经放在下列位置：

| 用途 | qj5090 路径 |
|---|---|
| 实验根目录、启动脚本 | /mnt/hurry-bench、/mnt/hurry-bench/tools |
| GLM-4.5-Air-106B 权重 | /mnt/data/models/GLM-4.5-Air |
| Qwen3.5-122B-A10B BF16 权重 | /mnt/data/models/Qwen3.5-122B-A10B |
| 训练数据 | /mnt/hurry-bench/data/fft_real_100.json、dataset_info.json |
| KT、FSDP2 Python 环境 | /mnt/data/venvs/sft-yyj-pro |
| ZeRO、MegaTrain Python 环境 | /mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311 |
| KT 的 Python overlays | /mnt/hurry-bench/py-overlay-contract、py-overlay-qwen |
| 框架源码 | /mnt/hurry-bench/src/accelerate、transformers、llamafactory-overlay、megatrain |
| 单次原始证据 | /mnt/hurry-bench/runs |
| 远端汇总 CSV、启动日志 | /mnt/hurry-bench/results、/mnt/hurry-bench/logs |

本私有仓库 yyj6666667/10.10-benchmark 保存启动脚本、配置、汇总表和小型诊断证据；远端 /mnt/hurry-bench **不是 Git 仓库**。修改本地脚本后，要同步到远端对应位置再运行。原始模型权重留在 qj5090，不纳入仓库。

## 共用口径

两个模型每个 case 都用 8 卡、全量微调、BF16 计算、每卡 batch 1、global batch 8。序列长度依次为 32、64、128、256、512、1024、2048、4096。TPS = 8 × 序列长度 ÷ 计入统计的平均 step 秒数。下文的步数用于**复现当前表格**；常规新测量可统一用 2 步预热 + 2 步正式测量，但应在结果记录中保留实际配置。

脚本的模型参数有两套名称：

| 模型 | KT / ZeRO | FSDP2 + oneDNN / MegaTrain |
|---|---|---|
| GLM-4.5-Air | glm | glm45_air |
| Qwen3.5-122B-A10B | qwen | qwen35_122b |

## 运行单个 case

以下命令都从 /mnt/hurry-bench 执行，以 seq1024 为例；把 1024 换成所需序列长度即可。对应本仓库的入口脚本也可直接点开审核。

### KT-FT

已发布 KT 行采用 15 步、前 5 步预热。设置 KT_BENCH_WARMUP_STEPS 是必要的：runner 的默认预热是 2 步。

~~~bash
KT_BENCH_WARMUP_STEPS=5 bash tools/run_qj_kt_case.sh 1024 15 glm
KT_BENCH_WARMUP_STEPS=5 bash tools/run_qj_kt_case.sh 1024 15 qwen
~~~

入口：[run_qj_kt_case.sh](tools/run_qj_kt_case.sh)。版本、patch、overlay 的准确来源见 [kt_qj_source_revisions.md](tools/kt_qj_source_revisions.md)。各 case 写入 /mnt/hurry-bench/results/glm_kt_performance.csv 或 qwen_kt_performance.csv。

### PyTorch FSDP2 + CPU oneDNN

已发布 seq32–2048 及 GLM seq4096 行为 8 步、前 2 步预热，并启用 BF16 CPU 梯度暂存。Qwen seq4096 行采用已记录的 4 步结果、前 2 步预热。

~~~bash
FSDP2_COMPRESS_CPU_GRADIENTS=1 bash tools/run_qj_fsdp2_onednn_case.sh glm45_air 1024 8 2
FSDP2_COMPRESS_CPU_GRADIENTS=1 bash tools/run_qj_fsdp2_onednn_case.sh qwen35_122b 1024 8 2
FSDP2_COMPRESS_CPU_GRADIENTS=1 bash tools/run_qj_fsdp2_onednn_case.sh qwen35_122b 4096 4 2
~~~

入口：[run_qj_fsdp2_onednn_case.sh](tools/run_qj_fsdp2_onednn_case.sh)；实现：[fsdp2_onednn_full_ft.py](tools/fsdp2_onednn_full_ft.py)。runner 按序列长度启用所需的 CPU 专家重计算、分块 loss 和 GPU 优化器状态卸载。结果写入 /mnt/hurry-bench/results/glm45_air_fsdp2_onednn_performance.csv 或 qwen35_122b_fsdp2_onednn_performance.csv。

### DeepSpeed ZeRO-3 CPU offload

已发布行均为 4 步、前 2 步预热；参数和优化器均卸载至 CPU。实际 ZeRO 配置见 [zero3_cpu_offload.json](tools/zero3_cpu_offload.json)；pin_memory 关闭。

~~~bash
bash tools/run_qj_zero_offload_case.sh glm 1024 4 2
bash tools/run_qj_zero_offload_case.sh qwen 1024 4 2
~~~

入口：[run_qj_zero_offload_case.sh](tools/run_qj_zero_offload_case.sh)。结果写入 /mnt/hurry-bench/results/glm_zero_offload_performance.csv 或 qwen_zero_offload_performance.csv。内存峰值的调查见 [qj5090_zero3_memory_audit.md](results/qj5090_zero3_memory_audit.md)。

### MegaTrain

已发布行均为 4 步、前 2 步预热，并启用 embedding/head GPU 卸载。普通长度直接运行：

~~~bash
MEGATRAIN_OFFLOAD_EMB_HEAD=1 bash tools/run_qj_megatrain_case.sh glm45_air 1024 4 2
MEGATRAIN_OFFLOAD_EMB_HEAD=1 bash tools/run_qj_megatrain_case.sh qwen35_122b 1024 4 2
~~~

入口：[run_qj_megatrain_case.sh](tools/run_qj_megatrain_case.sh)；实现：[megatrain_full_ft.py](tools/megatrain_full_ft.py)。Qwen seq32/64 的原测量另设 MEGATRAIN_DIRECT_SHARED_INIT=0。Qwen seq4096 的 runner 自动使用 checkpoint interval 2 和 3 个 gradient slabs。

GLM seq4096 在当前表格中使用以下 4 步固定 batch 运行的计时，复现时用对应的 alias-slot 入口：

~~~bash
MEGATRAIN_ALIAS_LAYER_SLOTS=1 \
MEGATRAIN_OFFLOAD_EMB_HEAD=1 \
MEGATRAIN_RELEASE_LAYER_SLOT_DURING_HEAD=0 \
MEGATRAIN_DIRECT_SHARED_INIT=0 \
MEGATRAIN_CHECKPOINT_INTERVAL=2 \
MEGATRAIN_REPEAT_BATCH=1 \
  bash tools/run_qj_megatrain_glm4096_alias_case.sh glm45_air 4096 4 2
~~~

这个 runner 将结果保存在单次 run 目录，结束时打印该目录；需要生成同格式 CSV 时执行：

~~~bash
run_dir=/mnt/hurry-bench/runs/这里填上一步打印的目录名
/mnt/sft_yyj_yyj/venvs/qwen3-30b-a3b-deepspeed-cpuadam-py311/bin/python \
  tools/collect_qj_fsdp2_onednn_result.py "$run_dir" > "$run_dir/performance.csv"
~~~

普通 MegaTrain 结果在 /mnt/hurry-bench/results/glm_megatrain_performance.csv 或 qwen_megatrain_performance.csv；GLM seq4096 的该次原始结果在 glm_megatrain_alias_slots_patch_performance.csv。Qwen seq4096 的资源调整见 [qj5090_megatrain_qwen4096_memory_retry.md](results/qj5090_megatrain_qwen4096_memory_retry.md)。

## 找回原始记录与汇总

每次运行会在 /mnt/hurry-bench/runs/ 下创建独立目录，并在终端打印完整路径。先看 run_config.json、command.txt、checksums.txt、versions.txt、launcher.log、exit_code.txt；再看 step_timing 或 case/step_timing.json、resources.json 和 performance.csv。实际存在的文件随框架略有不同。以 run_config.json 和 checksums.txt 为准，不要仅凭脚本当前默认值推断旧测量配置。

仓库的 [results](results/) 中有从远端同步的逐框架 CSV、[Figure 9 数据](results/qj5090_figure9.csv)和[图像](results/qj5090_figure9.png)。CPU RSS 峰值按进程树求和；共享内存在多进程 RSS 中可能重复计数，不能当作整机独占物理内存。实验不保存训练 checkpoint 或优化器状态，留存配置、日志、性能数据及所需诊断证据。

## 完整纯文字性能表

[qj5090_performance_tables.md](results/qj5090_performance_tables.md)
