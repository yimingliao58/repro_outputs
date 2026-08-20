# Wan2.2 T2V-A14B 单卡 Bit-Exact 可复现生成记录

记录日期:2026-08-19
验证状态:**已验证通过**(两次独立运行,md5 完全一致)

## 1. 验证结果

同一 prompt + 同一 seed,间隔约 4 小时跑了两次完全独立的生成(每次都重新加载模型、重新走 50 步采样):

```
out_seed42_run1.mp4  18:56 - 22:56 (2026-08-18)
out_seed42_run2.mp4  22:58 - 23:56 (2026-08-18/19)

md5sum:
a31acabeabea21955bcbe962594bb188  out_seed42_run1.mp4
a31acabeabea21955bcbe962594bb188  out_seed42_run2.mp4

cmp out_seed42_run1.mp4 out_seed42_run2.mp4  →  完全一致
```

## 2. 环境版本(精确记录,复现时需对齐)

| 项目 | 版本 |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Compute Capability | 12.0 (sm_120) |
| NVIDIA Driver | 595.84 |
| CUDA | 13.0 |
| Python | 3.12.3 |
| torch | 2.13.0+cu130 |
| torchvision | 0.28.0 |
| torchaudio | 2.11.0 |
| diffusers | 0.39.0 |
| transformers | 4.51.3 |
| numpy | 2.5.2 |
| flash_attn | 2.8.3.post1(源码编译,PyPI 无此架构预编译 wheel) |
| einops | 0.8.2 |
| decord | 0.6.0 |
| Wan2.2 repo commit | `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` (2026-03-17) |

路径:
- 仓库:`/home/lym/wan-vbench/Wan2.2`
- 权重:`/home/lym/wan-vbench/weights/Wan2.2-T2V-A14B`
- venv:`/home/lym/wan-vbench/envs/wan-gen`

跨机器/跨 GPU 架构复现时,即使参数完全相同,浮点运算结合顺序可能不同导致最后几位小数漂移,不保证 bit-exact,但视觉上应无差异。

## 3. 可复现生成命令

固定 prompt(写死,不要每次手输,防止打字误差):

```
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

完整命令(单卡,`CUDA_VISIBLE_DEVICES` 选一张空闲卡):

```bash
cd /home/lym/wan-vbench/Wan2.2
CUDA_VISIBLE_DEVICES=<gpu_id> /home/lym/wan-vbench/envs/wan-gen/bin/python3 generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir /home/lym/wan-vbench/weights/Wan2.2-T2V-A14B \
  --offload_model True \
  --base_seed 42 \
  --sample_solver unipc \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file out_seed42.mp4
```

也可以直接用本目录下的 `generate_reproducible.sh`:

```bash
./generate_reproducible.sh <seed> <output_tag>
```

## 4. 每个设置为什么必须这样(不要随意改动)

- **`--base_seed 42`**:初始化 `torch.Generator`,噪声初始化和每一步 `scheduler.step` 都吃同一个 generator,是可复现的根基。
- **`--sample_solver unipc`**:flow-matching + ODE solver,确定性,不涉及逐步随机噪声注入。`dpm++` 同样是确定性的可以互换,但两者输出不同,复现时要用同一个。
- **不加 `--use_prompt_extend`**(关键):这个开关会用 Qwen 模型重写/扩写 prompt,这一步本身带随机性。就算 seed 锁死,只要开着它,每次喂给 diffusion 的实际文本都不同,结果必然对不上。默认是关闭的,**千万不要在复现实验里手动加这个参数**。
- **`--offload_model True`(必须开,不是可选项)**:A14B 是 MoE 架构,同时有 `high_noise_model` 和 `low_noise_model` 两个专家,合计约 89GB(bf16),这张 96GB 卡上如果不开 offload 会在采样第一步就 OOM(已实测踩坑)。这个开关不影响数值结果,只影响显存/速度。
- **不加 `--convert_model_dtype`**:权重文件本身已是 bf16,加这个开关只是多做一次 CPU 上的显式类型转换,纯粹浪费时间,不影响精度,去掉更快。
- **显式写死 `--sample_steps/--sample_shift/--sample_guide_scale`**:不依赖代码默认值,防止 repo 更新后默认值变化导致"复现失败"却查不出原因。

## 5. 环境搭建踩过的坑(复现环境时会遇到)

1. `wan/__init__.py` 会一次性 eager import T2V/I2V/S2V/TI2V/Animate 全部子模块,即使只用 T2V 也必须装齐 `requirements.txt` + `requirements_s2v.txt` + `requirements_animate.txt` 全部依赖(包括 `einops`、`decord`、`openai-whisper`、`sam2` 等),否则 `import wan` 直接失败。
2. `wan/modules/model.py` 里 `WanSelfAttention`/`WanCrossAttention` 直接调用不带 fallback 的 `flash_attention()` 函数(不是那个有 SDPA fallback 的 `attention()` dispatcher),**没装真正的 flash_attn 会在采样第一步 assert 失败崩溃**,不会自动降级到 PyTorch SDPA。这张 Blackwell 卡(sm_120)PyPI 没有预编译 wheel,需要 `pip install flash-attn --no-build-isolation` 现场源码编译,耗时约 2-2.5 小时(72 个 `.cu` kernel 文件逐个编译,可以设 `MAX_JOBS` 环境变量并行加速)。

## 6. 严格 bit-exact 的额外保险(本次未必须,但建议保留)

`sitecustomize.py`(放在脚本同目录,通过 `PYTHONPATH` 让 Python 启动时自动生效):

```python
import os
import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

消除 GPU kernel 层面浮点非结合性带来的极小漂移。本次实测不加这套也做到了 bit-exact,但跨环境/跨批大小复现时建议保留作为双保险。

## 7. 如何自行验证

```bash
./generate_reproducible.sh 42 run_a
./generate_reproducible.sh 42 run_b
md5sum out_seed42_run_a.mp4 out_seed42_run_b.mp4   # 应完全一致
cmp out_seed42_run_a.mp4 out_seed42_run_b.mp4       # 应无输出(无差异)
```

## 8. 分段续接长视频(2026-08-19)

在单段 T2V 5 秒视频基础上,用 I2V-A14B 续接第 2 段,拼成约 10 秒长视频。方法和参数固定记录如下,便于复现。

**验证状态:已验证通过**(两条独立链路 chainA / chainB,md5 完全一致)。

**结果**:`chained_chainA_final.mp4` / `chained_chainB_final.mp4`,10.06 秒,1280x720,161 帧(81 + 80),两个文件 md5 完全一致(`87494a7a27c890e8a3962548b726e188`),中间产物 I2V 原始输出(`seg2.mp4`)md5 也完全一致(`b7bd10596951280c6cb8da98a3371df1`)——说明不仅单段可复现,I2V 续接 + 拼接这整条流程也是 bit-exact 可复现的。

**流程**:
1. 第 1 段直接复用第 3 节里已验证 bit-exact 的 `out_seed42_run1.mp4`(T2V-A14B,seed=42,固定 prompt),不重新生成。
2. 用 ffmpeg(`imageio_ffmpeg` 自带的 `ffmpeg-linux-x86_64-v7.0.2`)提取第 1 段最后一帧,存为 PNG,作为第 2 段的条件图。
3. 用 **I2V-A14B**(权重 `/home/lym/wan-vbench/weights/Wan2.2-I2V-A14B`,126GB,HuggingFace `Wan-AI/Wan2.2-I2V-A14B`)以该末帧为条件、**同一个固定 prompt**、**固定 seed=42**,生成第 2 段 81 帧。
4. 拼接:丢弃第 2 段的第 0 帧(I2V 的第 0 帧是条件图的重建,和第 1 段末帧几乎重复,保留会在拼接处出现明显停顿感),用 ffmpeg **concat filter** 把两段解码后统一重新编码一次(方法见下面"拼接踩过的坑")。

**固定参数**(与第 1 段完全一致的部分不再重复列出,只列 I2V 特有的):

```bash
--task i2v-A14B
--size 1280*720
--ckpt_dir /home/lym/wan-vbench/weights/Wan2.2-I2V-A14B
--offload_model True
--base_seed 42
--sample_solver unipc
--sample_steps 50
--sample_shift 5.0
--sample_guide_scale 5.0
--image <第1段末帧.png>
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

完整脚本:`generate_long_chained.sh`(所有 seed/prompt/拼接逻辑写死在脚本里,不依赖运行时输入,`./generate_long_chained.sh <tag>` 即可完整重跑整条链)。

### 拼接踩过的坑(重要,别重蹈覆辙)

第一版拼接用的是 `concat demuxer + -c copy`(直接拼字节流,不重新编码):第 1 段(Wan 官方 `save_video` 输出)是 H.264 **High** profile,第 2 段去重复帧时用 `-crf 0` 无损重编码,libx264 在无损模式下会**自动切到 High 4:4:4 Predictive profile**——两段 profile 不一致,`-c copy` 硬拼字节流之后,大多数解码器(包括 VSCode 内置播放器)从第 2 段开始解码花屏,肉眼可见"前 5 秒正常、后 5 秒损坏"。

修复方法:改用 **concat filter**(`[0:v][v1]concat=n=2:v=1:a=0`),把两段解码成原始帧后在同一次编码里统一输出,从根源上避免 profile 不一致;同时踩了第二个坑——filter 拿不到源视频的真实帧率(16fps)会 fallback 成 25fps,导致输出用重复帧硬凑时长(画面卡顿,不是真实帧率),必须显式加 `fps=16` 滤镜 + `-r 16` 输出参数。

**教训**:拼接不同来源/不同编码参数的视频片段时,不要想当然用 `-c copy` 图快,除非你已经用 `ffprobe`/`ffmpeg -i` 确认过两段的 profile、pix_fmt、帧率完全一致。

**已知局限,如实记录**:
- I2V 每步耗时比 T2V 慢约 9%(~73s/it vs ~66.7s/it),原因是 I2V 多了一条图像条件编码通路,这次是在不同物理 GPU(GPU0 vs 之前的 GPU1)上测的,不能完全排除硬件差异的干扰。
- 拼接处丢弃 I2V 第 0 帧是本方案的选择,不是官方推荐做法,官方仓库本身没有内置长视频拼接功能,这个 chaining 思路是社区通用做法,不代表 Wan2.2 官方支持。
- 只做了 2 段(约 10 秒),没有验证更多段拼接后画面是否会累积漂移(角色细节/动作连贯性随拼接次数下降是这类方法的已知问题,还没实测)。
- 最终拼接输出用了 `-crf 10` 有损重编码(而非第 3 节单段视频那种直接来自 `save_video` 的原始编码),所以这份"长视频"的 md5 可复现性验证的是"整条 pipeline(含 ffmpeg 处理步骤)的确定性",不代表画面质量和原始 81 帧输出完全无损等价。

## 9. `frame_num` 显存上限实测(2026-08-20)

只针对 T2V-A14B(14B 参数模型),不涉及 TI2V-5B。用 `--sample_steps 2`(只测显存分配是否成功,不追求生成质量)做二分查找,脚本见 `probe_single_gpu.sh` / `probe_multi_gpu.sh`。

**单卡结论(精确边界)**:`frame_num=93` 能跑,`97` OOM——93 和 97 是相邻的合法值(必须是 4n+1),已经是这套硬件能测到的最细粒度。**93 帧 @ 16fps ≈ 5.81 秒**,比官方默认 81 帧(5.06 秒)多约 15%。

**双卡(FSDP + Ulysses)结论(未精确到边界,但已确认区间)**:129 帧确认 OOM(用 `expandable_segments:True` 排除了碎片化,是真实显存不够);更高的一些取值(161/141)也 OOM。关键发现:**双卡合并显存并没有把上限明显推高**,原因是 T5/VAE 等模块开销不随卡数减少,且 `wan/distributed/sequence_parallel.py` 里 `rope_apply` 的 `pad_freqs` 是按**全局序列长度**(而非单卡分到的那一份)申请临时缓冲区,这部分开销不会因为多一张卡就减半。

**踩到的另一个坑,如实记录**:109 帧在双卡模式下复现性地(2/2 次)报 `NCCL Error 1: unhandled cuda error`,发生在 `wan/distributed/ulysses.py` 的 `all_to_all` 跨卡通信处。用 `CUDA_LAUNCH_BLOCKING=1` 强制同步执行后,109 帧其实能正常跑完——说明这**不是显存问题**,而是默认异步 CUDA 执行模式下,FSDP + Ulysses 的跨卡通信在某些时序下有真实的竞态问题(race condition),可能和这套很新的软硬件栈(CUDA 13.0、Blackwell 驱动 595.84)有关。**这意味着双卡场景下"OOM"报错不能直接当作显存上限的证据,必须先排除这类被 NCCL 掩盖的异步执行问题**,本次没有精测双卡的精确边界就是因为这个原因中止了进一步二分。

## 10. 多 prompt 可复现性验证(2026-08-20)

第 1 节只验证了一个固定 prompt("cats boxing")的可复现性,不能排除是这个 prompt/seed 组合碰巧可复现。这次额外选了 3 个内容差异明显的新 prompt(不同运动类型),单卡、81 帧(5.06s)、每个各自独立跑两遍(run A / run B,间隔数小时,期间模型完全重新加载),对比 md5。

| 场景 | seed | prompt(节选) | 结果 | md5 |
|---|---|---|---|---|
| 狗追飞盘(奔跑) | 100 | "A golden retriever running through a sunlit park chasing a red frisbee..." | ✅ MATCH | `11a1d763938bff7b48a6699ce353b29d` |
| 芭蕾旋转 | 200 | "A ballerina performing a graceful pirouette on a theater stage..." | ✅ MATCH | `a8cc92bf139dfa3c898716f9f6031d4c` |
| 滑板 kickflip(跳跃技巧) | 300 | "A skateboarder performing a kickflip trick on a concrete ramp..." | ✅ MATCH | `78ea421c0cb8e5524f164165e2bcfd95` |

**结论**:3/3 全部 bit-exact 一致,涵盖奔跑、旋转、跳跃三种不同运动类型。说明第 1 节的可复现性结论不是某个 prompt 的偶然结果,在本文档记录的固定配置下(seed 固定、`offload_model=True`、不开 `use_prompt_extend`、`sample_solver=unipc` 等)具有较好的普遍性。

完整 prompt 原文、生成脚本(`generate_repro_param.sh`、`run_multi_prompt_test.sh`)和全部 6 个视频文件见 `multi_prompt_test/` 子目录。
