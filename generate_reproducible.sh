#!/usr/bin/env bash
# 可复现生成脚本 —— Wan2.2 T2V-A14B 单卡版本
#
# 用法:
#   ./generate_reproducible.sh <seed> <output_tag>
#   ./generate_reproducible.sh 42 run1
#   ./generate_reproducible.sh 42 run2   # 用同一个 seed 再跑一次,验证输出是否一致
#
# 本机实际路径(已确认存在,不用改):
#   WAN_REPO = /home/lym/wan-vbench/Wan2.2
#   WAN_CKPT = /home/lym/wan-vbench/weights/Wan2.2-T2V-A14B
#   venv     = /home/lym/wan-vbench/envs/wan-gen (torch 2.13.0+cu130, diffusers 0.39.0)
#   flash_attn 未装,wan/modules/attention.py 会自动 fallback 到
#   torch.nn.functional.scaled_dot_product_attention,能跑,只是慢一点。
#
# 跑之前自己 nvidia-smi 看一眼选空闲的卡,别抢了正在用卡的人。

set -euo pipefail

SEED="${1:?用法: $0 <seed> <output_tag>}"
TAG="${2:?用法: $0 <seed> <output_tag>}"

WAN_REPO="${WAN_REPO:-/home/lym/wan-vbench/Wan2.2}"
WAN_CKPT="${WAN_CKPT:-/home/lym/wan-vbench/weights/Wan2.2-T2V-A14B}"
WAN_PYTHON="${WAN_PYTHON:-/home/lym/wan-vbench/envs/wan-gen/bin/python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# 固定 prompt,写死在这里而不是每次手输,避免手滑打错字导致"看起来复现失败"
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

cd "${WAN_REPO}"

"${WAN_PYTHON}" generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir "${WAN_CKPT}" \
  --offload_model True \
  --base_seed "${SEED}" \
  --sample_solver unipc \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \
  --prompt "${PROMPT}" \
  --save_file "${SCRIPT_DIR}/out_seed${SEED}_${TAG}.mp4"

# 注意:没有加 --use_prompt_extend —— 这是保证可复现的关键,千万别加。
