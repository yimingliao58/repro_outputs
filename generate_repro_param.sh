#!/usr/bin/env bash
# 可复现生成脚本(参数化版本)—— Wan2.2 T2V-A14B 单卡,81 帧(5.06s)
# 用法: ./generate_repro_param.sh <seed> <tag> <prompt>
# 例如: ./generate_repro_param.sh 100 dogA "A golden retriever running..."

set -euo pipefail

SEED="${1:?用法: $0 <seed> <tag> <prompt>}"
TAG="${2:?用法: $0 <seed> <tag> <prompt>}"
PROMPT="${3:?用法: $0 <seed> <tag> <prompt>}"

WAN_REPO="/home/lym/wan-vbench/Wan2.2"
T2V_CKPT="/home/lym/wan-vbench/weights/Wan2.2-T2V-A14B"
WAN_PYTHON="/home/lym/wan-vbench/envs/wan-gen/bin/python3"
OUTDIR="${OUTDIR:-/home/lym/wan-vbench/repro_outputs/multi_prompt_test}"

mkdir -p "${OUTDIR}"
cd "${WAN_REPO}"

"${WAN_PYTHON}" generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --ckpt_dir "${T2V_CKPT}" \
  --offload_model True \
  --base_seed "${SEED}" \
  --sample_solver unipc \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \
  --prompt "${PROMPT}" \
  --save_file "${OUTDIR}/${TAG}_seed${SEED}.mp4"

# 注意:没有加 --use_prompt_extend —— 保证可复现的关键
