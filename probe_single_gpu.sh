#!/usr/bin/env bash
# 单卡测 frame_num 上限,用法: CUDA_VISIBLE_DEVICES=<n> ./probe_single_gpu.sh <frame_num> <outdir>
FN="${1:?用法: $0 <frame_num> <outdir>}"
OUTDIR="${2:?用法: $0 <frame_num> <outdir>}"
WAN_REPO="/home/lym/wan-vbench/Wan2.2"
T2V_CKPT="/home/lym/wan-vbench/weights/Wan2.2-T2V-A14B"
WAN_PYTHON="/home/lym/wan-vbench/envs/wan-gen/bin/python3"

mkdir -p "${OUTDIR}"
cd "${WAN_REPO}"
LOG="${OUTDIR}/probe_single_fn${FN}.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 600 "${WAN_PYTHON}" generate.py \
  --task t2v-A14B \
  --size 1280*720 \
  --frame_num "${FN}" \
  --ckpt_dir "${T2V_CKPT}" \
  --offload_model True \
  --base_seed 42 \
  --sample_solver unipc \
  --sample_steps 2 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file "${OUTDIR}/probe_single_fn${FN}.mp4" \
  > "${LOG}" 2>&1

if grep -qiE "OutOfMemoryError|CUDA out of memory" "${LOG}"; then
  echo "RESULT_SINGLE frame_num=${FN}: OOM"
elif grep -qE "Traceback" "${LOG}"; then
  echo "RESULT_SINGLE frame_num=${FN}: OTHER ERROR"
else
  echo "RESULT_SINGLE frame_num=${FN}: OK"
fi
