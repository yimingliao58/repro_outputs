#!/usr/bin/env bash
# 分段续接生成长视频 —— Wan2.2 T2V-A14B (第1段) + I2V-A14B (第2段)
#
# 方法:
#   1. 第1段:直接复用已验证 bit-exact 可复现的 out_seed42_run1.mp4(T2V, seed=42, 同一固定 prompt)
#   2. 用 ffmpeg 提取第1段的最后一帧,作为第2段的起始图
#   3. 第2段:I2V-A14B 以该末帧为条件,固定 seed=42,固定 prompt(与第1段相同的 prompt,
#      保持场景连续性),再生成 81 帧
#   4. 拼接:丢弃第2段的第0帧(I2V 的第0帧是条件图本身的重建,和第1段末帧几乎重复,
#      保留会在拼接处产生一帧停顿感的重复帧)
#
# 所有 seed / prompt / sampler 参数都是固定值,写死在本脚本里,不依赖运行时输入,
# 这样整条链可以完整重跑并验证 bit-exact 可复现。

set -euo pipefail

TAG="${1:?用法: $0 <output_tag>   例如: $0 chainA}"

WAN_REPO="/home/lym/wan-vbench/Wan2.2"
T2V_CKPT="/home/lym/wan-vbench/weights/Wan2.2-T2V-A14B"
I2V_CKPT="/home/lym/wan-vbench/weights/Wan2.2-I2V-A14B"
WAN_PYTHON="/home/lym/wan-vbench/envs/wan-gen/bin/python3"
FFMPEG="/home/lym/wan-vbench/envs/wan-gen/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
SEG1_SEED=42
SEG2_SEED=42

SEG1_FILE="${SCRIPT_DIR}/out_seed42_run1.mp4"   # 第1段:复用已有验证过的文件
LAST_FRAME="${SCRIPT_DIR}/chained_${TAG}_seg1_lastframe.png"
SEG2_FILE="${SCRIPT_DIR}/chained_${TAG}_seg2.mp4"
FINAL_FILE="${SCRIPT_DIR}/chained_${TAG}_final.mp4"

if [ ! -s "${SEG1_FILE}" ]; then
  echo "第1段文件不存在: ${SEG1_FILE}"
  echo "请先跑 ./generate_reproducible.sh ${SEG1_SEED} run1 生成第1段"
  exit 1
fi

echo "=== [1/4] 第1段直接复用: ${SEG1_FILE} ==="

echo "=== [2/4] 提取第1段最后一帧作为第2段起始图 ==="
"${FFMPEG}" -y -sseof -0.1 -i "${SEG1_FILE}" -update 1 -q:v 1 "${LAST_FRAME}"
ls -la "${LAST_FRAME}"

echo "=== [3/4] I2V-A14B 生成第2段(条件图 + 固定 prompt + seed=${SEG2_SEED}) ==="
cd "${WAN_REPO}"
"${WAN_PYTHON}" generate.py \
  --task i2v-A14B \
  --size 1280*720 \
  --ckpt_dir "${I2V_CKPT}" \
  --offload_model True \
  --base_seed "${SEG2_SEED}" \
  --sample_solver unipc \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --sample_guide_scale 5.0 \
  --image "${LAST_FRAME}" \
  --prompt "${PROMPT}" \
  --save_file "${SEG2_FILE}"

echo "=== [4/4] 拼接第1段 + 第2段(丢弃第2段第0帧,用 concat filter 统一重编码) ==="
# 注意:这里不能用 concat demuxer + "-c copy" 直接拼字节流。
# seg1(Wan 官方 save_video 输出)和 seg2 如果分别独立编码,H.264 profile 可能不一致
# (实测踩过坑:seg2 用 -crf 0 无损重编码会被 libx264 自动切到 High 4:4:4 Predictive profile,
# 和 seg1 的 High profile 不一致,"-c copy" 硬拼字节流会导致从第2段开始解码花屏)。
# 改用 concat filter 把两段解码后在同一次编码里统一输出,从根源上避免 profile 不一致;
# 同时必须显式 "fps=16" + "-r 16",否则 filter 拿不到源帧率会 fallback 成 25fps,
# 导致输出用重复帧硬凑时长(画面卡顿,不是真实的帧率）。
"${FFMPEG}" -y -i "${SEG1_FILE}" -i "${SEG2_FILE}" -filter_complex \
  "[1:v]select='not(eq(n\,0))',setpts=PTS-STARTPTS[v1]; \
   [0:v][v1]concat=n=2:v=1:a=0,fps=16[outv]" \
  -map "[outv]" -r 16 -c:v libx264 -profile:v high -pix_fmt yuv420p -crf 10 -preset slow \
  "${FINAL_FILE}"

echo "=== 完成 ==="
ls -la "${FINAL_FILE}"
"${FFMPEG}" -i "${FINAL_FILE}" 2>&1 | grep -iE "duration|stream"
