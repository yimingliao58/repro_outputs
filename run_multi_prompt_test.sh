#!/usr/bin/env bash
# 单卡串行跑 3 个不同 prompt,每个各跑两遍验证 bit-exact 可复现
set -uo pipefail

SCRIPT="/home/lym/wan-vbench/repro_outputs/generate_repro_param.sh"
OUTDIR="/home/lym/wan-vbench/repro_outputs/multi_prompt_test"
RESULT_LOG="${OUTDIR}/results.txt"
mkdir -p "${OUTDIR}"
> "${RESULT_LOG}"

declare -a TAGS=("dog" "ballet" "skate")
declare -a SEEDS=(100 200 300)
declare -a PROMPTS=(
  "A golden retriever running through a sunlit park chasing a red frisbee, dynamic full-body running motion, grass field, shallow depth of field."
  "A ballerina performing a graceful pirouette on a theater stage under a spotlight, flowing white dress, smooth rotational motion, dark background."
  "A skateboarder performing a kickflip trick on a concrete ramp at sunset, dynamic athletic jumping motion, motion blur on wheels, urban skatepark."
)

for i in 0 1 2; do
  TAG="${TAGS[$i]}"
  SEED="${SEEDS[$i]}"
  PROMPT="${PROMPTS[$i]}"

  echo "=========================================="
  echo "[$((i+1))/3] ${TAG} (seed=${SEED}) - run A"
  echo "=========================================="
  CUDA_VISIBLE_DEVICES=0 "${SCRIPT}" "${SEED}" "${TAG}_runA" "${PROMPT}" \
    > "${OUTDIR}/${TAG}_runA.log" 2>&1
  RC_A=$?

  echo "=========================================="
  echo "[$((i+1))/3] ${TAG} (seed=${SEED}) - run B"
  echo "=========================================="
  CUDA_VISIBLE_DEVICES=0 "${SCRIPT}" "${SEED}" "${TAG}_runB" "${PROMPT}" \
    > "${OUTDIR}/${TAG}_runB.log" 2>&1
  RC_B=$?

  FILE_A="${OUTDIR}/${TAG}_runA_seed${SEED}.mp4"
  FILE_B="${OUTDIR}/${TAG}_runB_seed${SEED}.mp4"

  if [ "$RC_A" -ne 0 ] || [ "$RC_B" -ne 0 ]; then
    echo "${TAG}: GENERATION_FAILED (rc_a=$RC_A rc_b=$RC_B)" | tee -a "${RESULT_LOG}"
    continue
  fi

  if [ -s "$FILE_A" ] && [ -s "$FILE_B" ]; then
    MD5_A=$(md5sum "$FILE_A" | awk '{print $1}')
    MD5_B=$(md5sum "$FILE_B" | awk '{print $1}')
    if [ "$MD5_A" == "$MD5_B" ]; then
      echo "${TAG}: MATCH (md5=${MD5_A})" | tee -a "${RESULT_LOG}"
    else
      echo "${TAG}: MISMATCH (a=${MD5_A} b=${MD5_B})" | tee -a "${RESULT_LOG}"
    fi
  else
    echo "${TAG}: MISSING_OUTPUT_FILE" | tee -a "${RESULT_LOG}"
  fi
done

echo "ALL_DONE"
cat "${RESULT_LOG}"
