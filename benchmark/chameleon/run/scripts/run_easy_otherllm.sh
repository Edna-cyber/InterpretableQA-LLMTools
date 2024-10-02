#!/bin/bash
#SBATCH --job-name=easyotherllm
#SBATCH -t 24:00:00
#SBATCH --mem=100GB
#SBATCH -p compsci-gpu
#SBATCH --cpus-per-task 8
#SBATCH --output=./easyotherllm.log

export GOOGLE_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
eval "$(conda shell.bash hook)"
conda activate nlp-env

python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-4-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gemini --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine claude --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
