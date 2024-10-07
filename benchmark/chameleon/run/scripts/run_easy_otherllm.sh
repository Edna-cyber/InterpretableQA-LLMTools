#!/bin/bash
#SBATCH --job-name=easyotherllm
#SBATCH -t 24:00:00
#SBATCH --mem=100GB
#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:a6000:1 
#SBATCH --cpus-per-task 8
#SBATCH --output=./easyotherllm.log

: << 'COMMENT'
export GOOGLE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
COMMENT
export OPENAI_API_KEY="..."
eval "$(conda shell.bash hook)"
conda activate nlp-env

python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt cleantext --formula formula1
: << 'COMMENT'
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt interptext --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt interp --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness medium --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness medium --prompt interp --formula formula1
COMMENT
: << 'COMMENT'
python ../evaluate.py --policy_engine gpt-3.5-turbo --hardness easy --prompt cleantext --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-3.5-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt interptext --formula formula1
python ../evaluate.py --policy_engine gpt-3.5-turbo --hardness easy --prompt interptext --formula formula1
COMMENT
: << 'COMMENT'
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gpt-4-turbo --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine gemini --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
python ../run-v2-otherllm.py --label chameleon_chatgpt --policy_engine claude --policy_temperature 0.0 --policy_max_tokens 1000 --hardness easy --prompt clean --formula formula1
COMMENT