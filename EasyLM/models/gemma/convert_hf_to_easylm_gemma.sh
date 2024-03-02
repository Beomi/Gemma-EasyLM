export PYTHONPATH=$(realpath ../../../):$PYTHONPATH

python convert_hf_to_easylm_gqa.py \
--checkpoint_dir /data/jb.lee/tmp/gemma-7b \
--output_file ./easylm_gemma-7b_bf16.stream \
--model_size gemma-7b \
--streaming
