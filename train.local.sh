export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export CUDA_VISIBLE_DEVICES='1'

python -m EasyLM.models.gemma.gemma_train \
--load_checkpoint=flax_params::/home/jb.lee/tmp/gemma-7b/flax-1/flax_model.msgpack \
--mesh_dim=1,1,2 \
--dtype=bf16 \
--total_steps=320000 \
--log_freq=1 \
--save_model_freq=999320000 \
--save_milestone_freq=10000 \
--train_dataset.type='json' \
--train_dataset.text_processor.fields='text' \
--train_dataset.json_dataset.seq_length=8192 \
--train_dataset.json_dataset.batch_size=1 \
--train_dataset.json_dataset.path=/home/jb.lee/20241018_falcon_modu_text_shuffled.json \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.multiply_by_parameter_scale=True \
--optimizer.adamw_optimizer.weight_decay=0.1 \
--optimizer.adamw_optimizer.lr=0.00005 \
--optimizer.adamw_optimizer.end_lr=0.000001 \
--optimizer.adamw_optimizer.lr_warmup_steps=10000 \
--optimizer.adamw_optimizer.lr_decay_steps=320000 \
--checkpointer.save_optimizer_state=True \
--checkpointer.float_dtype=bf16 \
--logger.online=False \
--logger.output_dir=./gemma-checkpoint
