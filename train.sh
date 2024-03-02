# seq len 2048
export TPU_NAME='v4-64'
export ZONE='us-central2-b'

echo "[local] Killing TPU"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME \
--zone $ZONE --worker=all --command "sudo fuser -k /dev/accel0"

echo "[local] Removing TPU Lock"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME \
--zone $ZONE --worker=all --command "sudo rm -f /tmp/libtpu_lockfile"

echo "[local] Removing screens"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME \
--zone $ZONE --worker=all --command "killall screen"

echo "[local] Git pull"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME --zone $ZONE --worker=all --command \
"source EasyLMenv/bin/activate && \
cd Gemma-EasyLM && git fetch origin && \
git reset --hard origin/main && rm /home/beomi/Gemma-EasyLM/train.sh"

echo "[local] Set runner.sh"

# Log per 128 * 50 steps, matching the gradient accumulation steps = Real 1 step
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME --zone $ZONE --worker=all --command "
cat > /home/beomi/Gemma-EasyLM/runner.sh << 'EOF'
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

python -m EasyLM.models.gemma.gemma_train \
--load_checkpoint=flax_params::/home/beomi/flax_model.msgpack \
--mesh_dim=1,-1,4 \
--dtype=bf16 \
--total_steps=320000 \
--log_freq=128 \
--save_model_freq=999320000 \
--save_milestone_freq=10000 \
--train_dataset.type='json' \
--train_dataset.text_processor.fields='text' \
--train_dataset.json_dataset.seq_length=8192 \
--train_dataset.json_dataset.batch_size=8 \
--train_dataset.json_dataset.path=gs://kodataset/kor_falcon_hq/falcon_korean_stage_012_concat.jsonl \
--optimizer.accumulate_gradient_steps=64 \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.weight_decay=0.1 \
--optimizer.adamw_optimizer.lr=0.00005 \
--optimizer.adamw_optimizer.end_lr=0.000001 \
--optimizer.adamw_optimizer.lr_warmup_steps=10000 \
--optimizer.adamw_optimizer.lr_decay_steps=320000 \
--checkpointer.save_optimizer_state=True \
--checkpointer.float_dtype=bf16 \
--logger.online=True \
--logger.output_dir=gs://kodataset/gemma-checkpoint
EOF
chmod +x /home/beomi/Gemma-EasyLM/runner.sh"

echo "[local] RUN!!!"

gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME --zone us-central2-b --worker=all --command \
"screen -L -d -m bash -i -c 'export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=107374182400; \
source EasyLMenv/bin/activate && cd Gemma-EasyLM; /home/beomi/Gemma-EasyLM/runner.sh'"
