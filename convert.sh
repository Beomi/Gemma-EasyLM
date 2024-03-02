OUTPUT='gemma-ko-7b-dev'
CKPT_PATH='../streaming_train_state_140000'

rm -rf $OUTPUT
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/beomi/$OUTPUT

python convert_easylm_stream_to_hf_safetensors.py $CKPT_PATH

gcloud compute instances stop "instance-3" --zone "us-central2-b" --project "kcbert"
