Implementation of [Non-autoregressive Translation by Learning Target Categorical Codes (NAACL-2021)](https://arxiv.org/abs/2103.11405)

# Environment
- PyTorch 1.7
- fairseq==0.10.2
- nltk==3.5
- revtok
- tensorboard
- tensorboardX
- tqdm==0.50.2
- sacremoses

# Data Preparation
- IWSLT14 German-English & WMT14 English-German: we mostly follow the instruction of the [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation).

# Training scripts
```bash
DATA_BIN=[PATH OF YOUR PROCESSED BIN] 
USER_DIR=[path to latent_nat]
SAVE_DIR=[PATH OF YOUR MODEL TARGET]
LOG_DIR=[PATH OF YOUR LOG TARGET]

# 1-GPU for IWSLT14 German-English
python3 train.py ${DATA_BIN} --user-dir ${USER_DIR} --save-dir ${SAVE_DIR} --tensorboard-logdir ${LOG_DIR} --ddp-backend=no_c10d --task nat --criterion awesome_nat_loss --arch cnat_iwslt14 --share-decoder-input-output-embed --mapping-func interpolate --mapping-use output --share-rel-embeddings --block-cls highway --enc-block-cls highway --self-attn-cls shaw --enc-self-attn-cls shaw --max-rel-positions 4 --noise full_mask --optimizer adam --adam-betas (0.9,0.98) --lr 3e-4 --lr-scheduler polynomial_decay --end-learning-rate 1e-5 --warmup-updates 0 --total-num-update 250000 --dropout 0.3 --weight-decay 0 --encoder-learned-pos --pred-length-offset --apply-bert-init --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 2048 --update-freq 1 --save-interval-updates 500 --keep-best-checkpoints 1 --no-epoch-checkpoints --keep-interval-updates 1 --max-update 250000 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --left-pad-source False --latent-factor 0.5 --num-codes 64 --latent-layers 5 --crf-cls BCRF --crf-num-head 4 --vq-schedule-ratio 0.5 --vq-ema --vq-self-attn-cls abs --vq-block-cls resiudal --vq-schedule-ratio 0.5 --find-unused-parameters --find-unused-parameters

# 4-GPUs for WMT14 English-German
python3 train.py ${DATA_BIN} --user-dir ${USER_DIR} --save-dir ${SAVE_DIR} --tensorboard-logdir ${LOG_DIR} --ddp-backend=no_c10d --task nat --criterion awesome_nat_loss --arch cnat_wmt14 --self-attn-cls shaw --block-cls highway --max-rel-positions 4 --enc-self-attn-cls shaw --enc-block-cls highway --share-rel-embeddings --share-decoder-input-output-embed --mapping-func interpolate --mapping-use output --noise full_mask --apply-bert-init --optimizer adam --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 10000 --warmup-init-lr 1e-07 --min-lr 1e-09 --weight-decay 0.0 --dropout 0.1 --encoder-learned-pos --decoder-learned-pos --pred-length-offset --length-loss-factor 0.1 --label-smoothing 0.0 --log-interval 100 --fixed-validation-seed 7 --max-tokens 8000 --update-freq 1 --save-interval-updates 500 --keep-best-checkpoints 5 --no-epoch-checkpoints --keep-interval-updates 5 --max-update 300000 --num-workers 0 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --left-pad-source False --batch-size-valid 128 --latent-factor 0.5 --num-codes 64 --vq-ema --crf-cls BCRF --crf-num-head 4 --latent-layers 5 --vq-schedule-ratio 0.5 --find-unused-parameters

# Average the best or last-5 checkpoints:
python3 scripts/average_checkpoints.py --inputs ${SAVE_DIR}  \
  --num-update-checkpoints 5 --delete --best \
  --output ${SAVE_DIR}/checkpoint_best_avg.pt

python3 scripts/average_checkpoints.py --inputs ${SAVE_DIR}  \
  --num-update-checkpoints 5 --delete \
  --output ${SAVE_DIR}/checkpoint_avg.pt
```

# Test scripts
```shell
python3 test.py ${DATA_BIN} \
    --user-dir ${USER_DIR} \
    --task nat --beam 1 --remove-bpe --print-step --batch-size "${BATCH}" --quiet \
    --gen-subset test \
    --left-pad-source False \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --path ${SAVE_DIR}/checkpoint_best_avg.pt
```
