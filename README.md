# CNAT
Non-autoregressive Translation by Learning Target Categorical Codes

# Requirements
- PyTorch 1.7
- fairseq==0.10.2
- nltk==3.5
- revtok
- tensorboard
- tensorboardX
- tqdm==0.50.2
- sacremoses

# Data preparation
- We keep similar to Fairseq.

# Train IWSLT14
> DATA_BIN=[PATH OF YOUR PROCESSED BIN]
> 
> USER_DIR=[PATH OF YOUR CODE(cnat)]
> 
> SAVE_DIR=[PATH OF YOUR MODEL TARGET]
> 
> LOG_DIR=[PATH OF YOUR LOG TARGET]
> 
> python3 train.py ${DATA_BIN} --user-dir ${USER_DIR} --save-dir ${SAVE_DIR} --tensorboard-logdir ${LOG_DIR} --ddp-backend=no_c10d --task nat --criterion awesome_nat_loss --arch cnat_iwslt14 --share-decoder-input-output-embed --mapping-func interpolate --mapping-use output --share-rel-embeddings --block-cls highway --enc-block-cls highway --self-attn-cls shaw --enc-self-attn-cls shaw --max-rel-positions 4 --noise full_mask --optimizer adam --adam-betas (0.9,0.98) --lr 3e-4 --lr-scheduler polynomial_decay --end-learning-rate 1e-5 --warmup-updates 0 --total-num-update 250000 --dropout 0.3 --weight-decay 0 --encoder-learned-pos --pred-length-offset --apply-bert-init --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 2048 --update-freq 1 --save-interval-updates 500 --keep-best-checkpoints 1 --no-epoch-checkpoints --keep-interval-updates 1 --max-update 250000 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --left-pad-source False --latent-factor 0.5 --num-codes 64 --latent-layers 5 --crf-cls BCRF --crf-num-head 4 --vq-schedule-ratio 0.5 --vq-ema --vq-self-attn-cls abs --vq-block-cls resiudal --vq-share-input-output-embed

# MORE DETAILS ABOUT HOW RUN THIS REPOS WILL COME SOON.
