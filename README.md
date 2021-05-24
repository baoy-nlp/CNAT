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
> python3 train.py ${DATA_BIN} --user-dir ${USER_DIR} --save-dir ${SAVE_DIR} --tensorboard-logdir ${LOG_DIR} --ddp-backend=no_c10d --task nat --criterion generic_loss --arch vq_vae_iwslt14 --share-decoder-input-output-embed --mapping-func interpolate --mapping-use output --gradually --gradually-schedule glancing --vq-num 64 --gated-func gated --predictor-layers 5 --predictor-decoder-cls CRF --crf-cls BCRF --crf-num-head 4 --crf-input-last --use-info-exp --use-info-bound --info-z 0.0 --vq-kl 0.0 --vq-alpha 0.5 --vq-beta 0.0 --remove-pos --share-rel-embeddings --block-cls highway --self-attn-cls shaw --enc-self-attn-cls shaw --enc-block-cls highway --max-rel-positions 4 --content-window-size 4 2 0 --content-state-weight 0 0 --content-share-state 1 1 0 --content-layer-weight 0.0 --layer-aggregate-func sum --noise full_mask --optimizer adam --adam-betas (0.9,0.98) --lr 3e-4 --lr-scheduler polynomial_decay --end-learning-rate 1e-5 --warmup-updates 0 --total-num-update 250000 --dropout 0.3 --weight-decay 0 --encoder-learned-pos --pred-length-offset --apply-bert-init --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 2048 --update-freq 1 --save-interval-updates 500 --keep-best-checkpoints 5 --no-epoch-checkpoints --keep-interval-updates 5 --max-update 250000 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --left-pad-source False


# MORE DETAILS ABOUT HOW RUN THIS REPOS WILL COME SOON.
