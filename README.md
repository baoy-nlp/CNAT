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
> 
> /home/data_ti4_c/baoy/experiments/data-bin/iwslt14/deen --user-dir /home/data_ti4_c/baoy/projects/NAT-fairseq/latent_nat --save-dir /home/data_ti4_c/baoy/experiments/checkpoints/iwslt14/deen/CNAT-EXP0 --tensorboard-logdir /home/data_ti4_c/baoy/experiments/logs/iwslt14/deen/CNAT-EXP0 --ddp-backend=no_c10d --task nat --criterion awesome_nat_loss --arch cnat_iwslt14 --share-decoder-input-output-embed --mapping-func interpolate --mapping-use output --share-rel-embeddings --block-cls highway --self-attn-cls shaw --enc-self-attn-cls shaw --enc-block-cls highway --max-rel-positions 4 --noise full_mask --optimizer adam --adam-betas (0.9,0.98) --lr 3e-4 --lr-scheduler polynomial_decay --end-learning-rate 1e-5 --warmup-updates 0 --total-num-update 250000 --dropout 0.1 --weight-decay 0 --encoder-learned-pos --pred-length-offset --apply-bert-init --log-format simple --log-interval 100 --fixed-validation-seed 7 --max-tokens 2048 --update-freq 1 --save-interval-updates 500 --keep-best-checkpoints 5 --no-epoch-checkpoints --keep-interval-updates 5 --max-update 250000 --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --left-pad-source False --latent-factor 1.0 --num-codes 64 --latent-layers 2 --crf-cls BCRF --crf-num-head 4