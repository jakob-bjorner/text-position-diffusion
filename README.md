# Plaid: Likelihood-Based Diffusion Language Models

This repository contains code for training and evaluating the models in the paper *Likelihood-Based Diffusion Language Models*.

![Figure 1 from the Likelihood-Based Diffusion Language Models paper.](figure1.png)

## Installing requirements

Jakob's notes:

python sample.py --weights_path=/srv/flash2/jbjorner3/plaid-model/plaid1b_weights --dim=2048 --n_blocks=24 --n_heads=32 --seq_len=1024

sh cuda_11.8.0_520.61.05_linux.run --toolkit --toolkitpath=/srv/flash2/jbjorner3/cuda/toolkit --samples --samplespath=/srv/flash2/jbjorner3/cuda/samples --tmpdir=/srv/flash2/jbjorner3/cuda/


/coc/flash9/jbjorner3/miniforge3/envs/plaid-v2-2/lib/libnvrtc.so.11.8.89

export PATH=$PATH:/srv/flash2/jbjorner3/cuda/toolkit/bin

export LD_LIBRARY_PATH=/srv/flash2/jbjorner3/cuda/toolkit/lib64

export CUDA_HOME=/srv/flash2/jbjorner3/cuda/toolkit/

export CUDA_PATH=/srv/flash2/jbjorner3/cuda/toolkit/


This codebase requires PyTorch 2.0 and a few fused CUDA kernels that need to be installed manually. Most of the dependencies can be installed automatically:
```
pip install -r requirements.txt
```

Install FlashAttention with fused MLP and rotary embedding kernels:
```
git clone https://github.com/HazyResearch/flash-attention.git
pip install ./flash-attention
pip install ./flash-attention/csrc/rotary
pip install ./flash-attention/csrc/fused_dense_lib
```

Install NVIDIA Apex with fused kernels:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Generating samples from Plaid 1B

First download the weights from here: [Plaid 1B Weights Download Page](https://github.com/igul222/plaid/releases/tag/v1.0.0)

Extract them:
```
cat plaid1b_weights.tar.gz.* | tar xvzf -
```

Then run the sampling code:

```
python sample.py --weights_path=/srv/flash2/jbjorner3/plaid-model/plaid1b_weights --dim=2048 --n_blocks=24 --n_heads=32 --seq_len=1024

python sample.py --weights_path=. --dim=384 --n_blocks=16 --n_heads=6 --seq_len=256 --just_unconditional=True
```

## Computing zero-shot likelihoods

This repository supports computing zero-shot likelihoods on six datasets: Penn TreeBank, enwik8, text8, WikiText2, WikiText103, and the 1 Billion Word corpus.
To compute likelihood for one of these datasets, specify the dataset path in the corresponding constant at the top of `lib/datasets.py`. Then run this command (e.g. for WikiText2):

```
python train.py --weights_path=/srv/flash2/jbjorner3/plaid-model/plaid1b_weights --dim=2048 --n_blocks=24 --n_heads=32 --seq_len=1024 --dataset=wikitext2 --val_batch_size=1 --steps=0
```

## Training Plaid models

1. Download OpenWebText2 from here: [OpenWebText2 Download](https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar).
2. Update the `OPENWEBTEXT2_DATA_DIR` constant in `lib/datasets.py` with the path to the extracted files.
3. Run the OpenWebText2 preprocessing script:
   ```
   python -m misc.owt2_preprocess
   ```
4. Run the training script:
   ```
   python train.py
   ```

By default, this trains a small model (16 layers, dim 384, sequence length 256, 92K steps at batch size 256) which should take under a day on an 80GB A100. You can change these hyperparameters by passing different options to `train.py`.
   
If you don't have enough memory to train the model with default settings, you can enable gradient accumulation. The following commands should produce equivalent results:
```
python train.py # defaults to grad_accum_steps=1, batch_size=256
python train.py --grad_accum_steps=2 --batch_size=128
python train.py --grad_accum_steps=4 --batch_size=64

python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path=CHANGE_TO_RUN_NAME --diffusion_mode=[token|BoW|see options in train.py] --BoW_cumsum_gamma=[val between 0 and 1]
python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path=BoW_embedding_reconst_first_no_rescale_diff_lm_norms_v_1_g=5e-1 --hook_freq=5000 --diffusion_mode=BoW_embedding_reconst_first_no_rescale_diff_lm_norms --BoW_cumsum_gamma=0.5
```
python sample.py --weights_path="BoW_embedding_reconst_first_no_rescale_diff_lm_norms_v_1_g=5e-1" --dim=384 --n_blocks=16 --n_heads=6 --seq_len=256 --just_unconditional=True --diffusion_mode=BoW_embedding_reconst_first_no_rescale_diff_lm_norms --BoW_cumsum_gamma=0.5


python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path=hype_BoW_embedding_reconst_first_lr=0.0027_embeddim=16_wd=0.0015_g=0_rconstw=1.25_cosine --hook_freq=5000 --diffusion_mode=BoW_embedding_reconst_first_no_rescale_diff_lm_norms --BoW_cumsum_gamma=0.0 --reconst_weight=1.25 --lr=0.0027 --weight_decay=0.0015 --lr_scheduler=cosine

hype_BoW_embedding_reconst_first_lr=0.0004310754645272438_embeddim=16_wd=0.026147239888479386_g=0_rconstw=2.202273629452468
python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path=hype_BoW_embedding_reconst_first_lr=0.0004310754645272438_embeddim=16_wd=0.026147239888479386_g=0_rconstw=2.202273629452468 --hook_freq=5000 --diffusion_mode=BoW_embedding_reconst_first_no_rescale_diff_lm_norms --BoW_cumsum_gamma=0.0 --reconst_weight=2.202273629452468 --lr=0.0004310754645272438 --weight_decay=0.026147239888479386

hype_BoW_embedding_reconst_first_lr=0.0011371032911670497_embeddim=16_wd=0.0013817219447644924_g=0_rconstw=0.7915528412398056
python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path=hype_BoW_embedding_reconst_first_lr=0.0011371032911670497_embeddim=16_wd=0.0013817219447644924_g=0_rconstw=0.7915528412398056 --hook_freq=5000 --diffusion_mode=BoW_embedding_reconst_first_no_rescale_diff_lm_norms --BoW_cumsum_gamma=0.0 --reconst_weight=0.7915528412398056 --lr=0.0011371032911670497 --weight_decay=0.0013817219447644924

for linear now the best is:
'lr': 0.0031041041709632096, 'embed_dim': 16, 'weight_decay': 0.00034741448485511937, 'BoW_cumsum_gamma': 0, 'reconst_weight': 0.681476297799672

python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path="diffusion_hype_model_saves/BoW_embedding_double_logit_reg_lr=0.0028_embeddim=16_wd=6.8e-05_g=0_rconstw=0.83_rconst2w=0.31" --wandb_run_name="BoW_embedding_double_logit_reg_lr=0.0028_embeddim=16_wd=6.8e-05_g=0_rconstw=0.83_rconst2w=0.31" --lr_scheduler=cosine --gamma_1=3 --gamma_0=-1 --diffusion_mode=BoW_embedding_double_logit_reg --weight_decay=0.000068 --reconst_weight=0.83 --reconst_secondary_weight=0.31 --lr=0.0028 --BoW_cumsum_gamma=0

python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path="diffusion_hype_model_saves/BoW_embedding_reconst_first_pred_x0" --wandb_run_name="BoW_embedding_reconst_first_pred_x0" --lr_scheduler=cosine  --diffusion_mode=BoW_embedding_reconst_first_pred_x0 --BoW_cumsum_gamma=0


python train.py --grad_accum_steps=2 --batch_size=128  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg  --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0" --reconst_secondary_weight=0 --BoW_cumsum_gamma=0 --lr_scheduler=cosine

x_reconst2 loss scaled to 0



python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path="diffusion_hype_model_saves/BoW_embedding_reconst_first_pred_x0_embed=32" --wandb_run_name="BoW_embedding_reconst_first_pred_x0_embed=32" --lr_scheduler=cosine  --diffusion_mode=BoW_embedding_reconst_first_pred_x0 --BoW_cumsum_gamma=0 --embed_dim=32


----- trying different gama values for BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0 --lr_scheduler=linear


python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.25" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.25" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0.25 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.5" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.5" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0.5 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.75" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.75" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0.75 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.95" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0.95" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0.95 --lr_scheduler=linear

--------- adding an l2 ontop of the embedding matrix. 
python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0_regemb=0.1" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconstw=0_g=0_regemb=0.1" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0 --lr_scheduler=linear --embed_regularizer_lambda=0.1

--------- try the norm on the x_reconst for g=0
python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_norm_ret_embeddings_rconstw=0_g=0" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_norm_ret_embeddings_rconstw=0_g=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_norm_ret_embeddings --reconst_secondary_weight=0 --BoW_cumsum_gamma=0 --lr_scheduler=linear 



--------- rework the flags to have less complication. (this failed. Need to solve bug in how I handle the secondary logits)
python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_double_logit_reg_rconstw=0.05_g=0" --wandb_run_name="BoW_embedding_pred_x0_double_logit_reg_rconstw=0.05_g=0" --diffusion_mode=BoW_embedding_pred_x0_double_logit_reg --reconst_weight=0.05 --BoW_cumsum_gamma=0 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_double_logit_reg_rconstw=0_g=0" --wandb_run_name="BoW_embedding_pred_x0_double_logit_reg_rconstw=0_g=0" --diffusion_mode=BoW_embedding_pred_x0_double_logit_reg --reconst_weight=0 --BoW_cumsum_gamma=0 --lr_scheduler=linear

sampling from this model:
python sample.py --weights_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_double_logit_reg_rconstw=0_g=0" --dim=384 --n_blocks=16 --n_heads=6 --seq_len=256 --just_unconditional=True --diffusion_mode=BoW_embedding_pred_x0_double_logit_reg --BoW_cumsum_gamma=0 --use_secondary_logits=True

--------- adjust weight on secondary logits
python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0_g=0" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0_g=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0 --BoW_cumsum_gamma=0 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0.1_g=0" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0.1_g=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0.1 --BoW_cumsum_gamma=0 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0.05_g=0" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0.05_g=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0.05 --BoW_cumsum_gamma=0 --lr_scheduler=linear

python train.py --grad_accum_steps=4 --batch_size=64  --model_save_path="diffusion_hype_model_saves/BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0.01_g=0" --wandb_run_name="BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg_rconst2w=0.01_g=0" --diffusion_mode=BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg --reconst_secondary_weight=0.01 --BoW_cumsum_gamma=0 --lr_scheduler=linear

