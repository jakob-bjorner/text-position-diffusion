"""
File for running hyper parameter sweeps on BoW implementation of plaid.


Manual:
 see readme for up to date launch;
 eg python train.py --grad_accum_steps=2 --batch_size=128 --model_save_path="diffusion_hype_model_saves/BoW_embedding_double_logit_reg_lr=0.0028_embeddim=16_wd=6.8e-05_g=0_rconstw=0.83_rconst2w=0.31" --wandb_run_name="BoW_embedding_double_logit_reg_lr=0.0028_embeddim=16_wd=6.8e-05_g=0_rconstw=0.83_rconst2w=0.31" --lr_scheduler=cosine --gamma_1=3 --gamma_0=-1 ??? --diffusion_mode=BoW_embedding_double_logit_reg --weight_decay=0.000068 --reconst_weight=0.83 --reconst_secondary_weight=0.31 --lr=0.0028 --BoW_cumsum_gamma=0

"""
from train import main
import lib.ddp

import nevergrad as ng
import submitit

if __name__ == "__main__":
    inst = ng.p.Instrumentation(
        lr = 1.4e-3, #ng.p.Scalar(init=1.4e-3, lower=0.0001, upper=0.01),
        embed_dim = 16, # ng.p.Scalar(init=16, lower=8, upper=128).set_integer_casting(),
        weight_decay = 4e-5,#ng.p.Scalar(init=4e-5, lower=4e-7, upper=2e-2),
        BoW_cumsum_gamma = 0, # ng.p.Scalar(init=0.5, lower=.5, upper=1),
        reconst_weight = 1, #ng.p.Scalar(init=0.7, lower=0.001, upper=3.0), ####
        reconst_secondary_weight = 0, # ng.p.Scalar(init=0.3, lower=0.001, upper=3.0),
        weight_decay_embed = ng.p.Scalar(init=4e-5, lower=4e-10, upper=0.001),
        lr_scheduler = "linear", # ng.p.Choice(["linear", "cosine"]), linear tried extensively ~ 10 runs.
        diffusion_mode = "BoW_embedding_pred_x0_logits_direct_primary_double_logit_reg" # ng.p.Choice(["BoW_embedding_double_logit_reg", "BoW_embedding_double_logit_reg_direct_bow_x_to_logits"])

    )
    executor = submitit.AutoExecutor(folder='./submitit_logs/%j')
    executor.update_parameters(
        # set to time out after 2 days
        timeout_min=60 * 24 * 2,
        # set to short session
        slurm_partition="short", # slurm_partition="overcap", slurm_account="overcap", is also required with overcap
        # slurm_partition="overcap",
        # slurm_account="overcap",
        slurm_exclude="major,crushinator",
        slurm_constraint="a40", # can also do "a40|rtx_6000"
        gpus_per_node=1,
        cpus_per_task=7,
    )
    def partial_main(**args):
        wandb_run_name = f"hype_{args['diffusion_mode']}_lr={args['lr']}_embeddim={args['embed_dim']}_wd={args['weight_decay']}_g={args['BoW_cumsum_gamma']}_rconstw={args['reconst_weight']}_rconst2w={args['reconst_secondary_weight']}_wdemb={args['weight_decay_embed']}"
        model_save_path = "diffusion_hype_model_saves/" + wandb_run_name
        try:
            res = lib.ddp.wrap_main(main)(steps=10000, steps_max_for_lr_sched=92000, grad_accum_steps=4, batch_size=64, model_save_path=model_save_path, wandb_run_name=wandb_run_name, **args)[1]
            return res
        except Exception as e:
            print(e)
            return 10
    # python train.py --steps=9000 --steps_max_for_lr_sched=92000 --diffusion_mode=BoW_embedding_reconst_first --grad_accum_steps=2 --batch_size=128 --model_save_path=testhypr_lr=0.0014_embeddim=16_wd=4e-05_g=0.0_rconstw=0.7 --lr=0.0014 --embed_dim=16 --weight_decay=4e-05 --BoW_cumsum_gamma=0.0 --reconst_weight=0.7 --lr_scheduler=cosine
    optimizer = ng.optimizers.TwoPointsDE(parametrization=inst, budget=72, num_workers=6)
    recommendation = optimizer.minimize(partial_main, executor=executor, verbosity=2)
    print(recommendation.value)

