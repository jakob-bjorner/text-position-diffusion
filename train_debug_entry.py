from train import main
import fire

if __name__ == "__main__":
    fire.Fire(main, command="--debug=True --grad_accum_steps=2 --batch_size=128 --model_save_path=debug_BoW_embed_reconst_g=0.9 --diffusion_mode=BoW_embedding_reconst_first --BoW_cumsum_gamma=0.9")


