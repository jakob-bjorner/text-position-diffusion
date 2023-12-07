import contextlib
import fire
import mup
import numpy as np
import lib.datasets
import lib.models
import lib.utils
import os
import time
import torch
import torch.nn.functional as F
import tqdm
import sys
from torch import nn, optim, autograd

def main(**args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = lib.utils.AttributeDict(args)
    args.setdefault('seq_len', 256)
    args.setdefault('vocab_size', 32768)
    args.setdefault('weights_path', None)
    args.setdefault('dim', 2048)
    args.setdefault('n_blocks', 24)
    args.setdefault('n_heads', 32)
    args.setdefault('gamma_0', -3.)
    args.setdefault('gamma_1', 6.)
    args.setdefault('embed_dim', 16)
    args.setdefault('initial_noise_scale', 1.0)
    args.setdefault('n_samples', 8)
    args.setdefault('sampling_timesteps', 4096)
    args.setdefault('score_temp', 0.9)
    args.setdefault('output_scale', 1.)
    args.setdefault('owt2_tokenizer', True)
    args.setdefault('ddim_sampler', False)
    args.setdefault('guidance_weight', 2.)
    args.setdefault('diffusion_mode', 'token') # token or BoW_embedding or BoW_simplex
    args.setdefault('just_unconditional', False)
    args.setdefault('BoW_cumsum_gamma', None)
    args.setdefault('device', 'cuda')
    args.setdefault('use_secondary_logits', False)


    assert "token" in args.diffusion_mode or "BoW" in args.diffusion_mode, "diffusion_mode must be token or BoW"
    # assert args.diffusion_mode in ['token', 'BoW_embedding_post_bias_scale', 'BoW_embedding_pre_bias_scale', 'BoW_simplex'], "must be valid diffusion mode"
    if "BoW" in args.diffusion_mode:
        assert args.BoW_cumsum_gamma is not None, "must provide gamma for BoW diffusion mode"
    lib.utils.print_args(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_default_device(args.device)

    # Lots of annoying big/small numbers throughout this code, so we'll do
    # everything in fp64 by default and explicitly switch to fp32/bf16 where
    # appropriate.
    torch.set_default_dtype(torch.float64)

    def log1mexp(x):
        # Computes log(1-exp(-|x|))
        x = -x.abs()
        return torch.where(
            x > -0.693,
            torch.log(-torch.expm1(x)),
            torch.log1p(-torch.exp(x))
        )

    def create_modules(dim, n_heads):
        return {
            'noise_schedule': lib.models.NoiseSchedule().float(),
            'gamma_bounds': lib.models.GammaBounds(args.gamma_0, args.gamma_1).float(),
            'embedding_matrix': lib.models.EmbeddingMatrix(args.vocab_size, args.embed_dim).float(),
            'model': lib.models.DiffusionModel(dim, args.embed_dim, args.n_blocks, n_heads, args.vocab_size, diffusion_mode=args.diffusion_mode).float()
        }
    modules = create_modules(args.dim, args.n_heads)
    base_modules = create_modules(256, 4)
    delta_modules = create_modules(128, 2)
    for key in modules:
        main, base, delta = modules[key], base_modules[key], delta_modules[key]
        mup.set_base_shapes(main, base, delta=delta)
        main.to(args.device)

    print(f'Loading weights from {args.weights_path}')
    for name, module in modules.items():
        module.load_state_dict(torch.load(
            os.path.join(args.weights_path, f'{name}.pt'),
            map_location=torch.device(args.device)
        ))

    for key in modules:
        print(key+':')
        lib.utils.print_model(modules[key])


    def generate_samples(guidance_tokens, seq_len=args.seq_len, caputre_partials=None):
        """
        Sampling (implements Appendix A.4 eqn 33 in VDM). Needs float64 to work.
        guidance_tokens: [(token, weight, position, complement), ...]
            token: vocab index of token
            weight: guidance weight
            position: sequence index, or 'any', or 'all'
            complement: if True, do guidance on log(1-p(y|x))
        """
        with torch.no_grad():
            embedding_matrix = modules['embedding_matrix']()

            gamma_0, gamma_1 = modules['gamma_bounds']()
            alpha_0 = torch.sigmoid(-gamma_0).sqrt()
            sigma_0 = torch.sigmoid(gamma_0).sqrt()

            z = torch.randn((args.n_samples, seq_len, args.embed_dim), device=args.device) * args.initial_noise_scale
            x_selfcond = torch.zeros_like(z).float()
            for i, t in enumerate(tqdm.tqdm(torch.linspace(1., 0., args.sampling_timesteps))):
                t = t[None].to(args.device)
                s = t - 1. / args.sampling_timesteps
                gamma_s = modules['noise_schedule'](s).double()
                gamma_t = modules['noise_schedule'](t).double()
                gamma_s = gamma_0 + (gamma_1 - gamma_0) * gamma_s
                gamma_t = gamma_0 + (gamma_1 - gamma_0) * gamma_t
                alpha_squared_s = torch.sigmoid(-gamma_s)
                alpha_squared_t = torch.sigmoid(-gamma_t)
                alpha_s = alpha_squared_s.sqrt()
                alpha_t = alpha_squared_t.sqrt()
                sigma_squared_s = torch.sigmoid(gamma_s)
                sigma_squared_t = torch.sigmoid(gamma_t)
                sigma_s = sigma_squared_s.sqrt()
                sigma_t = sigma_squared_t.sqrt()

                if len(guidance_tokens) > 0:
                    with torch.enable_grad():
                        z.requires_grad = True
                        model_res = modules['model'](
                            z=z.to(torch.float32, copy=True),
                            gamma=gamma_t.float(),
                            embedding_matrix=embedding_matrix,
                            bias_scale=1.,
                            BoW_cumsum_gamma=args.BoW_cumsum_gamma,
                            x_selfcond=x_selfcond,
                            diffusion_mode=args.diffusion_mode
                        )
                        if isinstance(model_res, tuple):
                            logits, x_reconst = model_res
                        else:
                            if args.use_secondary_logits:
                                logits = model_res["logits_secondary"]
                            else:
                                logits = model_res["logits"]
                            x_reconst = model_res["x_reconst"]

                        if i % 100 == 0 and caputre_partials is not None:
                            caputre_partials.append(logits.clone().detach().cpu().argmax(dim=-1))

                        logprobs = F.log_softmax(logits.float(), dim=2)
                        logprobs_any = logprobs.logsumexp(dim=1)-float(seq_len)

                        sum_logp = 0.
                        for token, weight, position, complement in guidance_tokens:
                            if position == 'any':
                                logp = logprobs_any[:, token]
                            elif position == 'all':
                                logp = logprobs[:, :, token]
                            else:
                                logp = logprobs[:, position, token]
                            if complement:
                                logp = log1mexp(logp)
                            sum_logp += weight * logp.sum()

                        guidance_grad = autograd.grad(sum_logp, [z])[0]
                        z.requires_grad = False
                    x_selfcond = x_reconst.clone().detach()
                    x_reconst = x_reconst.double()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                    epsilon_pred /= args.score_temp
                    x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                    x_reconst += guidance_grad.double() * sigma_squared_t / alpha_squared_t.sqrt()
                    epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                else:
                    with torch.no_grad():
                        model_res = modules['model'](
                            z=z.to(torch.float32, copy=True),
                            gamma=gamma_t.float(),
                            embedding_matrix=embedding_matrix,
                            bias_scale=1.,
                            BoW_cumsum_gamma=args.BoW_cumsum_gamma,
                            x_selfcond=x_selfcond,
                            diffusion_mode=args.diffusion_mode
                        )
                        if isinstance(model_res, tuple):
                            _, x_reconst = model_res
                        else:
                            if args.use_secondary_logits:
                                _ = model_res["logits_secondary"]
                            else:
                                _ = model_res["logits"]
                            x_reconst = model_res["x_reconst"]
                        x_selfcond = x_reconst.clone().detach()
                        x_reconst = x_reconst.double()
                        epsilon_pred = (z - (alpha_t * x_reconst)) / sigma_t
                        epsilon_pred /= args.score_temp
                        x_reconst = (z - (sigma_t * epsilon_pred)) / alpha_t
                        if i % 100 == 0 and caputre_partials is not None:
                            caputre_partials.append(_.clone().detach().cpu().argmax(dim=-1))

                if t > 0:
                    if args.ddim_sampler:
                        z = (alpha_s * x_reconst) + (sigma_s * epsilon_pred)
                    else:
                        c = -torch.expm1(gamma_s - gamma_t)
                        z *= (1 - c) * alpha_squared_s.sqrt() / alpha_squared_t.sqrt()
                        z += c * (alpha_squared_s.sqrt() * x_reconst.double())
                        z += (c * (1 - alpha_squared_s)).sqrt() * torch.randn_like(z)
            with torch.no_grad():
                model_res = modules['model'](
                    z=z.float(),
                    gamma=gamma_t.float(),
                    embedding_matrix=embedding_matrix,
                    bias_scale=1.,
                    BoW_cumsum_gamma=args.BoW_cumsum_gamma,
                    x_selfcond=x_selfcond,
                    diffusion_mode=args.diffusion_mode
                )
                if isinstance(model_res, tuple):
                    logits = model_res[0]
                else:
                    if args.use_secondary_logits:
                        logits = model_res["logits_secondary"]
                    else:
                        logits = model_res["logits"]
                x_samples = logits.argmax(dim=-1)

            return x_samples

    def print_samples(x_samples):
        if args.owt2_tokenizer:
            owt2_tokenizer = lib.datasets.openwebtext2_tokenizer()
            for x in x_samples:
                x = owt2_tokenizer.decode(x.tolist(), skip_special_tokens=False)
                x = x.replace("\n", "↵")
                sys.stdout.buffer.write(x.encode('utf-8', 'ignore'))
        else:
            for x in x_samples:
                x = x.tolist()
                x = [idx2word[i].decode('utf-8', 'ignore') for i in x]
                x = ' '.join(x)
                x = x.replace('START','')
                x = x.replace('END','')
                x = x.replace('PAD','')
                x = x.replace(' .', '.')
                x = x.replace(' !', '!')
                x = x.replace(' ,', ',')
                x = x.replace(' \' ', '\'')
                x = x.strip()
                # replace newlines with '↵' symbol for cleaner printing
                print(x.replace("\n", "↵"))

    tokenizer = lib.datasets.openwebtext2_tokenizer()
    samples_dir = os.path.join(args.weights_path, f"samples-{time.time()}")
    os.mkdir(samples_dir)
    def write_and_print_samples(name, samples, partials=None):
        with open(os.path.join(samples_dir, name), 'w', encoding="utf-8") as f:
            for x in samples:
                x = tokenizer.decode(x.tolist(), skip_special_tokens=False)
                f.write(x.replace("\n", "↵") + '\n')
            if partials is not None:
                for j in range(partials[0].shape[0]):
                    f.write(f"partials for {j}:\n")
                    for i, partial in enumerate(partials):
                        for x in partial[[j]]:
                            x = tokenizer.decode(x.tolist(), skip_special_tokens=False)
                            f.write(x.replace("\n", "↵") + '\n')

        try:
            print_samples(samples)
            if partials is not None:
                for i, partial in enumerate(partials):
                    print(f"partial {i}:")
                    print_samples(partial)
            print("\n"*10)
        except:
            print('****Failed to print samples****')
    try:
        print('Unconditional:')
        partials = []
        samples = generate_samples([], seq_len=args.seq_len, caputre_partials=partials)
        write_and_print_samples('unconditional.txt', samples, partials)
    except Exception as e:
        print('****Failed to generate unconditional samples****', e)
        with open(os.path.join(samples_dir, 'error.txt'), 'w') as f:
            f.write('****Failed to generate unconditional samples****\n')
            f.write(str(e))
    if args.just_unconditional:
        return
    prefixes = [
        " The following is the statement of purpose for a student applying for a graduate fellowship. ```Natural Language Processing (NLP) has made great advances in recent years, but adopting it in real-world applications imposes challenges beyond current what is capable with modern architectures. These challenges call for"
    ]
    with open(os.path.join(samples_dir, 'prefixes.txt'), 'w') as f:
        f.write('\n'.join(prefixes))
    for i, prefix in enumerate(prefixes):
        print('Prefix completion: ', prefix)
        prefix = tokenizer.encode(prefix).ids
        samples = generate_samples(
            [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix)]
        )
        write_and_print_samples(f'prefix-completion-{i}.txt', samples)

    print('Infilling: A year ago in Paris, [...] Wow, what a great day!')
    tokenizer = lib.datasets.openwebtext2_tokenizer()
    prefix = tokenizer.encode(' A year ago in Paris,').ids
    suffix = tokenizer.encode('. Wow, what a great day!').ids
    infill_len = 40
    print_samples(generate_samples(
        [(token, args.guidance_weight, position, False) for position, token in enumerate(prefix)]
        + [(token, args.guidance_weight, position + len(prefix) + infill_len, False) for position, token in enumerate(suffix)]
    ))
    print("\n"*10)

    print('Word-level weights: Let\'s talk about law[10] and medicine[1].')
    guidance = [
        (tokenizer.encode(' Let').ids,      args.guidance_weight,   0,  False),
        (tokenizer.encode('\'s').ids,       args.guidance_weight,   1,  False),
        (tokenizer.encode(' talk').ids,     args.guidance_weight,   2,  False),
        (tokenizer.encode(' about').ids,    args.guidance_weight,   3,  False),
        (tokenizer.encode(' law').ids,      10.,                    4,  False),
        (tokenizer.encode(' and').ids,      args.guidance_weight,   5,  False),
        (tokenizer.encode(' medicine').ids, args.guidance_weight,   6,  False),
        (tokenizer.encode('.').ids,         args.guidance_weight,   7,  False),
    ]
    assert(all(len(a) == 1 for a,_,_,_ in guidance))
    guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    print_samples(generate_samples(guidance))
    print('\n'*10)

    print('Word-level weights: Let\'s talk about law[1] and medicine[10].')
    guidance = [
        (tokenizer.encode(' Let').ids,      args.guidance_weight,   0,  False),
        (tokenizer.encode('\'s').ids,       args.guidance_weight,   1,  False),
        (tokenizer.encode(' talk').ids,     args.guidance_weight,   2,  False),
        (tokenizer.encode(' about').ids,    args.guidance_weight,   3,  False),
        (tokenizer.encode(' law').ids,      args.guidance_weight,   4,  False),
        (tokenizer.encode(' and').ids,      args.guidance_weight,   5,  False),
        (tokenizer.encode(' medicine').ids, 10.,                    6,  False),
        (tokenizer.encode('.').ids,         args.guidance_weight,   7,  False),
    ]
    assert(all(len(a) == 1 for a,_,_,_ in guidance))
    guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    print_samples(generate_samples(guidance))
    print('\n'*10)

    print(f'Lexically constrained generation: Donald')
    guidance = [
        (tokenizer.encode(' Donald').ids, 3., 'any', False),
    ]
    assert(all(len(a) == 1 for a,_,_,_ in guidance))
    guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    print_samples(generate_samples(guidance))
    print("\n"*10)

    print(f'Negation: Donald but not Trump')
    guidance = [
        (tokenizer.encode(' Donald').ids, 3., 'any', False),
        (tokenizer.encode(' Trump').ids, 10., 'all', True),
    ]
    assert(all(len(a) == 1 for a,_,_,_ in guidance))
    guidance = [(a[0], b, c, d) for a,b,c,d in guidance]
    print_samples(generate_samples(guidance))
    print("\n"*10)


if __name__ == '__main__':
    fire.Fire(main)