import apex.normalization
import flash_attn.flash_attn_interface
import flash_attn.ops.fused_dense
import lib.utils
import mup
import numpy as np
import lib.rotary
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from einops import rearrange
from torch import nn, optim

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None,None,:]

def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, causal, residual_scale):
        super().__init__()

        self.causal = causal
        self.dim = dim
        self.n_heads = n_heads
        self.residual_scale = residual_scale

        self.rmsnorm1 = apex.normalization.FusedRMSNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3*dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.rmsnorm2 = apex.normalization.FusedRMSNorm(dim)
        self.mlp = flash_attn.ops.fused_dense.FusedMLP(
            dim, 4*dim, bias1=False, bias2=False, checkpoint_lvl=1)

    def forward(self, x, rotary_cos_sin, cu_seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Self-attention block
        x_skip = x
        x = self.rmsnorm1(x)
        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads
        )
        half_dtype = qkv.dtype
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = lib.rotary.apply_rotary_pos_emb(
                qkv, cos.to(half_dtype), sin.to(half_dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=self.causal)
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        x = residual_linear(
            x, self.attn_out.weight, x_skip, self.residual_scale
        )

        # Feedforward block
        x_skip = x
        x = self.rmsnorm2(x)
        x = self.mlp(x)
        x = torch.add(x_skip, x, alpha=self.residual_scale)

        return x

class EmbeddingMatrix(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.matrix.data /= self.matrix.data.norm(p=2, dim=1, keepdim=True)
    def forward(self, unnormalize=False):
        if unnormalize:
            return self.matrix
        norm = torch.linalg.norm(self.matrix, dim=1, keepdim=True)
        return (self.matrix / (norm + 1e-8))

class NoiseSchedule(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(1024, 1))
        self.b1 = nn.Parameter(torch.randn(1024))
        self.W2 = nn.Parameter(torch.randn(1, 1024))
    def forward(self, t):
        """t.shape: [n]"""
        W1 = F.softplus(self.W1.double())
        W2 = 0.01 * F.softplus(self.W2.double())
        def gamma_tilde(t):
            h = t[:,None] - 0.5
            h = (h @ W1.T) + self.b1[None,:].double()
            h = torch.tanh(h)
            h = (h @ W2.T)[:,0]
            return h
        gamma_tilde_0 = gamma_tilde(torch.tensor([0.], device='cuda'))
        gamma_tilde_1 = gamma_tilde(torch.tensor([1.], device='cuda'))
        gamma_tilde_t = gamma_tilde(t)
        return (
            (gamma_tilde_t - gamma_tilde_0) /
            (gamma_tilde_1 - gamma_tilde_0)
        )

class GammaBounds(nn.Module):
    def __init__(self, gamma_0, gamma_1):
        super().__init__()
        self.gamma_0 = nn.Parameter(torch.tensor(float(gamma_0)))
        self.gamma_1 = nn.Parameter(torch.tensor(float(gamma_1)))
    def forward(self):
        return self.gamma_0.clone().double(), self.gamma_1.clone().double()

class DiffusionModel(nn.Module):
    def __init__(self, dim, embed_dim, n_blocks, n_heads, vocab_size, diffusion_mode=""):
        super().__init__()

        self.input_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear = nn.Linear(embed_dim, dim, bias=False)
        self.selfcond_linear.weight.data.zero_()
        self.gamma_linear = nn.Linear(64, dim, bias=False)
        self.gamma_linear.weight.data.zero_()

        self.rotary_emb = lib.rotary.Rotary(dim // n_heads)

        residual_scale = float(1./np.sqrt(n_blocks))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, False, residual_scale)
            for i in range(n_blocks)
        ])

        if "reconst_first" in diffusion_mode:
            self.output_norm = lib.models.LayerNorm(dim)
            self.output_linear = mup.MuReadout(dim, embed_dim)
            self.output_linear.weight.data.zero_()
            self.output_linear.bias.data.zero_()
        elif "double_logit_reg" in diffusion_mode:
            self.output_norm_logits = lib.models.LayerNorm(dim)
            self.output_linear_logits = mup.MuReadout(dim, vocab_size)
            self.output_linear_logits.weight.data.zero_()
            self.output_linear_logits.bias.data.zero_()
            self.output_norm_embed = lib.models.LayerNorm(dim)
            self.output_linear_embed = mup.MuReadout(dim, embed_dim)
            self.output_linear_embed.weight.data.zero_()
            self.output_linear_embed.bias.data.zero_()
        else:
            self.output_norm = lib.models.LayerNorm(dim)
            self.output_linear = mup.MuReadout(dim, vocab_size)
            self.output_linear.weight.data.zero_()
            self.output_linear.bias.data.zero_()

        self.dim = dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, z, gamma, embedding_matrix, bias_scale, x_selfcond,
        diffusion_mode, BoW_cumsum_gamma=None, selfcond_mask=None, cu_seqlens=None, use_double_precision_for_conversion=False):
        if "BoW" in diffusion_mode:
            assert BoW_cumsum_gamma is not None
        if selfcond_mask is None:
            selfcond_mask = torch.ones(z.shape[0], device='cuda')

        alpha_squared = torch.sigmoid(-gamma)[:,None,None]
        sigma_squared = torch.sigmoid(gamma)[:,None,None]
        alpha = alpha_squared.sqrt()

        # change the z so that it subtract the noise then add it back after i have done the 
        # noise = None
        # z_cumsum = torch.einsum("bse,sn->bne", z-noise, torch.tril(BoW_cumsum_gamma ** ((torch.tril(torch.tile(torch.arange(seq_len).reshape(-1,1), (1, seq_len)) - torch.arange(seq_len).reshape(1,-1))))).T.float()).float()
        # z_cumsum = z_cumsum + noise
        # Rescale input to stdev 1
        z_variance = (alpha_squared / self.embed_dim) + sigma_squared
        x = z / z_variance.sqrt().float()

        x = self.input_linear(x)

        x = x + self.selfcond_linear(
            x_selfcond * float(np.sqrt(self.embed_dim))
        )

        gamma_embed = torch.linspace(-5., 5., 64 // 2, device='cuda')
        gamma_embed = gamma_embed.exp()[None,:] * gamma[:,None]
        gamma_embed = torch.cat([gamma_embed.sin(), gamma_embed.cos()], dim=1)
        gamma_embed = self.gamma_linear(gamma_embed.float())[:,None,:]
        x = x + gamma_embed

        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, cu_seqlens=cu_seqlens)
        # x [b, s, 384]
        if "double_logit_reg" in diffusion_mode:
            x_logits = self.output_norm_logits(x.float())
            x_embed = self.output_norm_embed(x.float())
            x_logits *= self.output_linear_logits.output_mult/self.output_linear_logits.width_mult()
            x_embed *= self.output_linear_embed.output_mult/self.output_linear_embed.width_mult()
            
            # logits_direct computation
            z_scaled_for_bias = bias_scale * (alpha/sigma_squared).float() * z

                
            W = torch.cat([
                self.output_linear_logits.weight.T,
                embedding_matrix.T,
                embedding_matrix.T.detach()
            ], dim=0)
            if "direct_bow_x_to_logits" in diffusion_mode:
                z_scaled_for_bias_logits = z_scaled_for_bias
            else:
                z_scaled_for_bias_logits = torch.cat([z_scaled_for_bias[:,[0],:], z_scaled_for_bias[:, 1:, :] - z_scaled_for_bias[:, :-1, :] * BoW_cumsum_gamma], dim=1)
                
            x_logits = torch.cat([
                x_logits,
                z_scaled_for_bias_logits * (1 - selfcond_mask.float()[:,None,None]),
                z_scaled_for_bias_logits * selfcond_mask.float()[:,None,None]
            ], dim=2)
            logits_direct = torch.addmm(
                self.output_linear_logits.bias.view(1, self.vocab_size),
                x_logits.view(-1, self.dim + 2*self.embed_dim),
                W.view(self.dim + 2*self.embed_dim, self.vocab_size)
            ).view(x_logits.shape[0], x_logits.shape[1], self.vocab_size)

            # x_reconst direct
            W = torch.cat([
                self.output_linear_embed.weight.T,
                torch.eye(self.embed_dim, dtype=torch.float) # this because we remove the noise from z as noise is computed by x.
            ], dim=0)
            x_embed = torch.cat([
                x_embed,
                torch.zeros_like(z_scaled_for_bias) if "pred_x0" in diffusion_mode else z_scaled_for_bias # z_scaled_for_bias
            ], dim=-1)
            x_reconst = torch.addmm(
                self.output_linear_embed.bias.view(1, self.embed_dim),
                x_embed.view(-1, self.dim + self.embed_dim),
                W.view(self.dim + self.embed_dim, self.embed_dim),
            ).view(x_embed.shape[0], x_embed.shape[1], self.embed_dim)
            x_reconst_no_cumsum = torch.cat([x_reconst[:,[0],:], x_reconst[:, 1:, :] - x_reconst[:, :-1, :] * BoW_cumsum_gamma], dim=1)
            # # Comment for 'no categorical reparameterization' ablation
            # x_reconst = F.softmax(logits, dim=2)
            # x_reconst = x_reconst @ torch.cat([
            #     embedding_matrix, embedding_matrix.detach()], dim=1)
            # x_reconst = torch.lerp(
            #     x_reconst[:,:,:self.embed_dim],
            #     x_reconst[:,:,self.embed_dim:],
            #     selfcond_mask.float()[:,None,None]
            # )
            logits_from_embed = x_reconst_no_cumsum @ torch.cat([embedding_matrix.T, embedding_matrix.T.detach()], dim=1)
            logits_from_embed = torch.lerp(
                logits_from_embed[:,:,:self.vocab_size],
                logits_from_embed[:,:,self.vocab_size:],
                selfcond_mask.float()[:,None,None]
            ).view(x.shape[0], x.shape[1], self.vocab_size)
            if "logits_direct_primary" in diffusion_mode:
                return {"logits": logits_direct, "x_reconst": x_reconst, "logits_secondary": logits_from_embed}
            else:
                return {"logits": logits_from_embed, "x_reconst": x_reconst, "logits_secondary": logits_direct}
        # going to run with pred_x0, logits_direct_primary, double_logit_reg x_reconst2 loss scaled to 0
        x = self.output_norm(x.float())
        x *= self.output_linear.output_mult/self.output_linear.width_mult()
        

        if "reconst_first" in diffusion_mode:
            # create Bow as a prediction off of the x and z directly, then predict the logits afterwards with difference equation.
            # change the prediction space from that of direct to words to that of direct to BoW.
            # compute using x (something which was spit out by the transformer), z (the noisey version of embeddings), and any new parameters necessary.
            z_scaled_for_bias = bias_scale * (alpha/sigma_squared).float() * z
            
            W = torch.cat([
                self.output_linear.weight.T,
                torch.eye(self.embed_dim, dtype=torch.float) # this because we remove the noise from z as noise is computed by x.
            ], dim=0)
            x = torch.cat([
                x,
                torch.zeros_like(z_scaled_for_bias) if "pred_x0" in diffusion_mode else z_scaled_for_bias
            ], dim=-1)
            x_reconst = torch.addmm(
                self.output_linear.bias.view(1, self.embed_dim),
                x.view(-1, self.dim + self.embed_dim),
                W.view(self.dim + self.embed_dim, self.embed_dim),
            ).view(x.shape[0], x.shape[1], self.embed_dim)

            x_reconst_no_cumsum = torch.cat([x_reconst[:,[0],:], x_reconst[:, 1:, :] - x_reconst[:, :-1, :] * BoW_cumsum_gamma], dim=1)
            if "diff_lm_norms" in diffusion_mode:
                # 9.4 NLL
                x_reconst_l2 = (x_reconst_no_cumsum ** 2).sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]
                embed_l2 = (embedding_matrix ** 2).sum(dim=-1).view(1, 1, -1)  # [1, 1, vocab_size]
                pre_logits = x_reconst_no_cumsum @ torch.cat([embedding_matrix.T, embedding_matrix.T.detach()], dim=1)
                pre_logits = torch.lerp(
                    pre_logits[:,:,:self.vocab_size],
                    pre_logits[:,:,self.vocab_size:],
                    selfcond_mask.float()[:,None,None]
                ).view(x.shape[0], x.shape[1], self.vocab_size)
                # pre_logits = x_reconst_no_cumsum @ embedding_matrix.T
                dist = x_reconst_l2 + embed_l2 - 2 * pre_logits  # [batch, seq_len, vocab_size]
                logits = - torch.sqrt(torch.clamp(dist, 0.0, np.inf))
            else:
                # 7.11 NLL
                logits = x_reconst_no_cumsum @ torch.cat([embedding_matrix.T, embedding_matrix.T.detach()], dim=1)
                logits = torch.lerp(
                    logits[:,:,:self.vocab_size],
                    logits[:,:,self.vocab_size:],
                    selfcond_mask.float()[:,None,None]
                ).view(x.shape[0], x.shape[1], self.vocab_size)
                # logits = x_reconst_no_cumsum @ embedding_matrix.T

            return logits, x_reconst

        W = torch.cat([
            self.output_linear.weight.T,
            embedding_matrix.T,
            embedding_matrix.T.detach()
        ], dim=0)
        # if 'pre_bias_scale' in diffusion_mode:
        #     z = torch.cat([z[:,[0],:], z[:, 1:, :] - z[:, :-1, :] * BoW_cumsum_gamma], dim=1)
        # if "post_bias_scale" in diffusion_mode:
        #     # x = torch.cat([x[:,[0],:], x[:, 1:, :] - x[:, :-1, :]], dim=1)
        #     z_scaled_for_bias = torch.cat([z_scaled_for_bias[:,[0],:], z_scaled_for_bias[:, 1:, :] - z_scaled_for_bias[:, :-1, :] * BoW_cumsum_gamma], dim=1)
        z_scaled_for_bias = bias_scale * (alpha/sigma_squared).float() * z
        x = torch.cat([
            x,
            z_scaled_for_bias * (1 - selfcond_mask.float()[:,None,None]),
            z_scaled_for_bias * selfcond_mask.float()[:,None,None]
        ], dim=2)
        logits = torch.addmm(
            self.output_linear.bias.view(1, self.vocab_size),
            x.view(-1, self.dim + 2*self.embed_dim),
            W.view(self.dim + 2*self.embed_dim, self.vocab_size)
        ).view(x.shape[0], x.shape[1], self.vocab_size)

        # Comment for 'no categorical reparameterization' ablation
        x_reconst = F.softmax(logits, dim=2)
        x_reconst = x_reconst @ torch.cat([
            embedding_matrix, embedding_matrix.detach()], dim=1)
        x_reconst = torch.lerp(
            x_reconst[:,:,:self.embed_dim],
            x_reconst[:,:,self.embed_dim:],
            selfcond_mask.float()[:,None,None]
        )
        if "BoW_embedding" in diffusion_mode:
            seq_len = x_reconst.shape[-2]
            if use_double_precision_for_conversion:
                x_reconst_bow = torch.einsum("bse,sn->bne", x_reconst, torch.tril(BoW_cumsum_gamma ** ((torch.tril(torch.tile(torch.arange(seq_len).reshape(-1,1), (1, seq_len)) - torch.arange(seq_len).reshape(1,-1))))).T.float()).float()
            else:
                x_reconst_bow = torch.einsum("bse,sn->bne", x_reconst.double(), torch.tril(BoW_cumsum_gamma ** ((torch.tril(torch.tile(torch.arange(seq_len).reshape(-1,1), (1, seq_len)) - torch.arange(seq_len).reshape(1,-1))))).T.double()).float()
            # recomputed_x_reconst = x_reconst_bow[:,1:,:] - x_reconst_bow[:,:-1,:] * BoW_cumsum_gamma
            # assert torch.max(torch.abs(recomputed_x_reconst - x_reconst[:,1:,:])) < 0.1, f"cumsum along sequence dimension is not correct {recomputed_x_reconst}, {x_reconst[:,1:,:]}"
            x_reconst = x_reconst_bow

        return logits, x_reconst

class UnetModule(nn.Module):
    '''Incomplete Unet module. Requires significant thought to come up with good equivelant to cnn unet'''
    def __init__(self, in_dim, out_dim, down_sample_factor, internal_func):
        super().__init__()
        # input norm
        self.input_norm = nn.LayerNorm(in_dim)

        self.linear1 = nn.Linear(in_dim, in_dim)
        self.non_linear1 = nn.GELU() # non_linearity? (yes because conv right after would be linear otherwise?)

        self.down_sample = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=down_sample_factor, stride=down_sample_factor, padding=0)
        self.internal_func = internal_func
        self.up_sample = nn.ConvTranspose1d(in_channels=out_dim, out_channels=in_dim, kernel_size=down_sample_factor, stride=down_sample_factor, padding=0)

        # should this be normed?
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.non_linear2 = nn.GELU() # else going into the next upconv you will be linear with the convolution. (being linear not necessarily bad imo becasue people do low rank approximations and then use these reps for down stream...?)
    def forward(self, x):
        x = self.input_norm(x)
        x = self.non_linear1(self.linear1(x))
        x_skip = x
        x = self.up_sample(self.internal_func(self.down_sample(x)))
        x = x_skip + x
        x = self.non_linear2(self.linear2(x))
        return x



class DiffusionModelUnetText(nn.Module):
    '''Incomplete. Further consideration needed for text unet.'''
    def __init__(self, dim, embed_dim, n_blocks, n_heads, vocab_size, diffusion_mode=""):
        super().__init__()
        """Plan for U-Net
        input: [B, S, 16]
        # input_norm Norm depending on the BoW Gamma used, so r -> 1 / (1 - r), so mult by (1 - r) on the cumsum input... to get it normalized. the noise will be impacted to that scale, so do I normalize before?
        input_linear: 16 -> 384
        # transformer on seq len, then ...? down sampling only works if enough sequences, so do I upscale? how to ensure enough seqs? add pad toks between? or just to the end, need to do to max down sampling ratio ie if some area with 16 tokens of the 1024 available, then do 2**6 tokens of pad possible, ie 64, not too much actually.
        16 4x down sampling layers? sounds reasonable... really would want to arch search over large space of hyper params. No idea about causal masking?
        output: [B, S, 384]
        Q's: residual scale wtf?
        """
        self.blocks = nn.ModuleList([])
    def forward(self, z, embedding_matrix):
        pass

class AutoregressiveModel(nn.Module):
    def __init__(self, dim, n_blocks, n_heads, vocab_size, tie_embeddings):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        if not tie_embeddings:
            self.input_embedding = nn.Embedding(vocab_size, dim)
        self.rotary_emb = lib.rotary.Rotary(dim // n_heads)

        residual_scale = float(1./np.sqrt(n_blocks))
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, True, residual_scale)
            for i in range(n_blocks)
        ])
        self.output_norm = apex.normalization.FusedRMSNorm(dim)
        self.output_linear = mup.MuReadout(dim, vocab_size)
        self.first_token_logits = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        if self.tie_embeddings:
            x = F.embedding(x, self.output_linear.weight) * float(np.sqrt(3*256))
        else:
            x = self.input_embedding(x)
        rotary_cos_sin = self.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin)
        x = x.float()
        x = self.output_norm(x)
        logits = self.output_linear(x)
        logits = torch.cat([
            self.first_token_logits[None,None,:].expand(x.shape[0],-1,-1),
            logits[:,:-1,:]
        ], dim=1)
        return logits