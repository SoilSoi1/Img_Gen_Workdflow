import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVAE(nn.Module):
    """Lightweight VAE: 256x256 -> 4x32x32 -> 256x256."""

    def __init__(self, in_channels=3, latent_channels=4, hidden_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Encoder: 256 -> 128 -> 64 -> 32
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1)  # 128

        self.enc2 = nn.Sequential(
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1)  # 64

        self.enc3 = nn.Sequential(
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
        )
        self.down3 = nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, stride=2, padding=1)  # 32

        # Latent space
        self.to_mu = nn.Conv2d(hidden_channels * 4, latent_channels, 3, padding=1)
        self.to_logvar = nn.Conv2d(hidden_channels * 4, latent_channels, 3, padding=1)

        # Decoder: 32 -> 64 -> 128 -> 256
        self.up0 = nn.Conv2d(latent_channels, hidden_channels * 4, 3, padding=1)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
        )  # 64

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
        )  # 128

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )  # 256

        self.to_rgb = nn.Conv2d(hidden_channels, in_channels, 3, padding=1)

    def encode(self, x):
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)
        x = self.down3(x)
        mu = self.to_mu(x)
        logvar = self.to_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.up0(z)
        z = self.up1(z)
        z = self.up2(z)
        z = self.up3(z)
        z = self.to_rgb(z)
        return torch.tanh(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ResBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )

        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual(x)


class SimpleUNet(nn.Module):
    """UNet for latent-space diffusion with proper time conditioning."""

    def __init__(self, in_channels=4, out_channels=4, model_channels=192, time_emb_dim=768):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding: sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        self.down1 = nn.ModuleList([
            ResBlock(model_channels, model_channels, time_emb_dim),
            ResBlock(model_channels, model_channels, time_emb_dim),
        ])
        self.down1_pool = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)

        self.down2 = nn.ModuleList([
            ResBlock(model_channels, model_channels * 2, time_emb_dim),
            ResBlock(model_channels * 2, model_channels * 2, time_emb_dim),
        ])
        self.down2_pool = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock(model_channels * 2, model_channels * 2, time_emb_dim),
            ResBlock(model_channels * 2, model_channels * 2, time_emb_dim),
        ])

        # Decoder
        self.up2 = nn.ModuleList([
            ResBlock(model_channels * 4, model_channels * 2, time_emb_dim),
            ResBlock(model_channels * 4, model_channels * 2, time_emb_dim),
        ])
        self.up2_conv = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)

        self.up1 = nn.ModuleList([
            ResBlock(model_channels * 2, model_channels, time_emb_dim),
            ResBlock(model_channels * 2, model_channels, time_emb_dim),
        ])
        self.up1_conv = nn.ConvTranspose2d(model_channels * 2, model_channels, 4, stride=2, padding=1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        # Time embedding: t is [B] long tensor of timestep indices
        t_emb = self._get_timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # Encoder
        h = self.conv_in(x)

        h1 = []
        for block in self.down1:
            h = block(h, t_emb)
            h1.append(h)
        h = self.down1_pool(h)

        h2 = []
        for block in self.down2:
            h = block(h, t_emb)
            h2.append(h)
        h = self.down2_pool(h)

        # Bottleneck
        for block in self.mid:
            h = block(h, t_emb)

        # Decoder
        h = self.up2_conv(h)
        for block in self.up2:
            h = torch.cat([h, h2.pop()], dim=1)
            h = block(h, t_emb)

        h = self.up1_conv(h)
        for block in self.up1:
            h = torch.cat([h, h1.pop()], dim=1)
            h = block(h, t_emb)

        h = self.conv_out(h)
        return h

    def _get_timestep_embedding(self, t, embedding_dim):
        """Sinusoidal timestep embeddings."""
        device = t.device
        half_dim = embedding_dim // 2
        emb = torch.arange(half_dim, device=device, dtype=torch.float32)
        emb = math.exp(-math.log(10000.0) / (half_dim - 1)) * emb
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class SimplifiedLDMWrapper(nn.Module):
    """Full LDM wrapper with pretrained SD VAE + UNet for latent-space diffusion."""

    def __init__(self, model_config, device, image_size=512, vae_path='/root/autodl-tmp/Img_Gen_Workdflow/weights/sd-vae-ft-mse'):
        super().__init__()
        self.device = device
        self.model_config = model_config
        self.parameterization = "eps"
        self.image_size = image_size
        self.channels = 3

        # Diffusion parameters
        params = model_config.params if hasattr(model_config, 'params') else model_config
        self.linear_start = params.linear_start if hasattr(params, 'linear_start') else params.get('linear_start', 0.0015)
        self.linear_end = params.linear_end if hasattr(params, 'linear_end') else params.get('linear_end', 0.0195)
        self.num_timesteps = params.timesteps if hasattr(params, 'timesteps') else params.get('timesteps', 1000)

        # Load pretrained SD VAE
        from diffusers import AutoencoderKL
        self.first_stage_model = AutoencoderKL.from_pretrained(vae_path)
        self.first_stage_model.to(device)
        self.first_stage_model.requires_grad_(False)
        self.first_stage_model.eval()
        self.scale_factor = self.first_stage_model.config.scaling_factor

        # UNet operates in latent space
        self.unet = SimpleUNet(in_channels=4, out_channels=4, model_channels=192, time_emb_dim=768)

        self._setup_noise_schedule()
        self.to(device)

    def _setup_noise_schedule(self):
        betas = torch.linspace(self.linear_start, self.linear_end, self.num_timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod, (1, 0), value=1.0)[:-1]

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    @torch.no_grad()
    def encode_first_stage(self, x):
        z = self.first_stage_model.encode(x).latent_dist.sample()
        z = z * self.scale_factor
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = z / self.scale_factor
        return self.first_stage_model.decode(z).sample

    def forward(self, x):
        """VAE encode -> latent diffusion -> loss. VAE is frozen."""
        with torch.no_grad():
            z = self.encode_first_stage(x)

        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(z)

        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])

        shape = [x.shape[0], 1, 1, 1]
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(*shape)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(*shape)

        z_t = sqrt_alphas_cumprod * z + sqrt_one_minus_alphas_cumprod * noise
        pred_noise = self.unet(z_t, t)

        return nn.functional.mse_loss(pred_noise, noise, reduction='mean')

    def apply_model(self, x, t, c=None):
        return self.unet(x, t)

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        shape = [x0.shape[0], 1, 1, 1]
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(*shape)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(*shape)
        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
