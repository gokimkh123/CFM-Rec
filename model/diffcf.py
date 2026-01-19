import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils.enum_type import InputType
import math
import os

class DiffusionNet(nn.Module):
    def __init__(self, dims, time_emb_size, act_func="gelu"):
        super(DiffusionNet, self).__init__()
        self.dims = dims
        self.time_emb_dim = time_emb_size
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # 활성화 함수 선택 로직
        if act_func.lower() == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            activation = nn.GELU()

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(activation)
                layers.append(nn.Dropout(p=0.2)) 
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)
        time_emb = self.emb_layer(time_emb)
        
        x = torch.cat([x, time_emb], dim=-1)
        out = self.mlp(x)
        return out

def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiffCF(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DiffCF, self).__init__(config, dataset)

        # 1. 설정 로드
        self.n_steps = config["n_steps"]
        
        # [수정됨] Config 객체는 .get()을 지원하지 않으므로 in 연산자 사용
        if 'act_func' in config:
            self.act_func = config['act_func']
        else:
            self.act_func = 'gelu'

        # Diffusion Beta Schedule (Linear Schedule)
        # config에 없을 경우를 대비해 안전하게 로드
        self.beta_start = config['beta_start'] if 'beta_start' in config else 0.0001
        self.beta_end = config['beta_end'] if 'beta_end' in config else 0.02
        
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.n_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 2. Side Info 로드 (FlowCF와 동일)
        npy_path = os.path.join("dataset", "ML1M", "mv-tag-emb.npy")
        if not os.path.exists(npy_path):
            npy_path = "/app/dataset/ML1M/mv-tag-emb.npy"
        
        raw_emb = np.load(npy_path)
        self.raw_side_emb = torch.FloatTensor(raw_emb).to(self.device)
        self.side_dim = self.raw_side_emb.shape[1]

        n_movies = self.n_users 
        aligned_emb = torch.zeros((n_movies, self.side_dim)).to(self.device)
        for internal_id in range(1, n_movies):
            try:
                raw_token = dataset.id2token(dataset.uid_field, internal_id)
                raw_id = int(raw_token)
                if 0 <= raw_id < self.raw_side_emb.shape[0]:
                    aligned_emb[internal_id] = self.raw_side_emb[raw_id]
            except:
                pass
        self.side_emb = aligned_emb

        inter_matrix = dataset.inter_matrix(form='csr').astype('float32')
        self.history_matrix = torch.FloatTensor(inter_matrix.toarray()).to(self.device)
        
        # 3. 모델 네트워크 구축
        self.target_dim = self.n_items 
        self.input_dim = self.target_dim + self.side_dim 
        self.time_emb_size = config["time_embedding_size"]
        
        self.dims_mlp = [self.input_dim + self.time_emb_size] + config["dims_mlp"] + [self.target_dim]

        self.net = DiffusionNet(
            dims=self.dims_mlp,
            time_emb_size=self.time_emb_size,
            act_func=self.act_func
        ).to(self.device)

    def calculate_loss(self, interaction):
        movie_ids = interaction[self.USER_ID]
        x_0 = self.history_matrix[movie_ids] 
        cond = self.side_emb[movie_ids]     
        batch_size = x_0.size(0)

        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(x_0).to(self.device)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        net_input = torch.cat([x_t, cond], dim=1)
        predicted_noise = self.net(net_input, t)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_0_pred = self.p_sample_loop(interaction)
        scores = x_0_pred.gather(1, item.unsqueeze(1)).squeeze(1)
        return scores

    @torch.no_grad()
    def p_sample_loop(self, interaction):
        movie_ids = interaction[self.USER_ID]
        cond = self.side_emb[movie_ids]
        batch_size = movie_ids.size(0)

        x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            net_input = torch.cat([x_t, cond], dim=1)
            predicted_noise = self.net(net_input, t_tensor)

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            
            mean = coef1 * (x_t - coef2 * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = mean + sigma_t * noise
            else:
                x_t = mean 

        return x_t

    def predict_cold_item(self, item_id_or_emb, num_samples=10):
        if isinstance(item_id_or_emb, int):
            if item_id_or_emb < self.side_emb.shape[0]:
                 cond = self.side_emb[item_id_or_emb].unsqueeze(0)
            else:
                 cond = torch.zeros(1, self.side_dim).to(self.device)
        else:
            cond = torch.FloatTensor(item_id_or_emb).to(self.device)
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)

        cond_expanded = cond.repeat(num_samples, 1)
        batch_size = num_samples
        
        x_t = torch.randn(batch_size, self.target_dim).to(self.device)

        with torch.no_grad():
            for t in reversed(range(self.n_steps)):
                t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                net_input = torch.cat([x_t, cond_expanded], dim=1)
                predicted_noise = self.net(net_input, t_tensor)

                beta_t = self.betas[t]
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alphas_cumprod[t]
                
                coef1 = 1 / torch.sqrt(alpha_t)
                coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
                
                mean = coef1 * (x_t - coef2 * predicted_noise)
                
                if t > 0:
                    noise = torch.randn_like(x_t)
                    sigma_t = torch.sqrt(beta_t)
                    x_t = mean + sigma_t * noise
                else:
                    x_t = mean

        return x_t.mean(dim=0, keepdim=True)