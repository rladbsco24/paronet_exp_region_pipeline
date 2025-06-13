import os
import yaml
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
import h5py
import seaborn as sns

# --------------------- Config 관리 ---------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

# --------------------- Seed / Device ---------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------- 데이터 로딩/생성/저장 ---------------------
def generate_data(N, cfg, save_path=None):
    x = np.random.uniform(-2, 2, N)
    y = np.random.uniform(-2, 2, N)
    z = np.random.uniform(0, 10, N)
    coords = np.stack([x, y, z], axis=1)
    threat_level = np.clip(np.random.randn(N)*0.3 + 0.5, 0, 1)
    sensor_status = np.clip(np.random.randn(N)*0.2 + 0.8, 0, 1)
    situation_flags = np.random.randint(0, 4, (N,))
    onehot_flags = np.eye(4)[situation_flags]
    situation_feat = np.concatenate([
        threat_level[:,None], sensor_status[:,None], onehot_flags
    ], axis=1)
    # region 정의
    target_center = np.array([0.5, 0.5, 8.5])
    target_radius = 0.8
    dists = np.linalg.norm(coords - target_center, axis=1)
    mask_geometry = dists < target_radius
    mask_threat = (threat_level > 0.7) | (situation_flags == 1)
    mask_jammer = (situation_flags == 2)
    mask_sensor = (sensor_status < 0.4) | (situation_flags == 3)
    region_mask = np.zeros(N, dtype=int)
    region_mask[mask_geometry] = 1
    region_mask[mask_threat & ~mask_geometry] = 2
    region_mask[mask_jammer & ~(mask_geometry | mask_threat)] = 3
    region_mask[mask_sensor & ~(mask_geometry | mask_threat | mask_jammer)] = 4
    # ground truth 생성
    gt = np.zeros((N, 2), dtype=np.float32)
    gt[region_mask==1,0] = np.exp(-((coords[region_mask==1,0]-0.5)**2 + (coords[region_mask==1,1]-0.5)**2 + (coords[region_mask==1,2]-8.5)**2)/0.12) + 0.08*np.random.randn(np.sum(region_mask==1))
    gt[region_mask==1,1] = 1.0
    gt[region_mask==2,0] = 0.6*np.sin(12*coords[region_mask==2,2]) + 0.3*np.random.randn(np.sum(region_mask==2))
    gt[region_mask==2,1] = 0.0
    gt[region_mask==3,0] = 0.5*np.cos(6*coords[region_mask==3,2]) + 0.2*np.random.randn(np.sum(region_mask==3))
    gt[region_mask==3,1] = 0.0
    gt[region_mask==4,0] = 0.3*np.random.rand(np.sum(region_mask==4))
    gt[region_mask==4,1] = 0.0
    gt[region_mask==0,0] = 0.08 * np.sin(coords[region_mask==0,2]) + 0.05*np.random.randn(np.sum(region_mask==0))
    gt[region_mask==0,1] = 0.0
    data = {
        'coords': coords, 'situation_feat': situation_feat, 'region_mask': region_mask, 'gt': gt
    }
    if save_path:
        with h5py.File(save_path, 'w') as hf:
            for k, v in data.items():
                hf.create_dataset(k, data=v)
    return data

def load_data(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = {k: hf[k][:] for k in hf.keys()}
    return data

# --------------------- Operator/Model ---------------------
class FFTOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(out_dim, in_dim, dtype=torch.cfloat) * 0.05)
    def forward(self, x):
        x_c = torch.complex(x, torch.zeros_like(x)) if not torch.is_complex(x) else x
        x_fft = torch.fft.fft(x_c, dim=1)
        x_fft_out = x_fft @ self.kernel.t()
        x_out = torch.fft.ifft(x_fft_out, dim=1)
        return torch.real(x_out)

class FNOOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 48), nn.GELU(),
            nn.Linear(48, 48), nn.GELU(),
            nn.Linear(48, out_dim)
        )
    def forward(self, x):
        return self.head(x)

class PSDTransformerNOOperator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        return self.trunk(x)

class RegionMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SoftRegionOperator(nn.Module):
    def __init__(self, in_dim, out_dim, n_regions=5):
        super().__init__()
        self.ops = nn.ModuleList([
            FFTOperator(in_dim, out_dim),
            FNOOperator(in_dim, out_dim),
            PSDTransformerNOOperator(in_dim, out_dim),
            RegionMLP(in_dim, out_dim),
            RegionMLP(in_dim, out_dim)
        ])
        self.n_regions = n_regions

    def forward(self, x, region_mask):
        outs = torch.zeros(x.shape[0], self.ops[0].kernel.shape[0], device=x.device)
        for i in range(self.n_regions):
            idxs = (region_mask == i)
            if torch.sum(idxs) > 0:
                outs[idxs] = self.ops[i](x[idxs])
        return outs

class PARONetExpRegion(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, n_regions=5):
        super().__init__()
        self.soft_op = SoftRegionOperator(in_dim, out_dim, n_regions)
        self.coord_encoder = nn.Linear(3, in_dim-5)

    def forward(self, coords, situation_feat, region_mask):
        feat = torch.cat([self.coord_encoder(coords), situation_feat], dim=1)
        out = self.soft_op(feat, region_mask)
        return out

# --------------------- Physics-Informed Loss (예시) ---------------------
def physics_loss(coords, pred_field, region_mask):
    # 간단 예시: target region의 gradient penalty (물리장 smoothness)
    grad = torch.autograd.grad(pred_field.sum(), coords, create_graph=True)[0]
    grad_norm = (grad[region_mask==1]**2).sum(1)
    return grad_norm.mean() if grad_norm.numel() > 0 else 0.0

def multi_loss(pred, gt, region_mask, coords):
    loss_target = F.mse_loss(pred[region_mask==1,0], gt[region_mask==1,0]) + F.binary_cross_entropy_with_logits(pred[region_mask==1,1], gt[region_mask==1,1])
    loss_risk = F.mse_loss(pred[region_mask==2,1], gt[region_mask==2,1])
    loss_jammer = F.mse_loss(pred[region_mask==3,1], gt[region_mask==3,1])
    loss_sensor = F.mse_loss(pred[region_mask==4,1], gt[region_mask==4,1])
    loss_bg = F.mse_loss(pred[region_mask==0,1], gt[region_mask==0,1])
    loss_phys = physics_loss(coords, pred[:,0], region_mask)
    total_loss = loss_target + 0.08*(loss_risk + loss_jammer + loss_sensor + loss_bg) + 0.05*loss_phys
    return total_loss

# --------------------- 학습/로깅/시각화 ---------------------
def run_experiment(cfg):
    set_seed(cfg['seed'])
    device = get_device()
    # 데이터 준비
    if cfg['data']['load_path'] and os.path.exists(cfg['data']['load_path']):
        data = load_data(cfg['data']['load_path'])
    else:
        data = generate_data(cfg['data']['N'], cfg, cfg['data']['save_path'])
    coords = torch.tensor(data['coords'], dtype=torch.float32, device=device).requires_grad_()
    situation_feat = torch.tensor(data['situation_feat'], dtype=torch.float32, device=device)
    region_mask = torch.tensor(data['region_mask'], dtype=torch.long, device=device)
    gt = torch.tensor(data['gt'], dtype=torch.float32, device=device)
    X_train, X_test, S_train, S_test, M_train, M_test, G_train, G_test = train_test_split(
        coords, situation_feat, region_mask, gt, test_size=0.2, random_state=cfg['seed']
    )
    # 모델
    model = PARONetExpRegion(in_dim=8, out_dim=2, n_regions=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    writer = SummaryWriter(cfg['logdir'])
    # 학습
    for epoch in tqdm(range(cfg['train']['epochs'])):
        model.train()
        pred = model(X_train, S_train, M_train)
        loss = multi_loss(pred, G_train, M_train, X_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), epoch)
        if epoch % 50 == 0:
            print(f"Epoch {epoch} - Train Loss: {loss.item():.5f}")
    # 테스트/분석
    model.eval()
    with torch.no_grad():
        pred = model(X_test, S_test, M_test)
        field_pred = pred[:,0].cpu().numpy()
        field_true = G_test[:,0].cpu().numpy()
        target_pred = (torch.sigmoid(pred[:,1]) > 0.5).long().cpu().numpy()
        target_true = (G_test[:,1] > 0.5).long().cpu().numpy()
        mse = mean_squared_error(field_true, field_pred)
        prec = precision_score(target_true, target_pred)
        recall = recall_score(target_true, target_pred)
        print(f"[TEST] Field MSE: {mse:.4f}, Precision: {prec:.3f}, Recall: {recall:.3f}")
        # 시각화
        fig, ax = plt.subplots(1,2,figsize=(12,4))
        ax[0].scatter(X_test.cpu().numpy()[:,2], field_pred, c=M_test.cpu().numpy(), cmap='jet', s=8, alpha=0.7)
        ax[0].set_title('Predicted Field (z axis)')
        sns.histplot(torch.sigmoid(pred[M_test==1,1]).cpu().numpy(), bins=20, ax=ax[1], color='blue', label='Target pred', stat="density")
        sns.histplot(torch.sigmoid(pred[M_test!=1,1]).cpu().numpy(), bins=20, ax=ax[1], color='orange', label='Non-target pred', stat="density")
        ax[1].set_title('Target/Non-target Class (Sigmoid)')
        ax[1].legend()
        plt.show()
    writer.close()

# --------------------- Config YAML 예시 ---------------------
EXAMPLE_YAML = """
seed: 42
data:
  N: 3000
  load_path: null
  save_path: './exp_simdata.h5'
train:
  epochs: 300
  lr: 0.002
logdir: './logs/exp_run'
"""

if __name__ == "__main__":
    # config 파일 없이 바로 실행 예시
    cfg = yaml.safe_load(EXAMPLE_YAML)
    run_experiment(cfg)
