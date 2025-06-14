import os
import sys
import yaml, itertools
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
from sklearn.metrics import mean_squared_error, precision_score, recall_score, confusion_matrix, f1_score
import h5py
import seaborn as sns
import datetime
import logging

# --------------------- 1. Config 및 환경 세팅 ---------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_logger(logfile=None):
    log_fmt = '[%(asctime)s][%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(logfile) if logfile else logging.NullHandler()
    ])

base_cfg = yaml.safe_load(open("base_config.yaml"))
param_grid = {
    'train.lr': [0.001, 0.002],
    'train.epochs': [300, 400]
}
combinations = list(itertools.product(*param_grid.values()))
for combo in combinations:
    cfg = base_cfg.copy()
    for k, v in zip(param_grid.keys(), combo):
        section, key = k.split('.')
        cfg[section][key] = v
    with open(f'sweep_cfg_{combo}.yaml', 'w') as f:
        yaml.dump(cfg, f)
    os.system(f'python main.py --config sweep_cfg_{combo}.yaml')
    
# --------------------- 2. 데이터 처리 ---------------------
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with h5py.File(save_path, 'w') as hf:
            for k, v in data.items():
                hf.create_dataset(k, data=v)
    return data

def load_data(h5_path):
    with h5py.File(h5_path, 'r') as hf:
        data = {k: hf[k][:] for k in hf.keys()}
    return data
    
def helmholtz_residual(coords, pred_field, region_mask, physics_param):
    # 예: Δu + k^2 u = 0
    grad = torch.autograd.grad(pred_field.sum(), coords, create_graph=True)[0]
    laplacian = torch.autograd.grad(grad.sum(), coords, create_graph=True)[0].sum(1)
    k = physics_param[:, 0]
    res = laplacian + (k**2) * pred_field
    return (res[region_mask==1]**2).mean()

def multi_loss(pred, gt, region_mask, coords, physics_param):
    loss_target = ...
    loss_helmholtz = helmholtz_residual(coords, pred[:,0], region_mask, physics_param)
    total_loss = loss_target + ... + 0.1*loss_helmholtz
    return total_loss
    
# Dropout 사용: 모델 정의시 Dropout 포함 (ex. nn.Dropout(0.1))
def predict_with_uncertainty(model, *args, mc=20):
    model.train()  # Dropout 활성화
    preds = []
    for _ in range(mc):
        preds.append(model(*args).detach().cpu().numpy())
    preds = np.stack(preds, axis=0)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)  # 표준편차=불확실도
    return pred_mean, pred_std

# 사용 예시
mean_pred, uncertainty = predict_with_uncertainty(model, coords, situation_feat, region_mask, physics_param)
plt.scatter(coords.cpu().numpy()[:,2], mean_pred[:,0], c=uncertainty[:,0], cmap='hot')
plt.colorbar(label='Uncertainty')

# --------------------- 3. Neural Operator 아키텍처 ---------------------
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

class Patchify(nn.Module):
    def __init__(self, in_dim, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_dim, embed_dim)
    def forward(self, x, coords):
        # 실제 3D patch id 생성
        patch_ids = (coords / self.patch_size).long()
        unique_patches, patch_indices = torch.unique(patch_ids, dim=0, return_inverse=True)
        # 각 patch별로 평균 또는 pooling (예시: mean)
        patch_feats = []
        for pid in range(unique_patches.shape[0]):
            patch_mask = (patch_indices == pid)
            patch_feats.append(self.proj(x[patch_mask]).mean(0, keepdim=True))
        patch_feats = torch.cat(patch_feats, dim=0)
        return patch_feats  # (num_patches, embed_dim)

class PhysicsConditionEncoder(nn.Module):
    """
    PDE 파라미터 등 physics 정보 임베딩
    """
    def __init__(self, param_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(param_dim, embed_dim)
    def forward(self, param):
        return self.fc(param)  # (N, embed_dim)

class PSDTransformerNOOperator(nn.Module):
    """
    Patchify + Physics Condition 임베딩 + Transformer Encoder + Output Projection
    """
    def __init__(self, in_dim, out_dim, patch_size=8, embed_dim=128, n_layers=4, n_heads=4, phys_param_dim=2):
        super().__init__()
        self.patchify = Patchify(in_dim, patch_size, embed_dim)
        self.physics_encoder = PhysicsConditionEncoder(phys_param_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, out_dim)
    def forward(self, x, coords, physics_param):
        # x: (N, in_dim)
        # coords: (N, 3)
        # physics_param: (N, phys_param_dim)
        x_patch = self.patchify(x, coords)  # (N, embed_dim)
        phys_embed = self.physics_encoder(physics_param)  # (N, embed_dim)
        patch_in = x_patch + phys_embed
        # (N, embed_dim) → (1, N, embed_dim) for transformer
        transformer_in = patch_in.unsqueeze(0)
        z = self.transformer(transformer_in)
        z = z.squeeze(0)
        out = self.output_proj(z)  # (N, out_dim)
        return out

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

physics_param = np.zeros((N, 2), dtype=np.float32)  # (예: wavenumber, damping 등)
physics_param[:, 0] = np.random.uniform(0.5, 3.0, N)  # 예시 wavenumber
physics_param[:, 1] = np.random.uniform(0.01, 0.2, N)  # 예시 damping
data = {..., 'physics_param': physics_param, ...}

class SoftRegionOperator(nn.Module):
    def __init__(self, in_dim, out_dim, n_regions=5, phys_param_dim=2):
        super().__init__()
        self.ops = nn.ModuleList([
            FFTOperator(in_dim, out_dim),
            FNOOperator(in_dim, out_dim),
            PSDTransformerNOOperator(in_dim, out_dim, phys_param_dim=phys_param_dim),  # region 2 등에서 할당
            RegionMLP(in_dim, out_dim),
            RegionMLP(in_dim, out_dim)
        ])
        self.n_regions = n_regions

    def forward(self, x, region_mask, coords, physics_param):
        outs = torch.zeros(x.shape[0], self.ops[0].kernel.shape[0], device=x.device)
        for i in range(self.n_regions):
            idxs = (region_mask == i)
            if torch.sum(idxs) > 0:
                if isinstance(self.ops[i], PSDTransformerNOOperator):
                    outs[idxs] = self.ops[i](x[idxs], coords[idxs], physics_param[idxs])
                else:
                    outs[idxs] = self.ops[i](x[idxs])
        return outs

class PARONetExpRegion(nn.Module):
    def __init__(self, in_dim=8, out_dim=2, n_regions=5):
        super().__init__()
        self.soft_op = SoftRegionOperator(in_dim, out_dim, n_regions)
        self.coord_encoder = nn.Linear(3, in_dim-5)

    def forward(self, coords, situation_feat, region_mask, physics_param):
        feat = torch.cat([self.coord_encoder(coords), situation_feat], dim=1)
        out = self.soft_op(feat, region_mask, coords, physics_param)
        return out

# --------------------- 4. Physics-Informed Loss ---------------------
def physics_loss(coords, pred_field, region_mask):
    # target region의 gradient penalty (물리장 smoothness)
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

# --------------------- 5. 실험/학습/평가 파이프라인 ---------------------
def run_experiment(cfg):
    set_seed(cfg['seed'])
    device = get_device()
    dt_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(cfg['logdir'], f'exp_{dt_now}')
    os.makedirs(exp_dir, exist_ok=True)
    setup_logger(os.path.join(exp_dir, 'exp.log'))
    writer = SummaryWriter(exp_dir)

    # 데이터 준비
    if cfg['data']['load_path'] and os.path.exists(cfg['data']['load_path']):
        data = load_data(cfg['data']['load_path'])
        logging.info(f"Loaded data from {cfg['data']['load_path']}")
    else:
        data = generate_data(cfg['data']['N'], cfg, cfg['data']['save_path'])
        logging.info(f"Generated new data, saved to {cfg['data']['save_path']}")
    coords = torch.tensor(data['coords'], dtype=torch.float32, device=device).requires_grad_()
    situation_feat = torch.tensor(data['situation_feat'], dtype=torch.float32, device=device)
    region_mask = torch.tensor(data['region_mask'], dtype=torch.long, device=device)
    gt = torch.tensor(data['gt'], dtype=torch.float32, device=device)
    X_train, X_test, S_train, S_test, M_train, M_test, G_train, G_test = train_test_split(
        coords, situation_feat, region_mask, gt, test_size=0.2, random_state=cfg['seed']
    )

    # 모델 선언
    model = PARONetExpRegion(in_dim=8, out_dim=2, n_regions=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    best_mse = float('inf')
    best_model_path = os.path.join(exp_dir, "best_model.pth")

    # 학습 루프
    for epoch in tqdm(range(cfg['train']['epochs']), desc='Training'):
        model.train()
        pred = model(X_train, S_train, M_train)
        loss = multi_loss(pred, G_train, M_train, X_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('loss/train', loss.item(), epoch)
        # 중간 validation
        if epoch % 20 == 0 or epoch == cfg['train']['epochs']-1:
            model.eval()
            with torch.no_grad():
                pred_val = model(X_test, S_test, M_test)
                mse_val = mean_squared_error(G_test[:,0].cpu().numpy(), pred_val[:,0].cpu().numpy())
                writer.add_scalar('mse/val', mse_val, epoch)
                if mse_val < best_mse:
                    torch.save(model.state_dict(), best_model_path)
                    best_mse = mse_val
            model.train()
        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch} - Train Loss: {loss.item():.5f} - Val MSE: {mse_val:.5f}")

    writer.close()
    logging.info(f"최적 모델을 {best_model_path}에 저장")

    # ----------------- 6. 결과 평가 및 시각화 -----------------
    # 최적 모델 불러오기
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(X_test, S_test, M_test)
        field_pred = pred[:,0].cpu().numpy()
        field_true = G_test[:,0].cpu().numpy()
        target_pred = (torch.sigmoid(pred[:,1]) > 0.5).long().cpu().numpy()
        target_true = (G_test[:,1] > 0.5).long().cpu().numpy()
        mse = mean_squared_error(field_true, field_pred)
        prec = precision_score(target_true, target_pred, zero_division=0)
        recall = recall_score(target_true, target_pred, zero_division=0)
        f1 = f1_score(target_true, target_pred, zero_division=0)
        confmat = confusion_matrix(target_true, target_pred)
        logging.info(f"[TEST] Field MSE: {mse:.4f}, Precision: {prec:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        logging.info(f"Confusion Matrix:\n{confmat}")

        # region별 MSE
        region_names = ['Background', 'Target', 'Risk', 'Jammer', 'Sensor']
        for i in range(5):
            if np.sum(M_test.cpu().numpy()==i) > 0:
                mse_i = mean_squared_error(
                    G_test[M_test==i,0].cpu().numpy(),
                    pred[M_test==i,0].cpu().numpy())
                logging.info(f"Region {i} ({region_names[i]}): MSE={mse_i:.5f}")

        # 결과 저장/시각화
        fig, ax = plt.subplots(1,2,figsize=(13,4))
        s1 = ax[0].scatter(X_test.cpu().numpy()[:,2], field_pred, c=M_test.cpu().numpy(), cmap='jet', s=8, alpha=0.7)
        plt.colorbar(s1, ax=ax[0], label="Region Mask")
        ax[0].set_title('Predicted Field (z axis)')
        ax[0].set_xlabel('z (penetration axis)')
        ax[0].set_ylabel('field')
        sns.histplot(torch.sigmoid(pred[M_test==1,1]).cpu().numpy(), bins=20, ax=ax[1], color='blue', label='Target pred', stat="density")
        sns.histplot(torch.sigmoid(pred[M_test!=1,1]).cpu().numpy(), bins=20, ax=ax[1], color='orange', label='Non-target pred', stat="density")
        ax[1].set_title('Target/Non-target Class (Sigmoid)')
        ax[1].set_xlabel('Predicted Target ID')
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'paronet_results.png'))
        plt.show()

        # 추가: 결과 csv 저장
        df = pd.DataFrame({
            'z': X_test.cpu().numpy()[:,2],
            'pred_field': field_pred,
            'true_field': field_true,
            'region_mask': M_test.cpu().numpy(),
            'pred_target_id': torch.sigmoid(pred[:,1]).cpu().numpy(),
            'true_target_id': G_test[:,1].cpu().numpy()
        })
        df.to_csv(os.path.join(exp_dir, 'paronet_test_results.csv'), index=False)
        logging.info(f"결과를 {exp_dir}에 저장 완료")

# --------------------- 7. Config YAML 예시 ---------------------
EXAMPLE_YAML = """
seed: 42
data:
  N: 3500
  load_path: null
  save_path: './simdata/exp_simdata.h5'
train:
  epochs: 400
  lr: 0.0015
logdir: './logs/defense_sim'
"""

# --------------------- 8. 명령행 인터페이스 및 실행 ---------------------
def parse_args():
    parser = argparse.ArgumentParser(description='PARONet Defense Simulation Full Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = yaml.safe_load(EXAMPLE_YAML)
    run_experiment(cfg)
