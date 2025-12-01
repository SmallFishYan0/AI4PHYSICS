import torch
import torch.nn as nn
import numpy as np
import random
import os
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data import load_efficiency_data
from data import DetectorDataset
from model import DetectorEfficiencyTransformer
import argparse


def setup_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 保证 CUDA 算法的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader 的 worker 初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logger():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def plot_training_curves(train_losses, val_losses):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in dataloader:
        pos_mask = batch['pos_mask'].to(device)
        efficiency = batch['efficiency'].to(device)
        momentum = batch['momentum'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(pos_mask, efficiency, momentum)
        loss = criterion(outputs, labels.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * pos_mask.size(0)
        total_samples += pos_mask.size(0)
    
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_mae = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pos_mask = batch['pos_mask'].to(device)
            efficiency = batch['efficiency'].to(device)
            momentum = batch['momentum'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(pos_mask, efficiency, momentum)
            loss = criterion(outputs, labels.squeeze())
            mae = torch.abs(outputs - labels.squeeze()).sum()
            
            total_loss += loss.item() * pos_mask.size(0)
            total_mae += mae.item()
            total_samples += pos_mask.size(0)
    
    return total_loss / total_samples, total_mae / total_samples


def main(args):
    # 设置全局随机种子
    setup_seed(42)
    
    # 创建一个 Generator 用于 DataLoader
    g = torch.Generator()
    g.manual_seed(42)
    
    logger = setup_logger()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    

    # 加载训练数据：如果要测分布外效果，去除指定的分布外数据；如果使用inner/outer数据则添加
    train_efficiency_filepaths = [
        filepath for filepath in [
            os.path.join(args.train_data_dir, f) for f in os.listdir(args.train_data_dir) if f.endswith('.root')
        ]
        if os.path.basename(filepath) not in args.ood_data_names and (args.use_inner_outer or ('inner' not in filepath and 'outer' not in filepath))
    ]

    overall_data_dict = {
        os.path.basename(filepath): load_efficiency_data(filepath, args.onelayer_data_dir)
        for filepath in train_efficiency_filepaths
    }

    # 构造验证集：训练集每个文件抽取10%的数据组成验证集
    train_data_dict = {}
    val_data_dict = {}
    for filename, datas in overall_data_dict.items():
        train_datas, val_datas = train_test_split(datas, test_size=0.1, random_state=42)
        train_data_dict[filename] = train_datas
        val_data_dict[filename] = val_datas

    train_data_list = []
    for datas in train_data_dict.values():
        train_data_list.extend(datas)

    val_data_list = []
    for datas in val_data_dict.values():
        val_data_list.extend(datas)

    train_dataset = DetectorDataset(train_data_list)
    val_dataset = DetectorDataset(val_data_list)
    
    # 加载测试数据
    if args.ood_data_names:
        test_efficiency_filepaths = [
            os.path.join(args.train_data_dir, name) for name in args.ood_data_names
        ]
        test_data_dict = {
            os.path.basename(filepath): load_efficiency_data(filepath, args.onelayer_data_dir)
            for filepath in test_efficiency_filepaths
        }
        test_data_list = []
        for datas in test_data_dict.values():
            test_data_list.extend(datas)
        test_dataset = DetectorDataset(test_data_list)
    else:
        test_data_dict = val_data_dict
        test_dataset = val_dataset
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # 在 DataLoader 中加入 worker_init_fn 和 generator
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # 创建模型
    model = DetectorEfficiencyTransformer(
        d_model=64,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )
    
    # 训练循环
    num_epochs = args.epochs
    best_val_mae = float('inf')
    train_losses = []
    val_losses = []
    
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 每10个epoch记录一行日志
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:3d}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, Best MAE: {best_val_mae:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    logger.info(f"Training completed! Best Val MAE: {best_val_mae:.6f}")
    
    # 绘制训练曲线
    logger.info("Plotting training curves...")
    plot_training_curves(train_losses, val_losses)
    
    # 评估
    logger.info("="*60)
    logger.info("Loading best model for final evaluation on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    val_loss, val_mae = evaluate(model, val_loader, criterion, device)
    test_loss, test_mae = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"Final Val Loss: {val_loss:.6f}, Final Val MAE: {val_mae:.6f}")
    logger.info(f"Final Test Loss: {test_loss:.6f}, Final Test MAE: {test_mae:.6f}")

    logger.info("Different distribution results:")
    # 分布评估：对分布外测试，得到训练分布外每个分布的测试结果；对分布内测试，得到训练分布内每个分布的测试结果
    for name, data in test_data_dict.items():
        dataset = DetectorDataset(data)
        loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        loss, mae = evaluate(model, loader, criterion, device)
        if args.ood_data_names:
            logger.info(f"OOD Dataset: {name} - Loss: {loss:.6f}, MAE: {mae:.6f}")
        else:
            logger.info(f"Dataset: {name} - Loss: {loss:.6f}, MAE: {mae:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--train_data_dir', type=str, default="./raw_data", help='训练数据文件路径列表')
    parser.add_argument('--onelayer_data_dir', type=str, default="./raw_data/oneLayer_data", help='单层效率数据文件路径列表')
    parser.add_argument('--ood_data_names', nargs='*', default=[], help='分布外数据文件路径列表')
    parser.add_argument('--use_inner_outer', action='store_true', help='是否使用inner/outer layer数据训练')
    args = parser.parse_args()
    main(args)