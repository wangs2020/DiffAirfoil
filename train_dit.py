import argparse
import datetime
import torch
import wandb

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from dataload import AirFoilMixParsec
import numpy as np
from models import script_utils,VAE
from utils import Fit_airfoil,vis_airfoil2,de_norm
import os

def get_datasets():
    """获得训练、验证 数据集"""
    train_dataset = AirFoilMixParsec(split='train')
    val_dataset = AirFoilMixParsec(split='val')
    return train_dataset, val_dataset


def evaluate_one_epoch(model, dataloader, device, epoch, args):
    """验证一个epoch"""
    model.eval()
    
    correct_pred = 0  # 预测正确的样本数量
    total_pred = 0  # 总共的样本数量
    total_loss = 0.0
    test_num = 1000

    test_loader = tqdm(dataloader)
    for i,data in enumerate(test_loader):

        if i >= test_num: #为了测试更快，只计算前test_num个
            break

        x = data['gt'][0,:,0] # (257,)
        gt = data['gt'][:,:,1:2] # (128, 257, 2)
        gt = de_norm(gt)
        y = data['params'] # (128, 11)
        y2 = data['keypoint'][:,:,1] # (128, 26)
        #gt = gt.to(device)
        y = y.to(device) 
        y2 = y2.to(device)

        samples = model.sample_ddim(batch_size=1, device=device, y=y, y2=y2,clip_denoised=args.clip_denoised).to(device)
        samples = samples[0,:,0].cpu().numpy()
        samples = np.stack([x,samples],axis=1)
        samples = de_norm(samples)
        #samples = de_norm(samples.reshape(1,257,2))
        #samples = samples.permute(0, 2, 1).cpu().numpy()
        
        # 对samples 进行可视化
        if i % 10 == 0: #每隔十次可视化一次
            source = de_norm(data['gt'][0].cpu().numpy())
            vis_airfoil2(source,samples,f'{epoch}_{i}',dir_name=args.log_dir,sample_type='ddim')

        total_pred += gt.shape[0]

        gt = gt[0].cpu()
        samples = torch.tensor(samples)

        loss = F.mse_loss(samples.view(-1, 257*2), gt.view(-1, 257*2), reduction='sum')
        total_loss += loss.item()

        # 计算点到直线的距离
        distances = torch.norm(gt - samples, dim=1) #(B,257)
        # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
        t = args.distance_threshold
        # 200个点中，预测正确的点的比例超过ratio，认为该形状预测正确
        ratio = args.threshold_ratio
        count = (distances < t).sum() #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
        correct_count = (count >= ratio*200).sum().item() # batch_size数量的样本中，正确预测样本的个数
        correct_pred += correct_count
        
    accuracy = correct_pred / total_pred
    avg_loss = total_loss / total_pred
    
    print(f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")


# airfoil generatoin condition on keypoint + parsec 

def main():
    args = create_argparser().parse_args()
    device = args.device
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        os.makedirs(args.log_dir, exist_ok=True)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='treaptofun',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size
        train_dataset, test_dataset = get_datasets()

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=2,
        ))
        test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=1, shuffle=False)
        
        for iteration in tqdm(range(args.start_iter, args.iterations + 1)):
            diffusion.train()

            data = next(train_loader)  
           
            gt = data['gt'][:,:,1:2] # (128, 257, 2)
            y = data['params'] # (128, 11)
            y2 = data['keypoint'][:,:,1] # (128, 26)
            gt = gt.to(device)  
            y = y.to(device) 
            y2 = y2.to(device)
 
            loss = diffusion(gt, y, y2)
            acc_train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            print(f"iteration: {iteration}, train_loss: {acc_train_loss}")

            #args.log_rate = 1
            if iteration % args.log_rate == 0:
                evaluate_one_epoch(diffusion, test_loader, device, iteration, args)
                  
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/model-airfoil-{iteration}.pth"
                optim_filename = f"{args.log_dir}/optim-airfoil-{iteration}.pth"
                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
            
        if args.log_to_wandb:
            run.finish()

    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-5,
        batch_size=256,
        iterations=300000,
        log_to_wandb=False,
        log_rate=10000,
        checkpoint_rate=10000,
        #log_dir="weights/dit_noclip_denoised",
        log_dir="weights/dit_clip_denoised",
        project_name='airfoil-dit',
        run_name=run_name,
        start_iter=1,
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule_low=1e-4,
        schedule_high=0.02,
        device=device,
        clip_denoised=False,
        # 评测指标相关
        distance_threshold=0.01, # xy点的距离小于该值，被认为是预测正确的点
        threshold_ratio=0.75, # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()