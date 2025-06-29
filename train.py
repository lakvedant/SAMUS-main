from ast import arg
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from tqdm import tqdm  # Added tqdm import
import wandb  # Added wandb import
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import  get_click_prompt
from utils.gradcam import SAMUSGradCAM

def initialize_gradcam(model, opt):
    """Initialize GradCAM if visualization is enabled."""
        # Try different layer combinations for SAMUS
    possible_layers = [
            ['image_encoder.neck.3']
    ]
        
    model_layers = [name for name, _ in model.named_modules()]
        
    for target_layers in possible_layers:
        if all(layer in model_layers for layer in target_layers):
            print(f"Using GradCAM layers: {target_layers}")
            return SAMUSGradCAM(model, target_layers)
        
    return None


def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS')
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS')
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='/home/hoprus/iitm_interns_ws/lakshit/SAMUS-FT-main/SAMUS-FT/sam_vit_b_01ec64 .pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train (overrides config file)')
    parser.add_argument('--early_stopping', type=int, default=50, help='early stopping patience (epochs)')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')  # Added wandb argument

    args = parser.parse_args()
    opt = get_config(args.task) 
    wandb.login(key='4ac28743425731f3f01c3d7a2013e64ff47949cf')
    
    # Override epochs from command line if provided
    if args.epochs is not None:
        opt.epochs = args.epochs 

    # Initialize wandb with enhanced logging
    if args.use_wandb:
        run = wandb.init(project="SAMUS-Medical-Segmentation", name=f"{args.modelname}_{args.task}_{time.strftime('%m%d_%H%M')}", 
                   config=vars(args))
        print(f"\n🔗 WandB Project URL: {run.get_project_url()}")
        print(f"🔗 WandB Run URL: {run.get_url()}\n")

    device = torch.device(opt.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    tf_train = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
                                p_contr=0.5, p_gama=0.5, p_distor=0.0, color_jitter_params=None, long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    train_dataset = ImageToImage2D(dataset_path=opt.data_path, split=opt.train_split, joint_transform=tf_train,model=model, modelname=args.modelname, img_size=args.encoder_input_size,opt=opt)
    val_dataset = ImageToImage2D(dataset_path=opt.data_path, split=opt.val_split, joint_transform=tf_val,model=model,modelname=args.modelname, img_size=args.encoder_input_size,opt=opt)  # return image, mask, and filename
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    opt.pre_trained=True
    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k,v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
      
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(opt.epochs+1), np.zeros(opt.epochs+1)
    
    # Early stopping and model saving setup
    patience_counter = 0
    timestr = time.strftime('%m%d%H%M')
    
    # Create checkpoints folder
    checkpoint_dir = os.path.join(opt.save_path, 'checkpoints/PNSAMUS')
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Initialize variables for best model path tracking
    best_model_path = None
    val_losses = 0.0
    mean_dice = 0.0
    
    for epoch in range(opt.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        model.train()
        train_losses = 0
        gradcam_obj = initialize_gradcam(model, opt)
        # Create a new progress bar for each epoch
        epoch_pbar = tqdm(total=len(trainloader), 
                         desc=f"Epoch {epoch+1}/{opt.epochs}", 
                         unit="batch", 
                         dynamic_ncols=True,
                         leave=True)  # leave=True keeps the completed progress bars visible
        count=0

        for batch_idx, datapack in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype = torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)
            count+=1
            if count==1:
                print(pt)
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = model(imgs, pt)
            train_loss = criterion(pred, masks) 
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            
            # Update epoch progress bar with comprehensive information
            epoch_pbar.set_postfix({
                'Loss': f'{train_loss.item():.4f}',
                'AvgLoss': f'{train_losses/(batch_idx+1):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Best': f'{best_dice:.4f}' if best_dice > 0 else 'N/A',
                'Pat': f'{patience_counter}/{args.early_stopping}'
            }, refresh=True)
            
            # Update progress bar by 1 batch
            epoch_pbar.update(1)
            
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        # Close the current epoch's progress bar
        epoch_pbar.close()

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        avg_train_loss = train_losses / len(trainloader)
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', avg_train_loss, epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = avg_train_loss

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % opt.eval_freq == 0:
            model.eval()
            
            # Create a separate progress bar for validation if needed
            print(f"Evaluating epoch {epoch+1}...")
            
            # Note: You might need to modify get_eval function to accept tqdm progress bar
            # or create a wrapper around it to show validation progress
            dices, mean_dice, _, val_losses = get_eval(valloader, model, criterion=criterion, opt=opt, args=args)
            
            print(f'\nEpoch [{epoch+1}/{opt.epochs}], Val Loss: {val_losses:.4f}')
            print(f'Epoch [{epoch+1}/{opt.epochs}], Val Dice: {mean_dice:.4f}')
            
            # Enhanced WandB logging with comprehensive metrics
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch, "train/loss": avg_train_loss, "val/loss": val_losses, 
                    "val/dice_score": mean_dice, "train/learning_rate": optimizer.param_groups[0]['lr'], 
                    "val/best_dice": best_dice, "train/patience_counter": patience_counter,
                    "model/total_params": pytorch_total_params
                })
            
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
                
            # FIXED: Only save best model and replace old one
            if mean_dice > best_dice:
                best_dice = mean_dice
                patience_counter = 0  # Reset patience counter
                
                # Delete previous best model if it exists
                if best_model_path and os.path.exists(best_model_path):
                    try:
                        os.remove(best_model_path)
                        print(f"Removed previous best model: {best_model_path}")
                    except Exception as e:
                        print(f"Warning: Could not remove previous model {best_model_path}: {e}")
                
                # Save new best model
                best_model_path = os.path.join(checkpoint_dir, 
                    f'{args.modelname}_best_dice_{best_dice:.4f}_epoch_{epoch}.pth')
                torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=False)
                print(f"New best model saved: {best_model_path} (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{args.early_stopping}")
                
                # Early stopping check
                if patience_counter >= args.early_stopping:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    print(f"Best model saved at: {best_model_path}")
                    if args.use_wandb:
                        wandb.finish()
                    return
        
        # REMOVED: The duplicate model saving that was happening every epoch
        # This was causing the double saving issue
        
        # Only save logs at specified intervals
        if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
            if args.keep_log:
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

    print(f"\nTraining completed! Best Dice score: {best_dice:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()