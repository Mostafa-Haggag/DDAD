import torch
import numpy as np
import os
import argparse
import time
from unet import *
from test import validate
from omegaconf import OmegaConf
from utilities import *
import torch.nn.functional as F
from train import trainer
from datetime import timedelta
from EMA import EMAHelper
from feature_extractor import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


def constant(config):
    # Define beta schedule

    betas = beta_schedule(beta_schedule = config.model.schedule, beta_start = config.model.beta_start, beta_end=config.model.beta_end, num_diffusion_timesteps=config.model.trajectory_steps)

    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    constants_dict = {
        'betas' : betas,
        'alphas': alphas,
        'alphas_cumprod' : alphas_cumprod,
        'alphas_cumprod_prev' : alphas_cumprod_prev,
        'sqrt_recip_alphas' : sqrt_recip_alphas,
        'sqrt_alphas_cumprod' : sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod' : sqrt_one_minus_alphas_cumprod,
        'posterior_variance' : posterior_variance,
    }
    return constants_dict


def build_model(config):
    #model = SimpleUnet()
    unet = UNetModel(256, 64, dropout=0, n_heads=4 ,in_channels=config.data.imput_channel)
    return unet

    

def train(args):
    config = OmegaConf.load(args.config)
    
    unet = build_model(config)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    unet = unet.to(config.model.device)
    unet.train()
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        
        ema_helper.register(unet)
    else:
        ema_helper = None
    unet = torch.nn.DataParallel(unet)
    # checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'30000'))
    # model.load_state_dict(checkpoint)  
    constants_dict = constant(config)
    start = time.time()
    trainer(unet, constants_dict, ema_helper, config)
    end = time.time()
    print('training time on ',config.model.epochs,' epochs is ', str(timedelta(seconds=end - start)),'\n')
    with open('readme.txt', 'a') as f:
        f.write('\n training time is {}\n'.format(str(timedelta(seconds=end - start))))



def evaluate(args):
    start = time.time()
    config = OmegaConf.load(args.config)
    unet = build_model(config)
    if config.data.category:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), config.data.category,'5000')) # config.model.checkpoint_name 300+50
    else:
        checkpoint = torch.load(os.path.join(os.path.join(os.getcwd(), config.model.checkpoint_dir), '8000'))
    unet = torch.nn.DataParallel(unet)
    unet.load_state_dict(checkpoint)    
    unet.to(config.model.device)
    unet.eval()
    if False: #config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)
        ema_helper = torch.nn.DataParallel(ema_helper)
        ema_helper.load_state_dict(checkpoint)
        
        ema_helper.ema(model)
    else:
        ema_helper = None
    constants_dict = constant(config)
    validate(unet, constants_dict, config)
    end = time.time()
    print('Test time is ', str(timedelta(seconds=end - start)))





def parse_args():
    cmdline_parser = argparse.ArgumentParser('DDAD')    
    cmdline_parser.add_argument('-cfg', '--config', 
                                default= os.path.join(os.path.dirname(os.path.abspath(__file__)),'config.yaml'), 
                                help='config file')
    cmdline_parser.add_argument('--eval', 
                                default= False, 
                                help='only evaluate the model')
    args, unknowns = cmdline_parser.parse_known_args()
    return args


    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if args.eval:
        print('evaluating')
        # config = OmegaConf.load(args.config)
        # constants_dict = constant(config)
        # fake_real_dataset(args, constants_dict)
        evaluate(args)
    else:
        train(args)
        evaluate(args)

        