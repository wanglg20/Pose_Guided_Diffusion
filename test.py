from models.epipolar_unet import *

from models.unet import *
from utils.script_util import *

#MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 
from utils.dataset import *
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utils.logger as logger
from utils.train_util import *

# print(feature.shape)from utils.script_utils import *
from utils.resample import *

keys = {'image_size': 64, 'num_channels': 192, 
        'num_res_blocks': 3, 'num_heads': 4, 'num_heads_upsample': -1, 
        'num_head_channels': 64, 'attention_resolutions': '32,16,8', 'channel_mult': '', 'dropout': 0.1, 
        'class_cond': True, 'use_checkpoint': False, 'use_scale_shift_norm': True, 
        'resblock_updown': True, 'use_fp16': True, 'use_new_attention_order': True, 
        'learn_sigma': True, 'diffusion_steps': 1000, 'noise_schedule': 'cosine',
        'timestep_respacing': '250', 'use_kl': False, 'predict_xstart': False, 'rescale_timesteps': False, 
        'rescale_learned_sigmas': False, 'use_epipolar': True}

sample_args = {'schedule_sampler': 'uniform', 'num_epochs': 1000}

model, diffusion = create_model_and_diffusion(**keys)
schedule_sampler = create_named_schedule_sampler(sample_args['schedule_sampler'], diffusion)


batch_size = 1
microbatch = batch_size
info_root = 'dataset/RealEstate10K'
data_root = 'dataset/RealEstate_data'
data = load_data(data_dir= data_root, info_dir=info_root, batch_size=batch_size, image_size = 64, mode='test')
lr = 1e-4 


TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ).run_loop()

