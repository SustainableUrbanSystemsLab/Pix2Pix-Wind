import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
import math
def lcm(a,b): return abs(a * b)//math.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer

def save_epoch_visuals(epoch, input_tensor, fake_tensor, real_tensor, opt, channel_names=None):
    """Save per-channel heatmaps at the end of each epoch for visual progress tracking."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed, skipping epoch visuals')
        return

    vis_dir = os.path.join('results', opt.name, 'epoch_visuals')
    os.makedirs(vis_dir, exist_ok=True)

    # Convert tensors to numpy (C, H, W) -> (H, W, C)
    inp = input_tensor.cpu().detach().numpy()
    fake = fake_tensor.cpu().detach().numpy()
    real = real_tensor.cpu().detach().numpy()

    n_out = fake.shape[0]  # number of output channels
    n_in = inp.shape[0]    # number of input channels

    if channel_names is None:
        out_names = [f'ch{i}' for i in range(n_out)]
    else:
        out_names = channel_names

    # Create comparison grid: 3 rows (input/fake/real first col, then output channels)
    n_cols = max(n_in, n_out)
    fig, axes = plt.subplots(3, n_cols, figsize=(4*n_cols, 12))
    fig.suptitle(f'Epoch {epoch}', fontsize=16, fontweight='bold')

    row_labels = ['Input', 'Generated', 'Ground Truth']
    data_rows = [inp, fake, real]
    for row_idx, (label, data) in enumerate(zip(row_labels, data_rows)):
        n_ch = data.shape[0]
        for ch in range(n_cols):
            ax = axes[row_idx, ch] if n_cols > 1 else axes[row_idx]
            if ch < n_ch:
                im = ax.imshow(data[ch], cmap='viridis', aspect='equal')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if row_idx == 0:
                    ax.set_title(f'Input ch{ch}', fontsize=10)
                else:
                    ch_name = out_names[ch] if ch < len(out_names) else f'ch{ch}'
                    ax.set_title(f'{label}: {ch_name}', fontsize=10)
            else:
                ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
        # Add row label
        if n_cols > 1:
            axes[row_idx, 0].set_ylabel(label, fontsize=12, fontweight='bold')
        else:
            axes[row_idx].set_ylabel(label, fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(vis_dir, f'epoch_{epoch:04d}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved epoch visual -> {save_path}')

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
# Helper to access model attributes whether wrapped in DataParallel or not
def get_model(m):
    return m.module if hasattr(m, 'module') else m
visualizer = Visualizer(opt)
if opt.fp16:    
    from apex import amp
    model, [optimizer_G, optimizer_D] = amp.initialize(model, [model.optimizer_G, model.optimizer_D], opt_level='O1')             
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
else:
    optimizer_G, optimizer_D = get_model(model).optimizer_G, get_model(model).optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(get_model(model).loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_D, optimizer_D) as scaled_loss: scaled_loss.backward()                
        else:
            loss_D.backward()        
        optimizer_D.step()        

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}            
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            get_model(model).save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save epoch visualization (per-channel heatmaps)
    if generated is not None and data is not None:
        out_names = None
        if getattr(opt, 'dataset_mode', '') == 'wind':
            out_names = ['mag_U', 'Cp_from_U', 'k_from_U', 'epsilon_from_U']
        save_epoch_visuals(epoch, data['label'][0], generated.data[0],
                           data['image'][0], opt, channel_names=out_names)

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        get_model(model).save('latest')
        get_model(model).save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        get_model(model).update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        get_model(model).update_learning_rate()
