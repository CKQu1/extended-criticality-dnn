import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ast import literal_eval
from tqdm import tqdm

# changed
from PHDimGeneralization.topology import calculate_ph_dim
from PHDimGeneralization.utils import accuracy, get_data
from PHDimGeneralization.models import alexnet, vgg

# self 
from os.path import join
from path_names import root_data


def str_force_bool(s):
    s = s if isinstance(s,bool) else literal_eval(s)
    return s


def get_weights(net):
    with torch.no_grad():
        w = []
        for p in net.parameters():
            w.append(p.view(-1).detach().to(torch.device('cpu')))
        return torch.cat(w)


def eval(eval_loader, net, crit, opt, dev, test=True):
    net.eval()

    # run over both test and train set
    with torch.no_grad():    
        total_size = 0
        total_loss = 0
        total_acc = 0
        grads = []
        outputs = []

        P = 0 # num samples / batch size
        for x, y in eval_loader:
            P += 1
            # loop over dataset
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(x)
            
            outputs.append(out)

            loss = crit(out, y)
            prec = accuracy(out, y)
            bs = x.size(0)

            total_size += int(bs)
            total_loss += float(loss) * bs
            total_acc += float(prec) * bs
        
    hist = [
        total_loss / total_size, 
        total_acc / total_size,
        ]

    print(hist)
    
    return hist, outputs, 0#, noise_norm

def train_dnn(alpha100, g100, hidden_structure, dataset, model, depth, width, 
              optimizer, iterations, batch_size_train, batch_size_eval, lr, 
              ignore_previous=True, mom=0, path='data',                   
              print_freq=100, eval_freq=100, seed=0, scale=64, 
              meta_data='results', save_file='dim.txt', double=False, lr_schedule=False, bn=False):

    from time import time
    from NetPortal.models import ModelFactory              

    t0 = time()
    dev = torch.device(f"cuda:{torch.cuda.device_count()-1}"
                    if torch.cuda.is_available() else "cpu")
    print(f"Device in use:{dev}")    

    alpha100, g100 = int(alpha100), int(g100)
    hidden_structure = int(hidden_structure)
    depth, width = int(depth), int(width)
    iterations = int(iterations)
    batch_size_train, batch_size_eval = int(batch_size_train), int(batch_size_eval)
    lr = float(lr)
    ignore_previous = str_force_bool(ignore_previous)
    
    # initial setup
    if double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    torch.manual_seed(seed)

    # check to see if stuff has run already
    if not ignore_previous:
        with open(save_file, 'r') as f:
            for line in f.readlines():
                if meta_data == line.split(',')[0]:
                    print(f"Metadata {meta_data} already ran. Exiting.")
                    exit()

    # training setup
    train_loader, test_loader_eval, train_loader_eval, num_classes = get_data(dataset, path, batch_size_train, batch_size_eval)

    if model == 'fc':
        alpha100, g100 = int(alpha100), int(g100)
        hidden_structure = hidden_structure
        if dataset == 'mnist':
            N, C = 784, 10
        elif dataset == 'cifar10':
            N, C = 3072, 10

        hidden_N = [N]
        if hidden_structure == 1:   # network setup 1 (power-law decreasing layer size)
            a = (C/N)**(1/depth)
            hidden_N = hidden_N + [int(N*a**l) for l in range(1, depth)]
        elif hidden_structure == 2:   # network setup 2 (square weight matrix)
            hidden_N = hidden_N + [N]*(depth - 1)
        else: 
            assert isinstance(hidden_structure,list), 'hidden_N must be a list!'
            assert len(hidden_structure) == depth - 1, 'hidden_N length and depth inconsistent!'
            hidden_N += hidden_structure

        hidden_N.append(C)
        alpha, g = alpha100/100, g100/100
        activation = "tanh"
        with_bias = False
        net_type = model

        kwargs = {"dims": hidden_N, "alpha": alpha, "g": g,
                 "init_path": None, "init_epoch": None,
                 "activation": activation, "with_bias": with_bias,
                 "architecture": net_type}
        net = ModelFactory(**kwargs)

    elif model == 'alexnet':
        if dataset == 'mnist':
            net = alexnet(input_height=28, input_width=28, input_channels=1, num_classes=num_classes)
        else:
            net = alexnet(ch=scale, num_classes=num_classes).to(dev)
    elif model == 'vgg':
        net = vgg(depth=depth, num_classes=num_classes, batch_norm=bn).to(dev)

    print(net)
    
    opt = getattr(optim, optimizer)(
        net.parameters(), 
        lr=lr
        )

    if lr_schedule:
        milestone = int(iterations / 3)
        scheduler = optimizer.lr_scheduler.MultiStepLR(opt, 
            milestones=[milestone, 2*milestone],
            gamma=0.5)
    
    crit = nn.CrossEntropyLoss().to(dev)
    
    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)
    
    # training logs per iteration
    training_history = []

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []

    # weights
    weights_history = deque([])

    STOP = False

    for i, (x, y) in tqdm(enumerate(circ_train_loader)):

        if i % eval_freq == 0:
            # first record is at the initial point
            te_hist, te_outputs, te_noise_norm = eval(test_loader_eval, net, crit, opt, dev)
            tr_hist, tr_outputs, tr_noise_norm = eval(train_loader_eval, net, crit, opt, dev, test=False)
            evaluation_history_TEST.append([i, *te_hist])
            evaluation_history_TRAIN.append([i, *tr_hist])
            if int(tr_hist[1]) == 100:
                print('yaaay all training data is correctly classified!!!')
                STOP = True

        net.train()
        
        x, y = x.to(dev), y.to(dev)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        if torch.isnan(loss):
            print('Loss has gone nan :(.')
            STOP = True

        # calculate the gradients
        loss.backward()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])

        # take the step
        opt.step()

        if i % print_freq == 0:
            print(training_history[-1])

        if lr_schedule:
            scheduler.step(i)

        if i > iterations:
            STOP = True

        weights_history.append(get_weights(net))
        if len(weights_history) > 1000:
            weights_history.popleft()

        # clear cache
        torch.cuda.empty_cache()

        if STOP:
            assert len(weights_history) == 1000

            # final evaluation and saving results
            print('eval time {}'.format(i))
            te_hist, te_outputs, te_noise_norm = eval(test_loader_eval, net, crit, opt, dev)
            tr_hist, tr_outputs, tr_noise_norm = eval(train_loader_eval, net, crit, opt, dev, test=False)
            evaluation_history_TEST.append([i + 1, *te_hist]) 
            evaluation_history_TRAIN.append([i + 1, *tr_hist])

            weights_history_np = torch.stack(tuple(weights_history)).numpy()
            del weights_history
            ph_dim = calculate_ph_dim(weights_history_np)

            test_acc = evaluation_history_TEST[-1][2]
            train_acc = evaluation_history_TRAIN[-1][2]

            with open(save_file, 'a') as f:
                f.write(f"{meta_data}, {train_acc}, {test_acc}, {ph_dim}\n")
            
            break    

    total_time = time() - t0
    print(f"All computations finished in {total_time}s")


def fcn_submit(*args):
    from qsub import qsub, job_divider, project_ls

    dataset_ls = ["mnist"]

    alpha100_ls = [120, 200]
    #g100_ls = [25,100,300]
    g100_ls =[100]
    optimizer_ls = ["SGD"]
    bs_ls = [1024]  
    lr_ls = [0.001]
    hidden_structure = 2
    model = 'fc'
    depth = 3
    width = 0
    iterations = 1000
    batch_size_train = 1024
    batch_size_eval = batch_size_train
    #ignore_previous = True    


    # submissions
    pbs_array_data = [(alpha100, g100, hidden_structure, dataset_name, model, depth, width, 
                       optimizer, iterations, batch_size_train, batch_size_eval, lr)
                      for alpha100 in alpha100_ls
                      for g100 in g100_ls
                      for dataset_name in dataset_ls
                      for optimizer in optimizer_ls
                      for lr in lr_ls
                      ]


    
    perm, pbss = job_divider(pbs_array_data, len(project_ls))
    for idx, pidx in enumerate(perm):
        pbs_array_true = pbss[idx]
        print(project_ls[pidx])
        qsub(f'python {sys.argv[0]} {" ".join(args)}',    
             pbs_array_true, 
             path=join("phdim_analysis"),
             P=project_ls[pidx],
             #ngpus=1,
             ncpus=1,
             #walltime='0:59:59',
             walltime='23:59:59',
             mem='4GB') 


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])


