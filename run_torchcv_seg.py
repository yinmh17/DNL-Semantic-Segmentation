import argparse
import os
import time
def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)
print('rank: {}'.format(ompi_rank()))
#os.environ['NCCL_IB_HCA'] = 'mlx5_0,mlx5_2'
print('env:')
print(os.environ)
os.system("ls")
parser = argparse.ArgumentParser(description='Helper run')
# general
parser.add_argument('--auth', help='auth', required=True, type=str)
#parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--branch', help="branch of code", type=str, default='master')
parser.add_argument('--data_dir', help='data input', type=str, default='/vcgrr/v-zhuyao/data/VOCContext/torchcv')
parser.add_argument('--restore', help='pretrained model', type=str, default='./pretrained_models/3x3resnet101-imagenet.pth')
parser.add_argument('--config', help='experiment configure file name', required=True, type=str)
parser.add_argument('--nettype', help='type of network', required=True, type=str)
parser.add_argument('--dataset', help='dataset', required=True, type=str)

# nettype in [nonlocal, nonlocalnowd, gcnet, pspnet]
# dataset in [pcontext, ade20k, cityscapes]

args, rest = parser.parse_known_args()
print(args)
print(rest)
os.system('ls')
#if os.path.exists('torchcv'):
#    os.system('rm -r torchcv')
# is_worker = ompi_rank() != 0 and ompi_size() > 1
os.system("git clone https://{0}@github.com/yinmh17/torchcv -b {1} $HOME/torchcv".format(args.auth, args.branch))
# only master need to install package
os.chdir(os.path.expanduser('~/torchcv'))
os.system('pip3 install -r requirements.txt')
#os.system('pip3 install pyhocon')
os.chdir(os.path.expanduser('exts'))
os.system('sh make.sh')
os.chdir(os.path.abspath('..'))

#os.system('ln -s /hdfs/resrchprojvc2/zhez/data .')
#os.system('ln -s /hdfs/resrchprojvc2/zhez/model .')
# os.system('ln -s /home/zhez/data_local .')
os.system('ls')

os.chdir(os.path.expanduser('~/torchcv/scripts/seg/{0}'.format(args.dataset)))
os.system('ls')
print('Rank {}'.format(ompi_rank()))
os.system('bash run_fs_res101_{0}_{1}_seg.sh train tag {2} {3} {4}'.format(args.nettype, args.dataset, args.data_dir, args.restore, args.config))
os.system('bash run_fs_res101_{0}_{1}_seg.sh val tag {2} {3} {4}'.format(args.nettype, args.dataset, args.data_dir, args.restore, args.config))
