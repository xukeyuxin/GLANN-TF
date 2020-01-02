import tensorflow as tf
from GLANN import GLANN
from IMLE import IMLE
from GLO import GLO
import argparse
import os
import sys


parser = argparse.ArgumentParser()

# Train Data
parser.add_argument("-dd", "--data_dir", type=str, default="./Data")
parser.add_argument("-sd", "--summary_dir", type=str, default="./logs")
parser.add_argument("-ms", "--model_save_path", type=str, default="./model")

# Train Iteration
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("-vs", "--vocab_size", type=int, default=1000)
parser.add_argument("-vb", "--vocab_dim", type=int, default=100)
parser.add_argument("-es", "--embedding_size", type=int, default=512)

parser.add_argument("-ic", "--image_channels", type=int, default=3)
parser.add_argument("-iw", "--image_weight", type=int, default=64)
parser.add_argument("-ih", "--image_height", type=int, default=64)

parser.add_argument("-ndeep", "--imle_deep", type=int, default=16)
parser.add_argument("-ndim", "--z_deep", type=int, default=1000)

# parser.add_argument("-ndim", "--n_dim", type=int, default=64)
# parser.add_argument("-ed", "--en_dim", type=int, default=512)

parser.add_argument("-ind", "--input_deep", type=int, default=2048)
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-g", "--gpu_num", type=int, default="1")
parser.add_argument("-tu", "--train_utils", type=str, default='gpu')
parser.add_argument("-zl", "--z_lr", type=float, default=1e-2)
parser.add_argument("-l", "--lr", type=float, default=1e-4)




parser.add_argument("-ac", "--action", type=str, default='train')
parser.add_argument("-m", "--model", type=str, default='GLANN')



args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] =  '1'


dir_names = ['eval','logs','model','data']
for dir in dir_names:
    if(not os.path.exists(dir)):
        os.mkdir(dir)

if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        if(args.model == 'IMLE'):
            model = IMLE(args,sess)
        elif(args.model == 'GLANN'):
            model = GLANN(args,sess)
        elif(args.model == 'GLO'):
            model = GLO(args,sess)
        if(args.action == 'train'):
            model.train()
        elif(args.action == 'test'):
            model.eval()
