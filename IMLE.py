import tensorflow as tf
import numpy as np
import os
import layer as ly
from vgg19 import VGG19
from op_base import op_base
from util import *
from functools import reduce
import math
import pickle

class IMLE(op_base):
    def __init__(self,args,sess):
        op_base.__init__(self,args)
        self.sess = sess
        self.sess_arg = tf.Session()
        self.summaries = []
        self.vgg = VGG19()
        
        # self.train_data_generater = load_image()

    def get_vars(self,name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = name)
        
    def get_single_var(self,name):
        var_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        if(var_collection):
            return  var_collection[0]
        else:
            return []

    def encoder(self,input_z,name = 'generate_img',is_training = True):
        hidden_num = 64
        output_dim = 64
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):

            x = ly.fc(input_z, hidden_num * 8 * (output_dim // 16) * (output_dim // 16),name = 'gen_fc_0')
            x = tf.reshape(x, shape=[self.imle_deep, output_dim // 16, output_dim // 16, hidden_num * 8]) ## 4, 4, 8*64

            x = ly.deconv2d(x,hidden_num * 4,name = 'g_deconv2d_0') ### 8,8, 256
            x = ly.batch_normal(x,name = 'g_deconv_bn_0',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,hidden_num * 2,name = 'g_deconv2d_1') ### 16,16, 128
            x = ly.batch_normal(x,name = 'g_deconv_bn_1',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,hidden_num,name = 'g_deconv2d_2') ### 32,32, 64
            x = ly.batch_normal(x,name = 'g_deconv_bn_2',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x, 3, name = 'g_deconv2d_3') ### 64,64, 3
            x = ly.batch_normal(x,name = 'g_deconv_bn_3',is_training = is_training)
            x = tf.nn.tanh(x)

            return x

    def eval(self):
        self.epoch = 1
        label, name = self.make_data_queue(eval = True)
        self.start(label, name,is_training = False)

    def perceptual_loss(self, gen, origin):
        alpha1 = 1.
        alpha2 = 1.
        _imle_deep = gen.get_shape().as_list()[0]
        ### content
        gen_conv3_content = self.vgg(gen,conv3 = True) ## imle_deep, h,w,c
        origin_conv3_content = self.vgg(origin,conv3 = True)  ## 1, h,w,c
        mix_origin_conv3_content = tf.concat( [ origin_conv3_content for i in range(_imle_deep) ], axis = 0 ) ## imle_deep, h,w,c
        content_distance = gen_conv3_content - mix_origin_conv3_content
        content_loss = tf.reduce_sum(tf.square(content_distance),axis = [1,2,3]) / 2. ## imle_deep

        ### style
        def cell_style_distance(fake,origin):
            origin = tf.concat( [ origin for _ in range(self.imle_deep) ], axis = 0 )
            _style_distance = fake - origin
            _distance_loss = tf.reduce_sum(tf.square(_style_distance),axis = [1,2])
            return _distance_loss
        gen_conv_style = gram(self.vgg(gen,include_all = True)) ## 3 , imle_deep, c, c
        origin_conv_style = gram(self.vgg(origin,include_all = True)) ## 3, 1, c, c
        
        mix_style_loss = tf.reduce_sum( [ cell_style_distance(item_fake, item_origin) for item_fake, item_origin in zip(gen_conv_style, origin_conv_style) ] )
        return alpha1 * content_loss + alpha2 * mix_style_loss

    def local_moment_loss(self, pred, gt):
        with tf.name_scope('local_moment_loss'):

            ksz, kst = 4, 2
            local_patch = tf.ones((ksz, ksz, 1, 1))
            c = pred.get_shape()[-1]

            # Normalize by kernel size
            pr_mean = tf.concat([tf.nn.conv2d(x, local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(pred, c, axis=3)], axis=3)
            pr_var = tf.concat([tf.nn.conv2d(tf.square(x), local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(pred, c, axis=3)], axis=3)
            pr_var = (pr_var - tf.square(pr_mean)/(ksz**2)) / (ksz ** 2)
            pr_mean = pr_mean / (ksz ** 2)

            gt_mean = tf.concat([tf.nn.conv2d(x, local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(gt, c, axis=3)], axis=3)
            gt_var = tf.concat([tf.nn.conv2d(tf.square(x), local_patch, strides=[1, kst, kst, 1], padding='VALID') for x in tf.split(gt, c, axis=3)], axis=3)
            gt_var = (gt_var - tf.square(gt_mean)/(ksz**2)) / (ksz ** 2)
            gt_mean = gt_mean / (ksz ** 2)

            # scaling by local patch size
            local_mean_loss = tf.reduce_mean(tf.abs(pr_mean - gt_mean), axis = [1,2,3])
            local_var_loss = tf.reduce_mean(tf.abs(pr_var - gt_var), axis = [1,2,3])
        return local_mean_loss + local_var_loss

    def normalizer(self,input,name = 'generator_z'):
        def _normal(item):
            _normal_weight = tf.reduce_sum(tf.square(input))
            return item / _normal_weight
        with tf.name_scope(name):
            x = tf.map_fn(_normal,input)
            return x

    def xavier_initializer(self,shape, gain = 1.):
        if(len(shape) == 4):
            fan_in = reduce( np.multiply, shape[1:] )  # 输入通道
            fan_out = reduce( np.multiply, shape[1:] )  # 输出通道
        if(len(shape) == 2):
            fan_in = 1000
            fan_out = 1000
        variation = (2/( fan_in +fan_out)) * gain
        std = math.sqrt(variation)
        result = np.random.normal(0,std,shape)
        return result

    def init_z(self,init_code):
        init_op = tf.assign(self.input_z,init_code)
        return init_op
    def imle_graph(self,g_opt):
        
        # tf.get_variable('noise',shape = [self.imle_deep,1000],initializer=tf.random_normal_initializer(mean=0.,stddev = 0.02))
        self.z = self.normalizer(self.input_z)
        fake_img = self.encoder(self.z) 
        mix_input_image = tf.concat( [ self.input_image for i in range(self.imle_deep)], axis = 0 )

        #### local moment loss
        # moment_loss = self.local_moment_loss(fake_img, mix_input_image)
        # imle_z_grad = tf.gradients(moment_loss,self.input_z)[0] ### 16, 1000
        
        #### l2 loss
        img_distance = fake_img - mix_input_image
        l2_loss = tf.reduce_sum(tf.square(img_distance),axis = [1,2,3]) / 2.
        min_index = tf.argmin(l2_loss)

        self.fake_img = fake_img[min_index]
        self.choose_noise = self.z[min_index]
        #### tv loss 
        _tv_loss = 0.0001 * tv_loss( tf.expand_dims(self.fake_img,axis = 0) )
        imle_gen_loss = tf.reduce_min(l2_loss) + _tv_loss

        self.summaries.append(tf.summary.scalar('g_min_loss',imle_gen_loss)) 
        self.gen_saver = tf.train.Saver(var_list=self.get_vars('generate_img'))

        gen_grad = g_opt.compute_gradients(imle_gen_loss,var_list = self.get_vars('generate_img') )
        ### clip gridents
        gen_op = g_opt.apply_gradients(gen_grad)

        return  gen_op

    def make_img(self,img,name):
        rgb_img = float_rgb(img)
        cv2.imwrite('eval/%s' % name ,rgb_img)

    def write_pickle(self, name, choose_z):
        pickle_write_path = os.path.join('result','%s.pickle' % name)
        with open(pickle_write_path,'wb') as f:
            f.write(pickle.dumps(choose_z))
    def save_gen(self):
        self.gen_saver.save(self.sess,os.path.join('imle_model','generator'))
    def restore_gen(self):
        if(os.listdir('imle_model')):
            self.gen_saver.restore(self.sess,os.path.join('imle_model','generator'))
    def train(self):
        self.input_image = tf.placeholder(tf.float32, shape = [1,self.image_height,self.image_weight,self.image_channels] )
        self.input_z = tf.placeholder(tf.float32, shape = [self.imle_deep,1000] ) ### 16, 1000 
        gen_optimizer = tf.train.AdamOptimizer(self.lr)
        gen_opt = self.imle_graph(gen_optimizer)

        ## init
        self.sess.run(tf.global_variables_initializer())
        self.restore_gen()
        
        ## summary init
        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        self.train_data_generater = load_image()
        for i in range(1914):
            img_content, name = next(self.train_data_generater)
            _img_content = np.expand_dims(img_content,axis = 0)

            for _ in range(500):
                init_noise = np.random.normal(scale=0.02,size = [self.imle_deep,1000])
                _feed_dict = {self.input_image:_img_content,self.input_z:init_noise}
                _g_op,_img,_summary_op = self.sess.run([gen_opt,self.fake_img,summary_op], feed_dict = _feed_dict)

            print('write img %s' % _)
            self.make_img(_img,name)
            self.save_gen()
            print('finish %s' % i)

            












