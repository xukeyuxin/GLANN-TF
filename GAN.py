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

class GAN(op_base):
    def __init__(self,args,sess):
        op_base.__init__(self,args)
        self.sess = sess
        self.sess_arg = tf.Session()
        self.summaries = []
        self.vgg = VGG19()
        self.model_path = os.path.join('gan_result','gan_model')
        self.code_path = os.path.join('gan_result','gan_encoder_code')
        self.eval_path = os.path.join('gan_result','gan_eval')
        
        # self.train_data_generater = load_image()

    def get_vars(self,name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = name)
        
    def get_single_var(self,name):
        var_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        if(var_collection):
            return  var_collection[0]
        else:
            return []

    def decoder(self,input_z,name = 'generate_img',is_training = True):
        hidden_num = 64
        output_dim = 64
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):

            x = ly.fc(input_z, hidden_num * 8 * (output_dim // 16) * (output_dim // 16),name = 'gen_fc_0')
            x = tf.reshape(x, shape=[self.batch_size, output_dim // 16, output_dim // 16, hidden_num * 8]) ## 4, 4, 8*64

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

    def discriminator(self,x,name = 'discriminator_img',is_training = True): ## 64,64,3
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
            x = ly.conv2d(x,64,strides=2,use_bias=True,name = 'd_conv_0') ## 32,32,64
            x = ly.batch_normal(x,name = 'd_bn_0',is_training = is_training)
            x = ly.relu(x,0.2)

            x = ly.conv2d(x,128,strides=2,use_bias=True,name = 'd_conv_1') ## 16,16,128
            x = ly.batch_normal(x,name = 'd_bn_1',is_training = is_training)
            x = ly.relu(x,0.2)

            x = ly.conv2d(x,256,strides=2,use_bias=True,name = 'd_conv_2') ## 8,8,256
            x = ly.batch_normal(x,name = 'd_bn_2',is_training = is_training)
            x = ly.relu(x,0.2)

            x = ly.conv2d(x,512,strides=2,use_bias=True,name = 'd_conv_3') ## 4,4,512
            x = ly.batch_normal(x,name = 'd_bn_3',is_training = is_training)
            x = ly.relu(x,0.2)

            x = ly.fc(x,1,name = 'fc_0')
            x = tf.nn.sigmoid(x)
            return x

    def eval(self):
        self.epoch = 1
        label, name = self.make_data_queue(eval = True)
        self.start(label, name,is_training = False)

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

    def gan_graph(self,g_opt,d_opt,is_training = True):
        
        fake_img = self.decoder(self.input_z, is_training = is_training) 
        fake_discriminate = self.discriminator(fake_img, is_training = is_training)
        real_discriminate = self.discriminator(self.input_image, is_training = is_training)

        #### tv loss 
        _tv_loss = 0.0005 * tv_loss(fake_img)
        ### log 
        safe_eplise = 1e-12
        discri_loss = - tf.reduce_sum(tf.log( real_discriminate + safe_eplise )) - tf.reduce_sum(tf.log( 1 - fake_discriminate + safe_eplise ))
        generate_loss = - tf.reduce_sum(tf.log( fake_discriminate + safe_eplise )) + _tv_loss


        self.summaries.append(tf.summary.scalar('generate_loss',generate_loss)) 
        self.summaries.append(tf.summary.scalar('discri_loss',discri_loss)) 
        self.fake_img = fake_img

        gen_grad = g_opt.compute_gradients(generate_loss,var_list = self.get_vars('generate_img') )
        gen_op = g_opt.apply_gradients(gen_grad)

        dis_grad = g_opt.compute_gradients(discri_loss,var_list = self.get_vars('discriminator_img') )
        dis_op = g_opt.apply_gradients(dis_grad)
        return dis_op, gen_op

    def make_img(self,img,name):
        if(len(img.shape) == 4):
            img = img[0]
        rgb_img = float_rgb(img)
        cv2.imwrite(os.path.join(self.eval_path,name) ,rgb_img)

    def write_pickle(self, name, choose_z):
        if(len(choose_z.shape) == 4):
            choose_z = choose_z[0]
        pickle_write_path = os.path.join(self.code_path,'%s.pickle' % name)
        with open(pickle_write_path,'wb') as f:
            f.write(pickle.dumps(choose_z))

    def restore_gen(self):
        if( os.path.exists(self.model_path) and os.listdir(self.model_path)):
            self.gen_saver.restore(self.sess,os.path.join(self.model_path,'generator'))
    def train(self,is_training = True):
        self.input_image = tf.placeholder(tf.float32, shape = [self.batch_size,self.image_height,self.image_weight,self.image_channels] )
        self.input_z = tf.get_variable('noise',shape = [self.batch_size,1000],initializer = tf.random_normal_initializer(stddev = 0.02))

        gen_optimizer = tf.train.AdamOptimizer(self.lr)
        dis_optimizer = tf.train.AdamOptimizer(self.lr)
        dis_op, gen_op = self.gan_graph(gen_optimizer,dis_optimizer,is_training = is_training)

        ## init
        self.sess.run(tf.global_variables_initializer())
        
        
        ## summary init
        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        self.train_data_generater = load_image(eval = not is_training)
        self.saver = tf.train.Saver()

        step = 0
        while True:
            try:
                img_content, name = next(self.train_data_generater)
                step += 1
            except StopIteration:
                self.train_data_generater = load_image(eval = False)
            
            ### with batch_size == 1
            _img_content = np.expand_dims(img_content,axis = 0)
            _feed_dict = {self.input_image:_img_content}

            if(is_training):
                _dis_op,_summary_str = self.sess.run([dis_op,summary_op], feed_dict = _feed_dict)
                summary_writer.add_summary(_summary_str,step)

                _gen_op,_summary_str = self.sess.run([gen_op,summary_op], feed_dict = _feed_dict)
                summary_writer.add_summary(_summary_str,step)
                
                if(step % 500 == 0):
                    self.saver.save(self.sess,os.path.join(self.model_path,'gan_%s' % step))
                    print('sucess save gan')
            else:
                self.restore_gen()
                _img = self.sess.run(self.fake_img,feed_dict = _feed_dict)
                self.make_img(_img,name)












