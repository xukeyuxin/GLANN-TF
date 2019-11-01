import tensorflow as tf
import numpy as np
import os
import layer as ly
from vgg19 import VGG19
from op_base import op_base
from util import *
from functools import reduce

class GLANN(op_base):
    def __init__(self,args,sess):
        op_base.__init__(self,args)
        self.sess = sess
        self.sess_arg = tf.Session()
        self.summaries = []
        self.vgg = VGG19()

    def init_sess(self,sess,init):
        sess.run(init)

    def get_vars(self,name):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = name)

    def embedding_graph(self,update_vocab):

        encode = ly.fc(update_vocab,self.embedding_size,name = 'embedding_fc_0')
        encode = ly.batch_normal(encode,name = 'embedding_bn_0')
        encode = ly.relu(encode)

        return  encode

    def vgg_graph(self,input_z,label,vgg_opt,is_training = True):
        fake = self.encoder(input_z,name = 'encode',is_training = is_training)
        vgg_loss = tf.reduce_sum( tf.square( self.vgg(fake) - self.vgg(label) ),axis = [0,1,2,3] )
        print(vgg_loss)
        vgg_gred = vgg_opt.compute_gradients(vgg_loss,var_list = self.get_vars('encode') + self.get_vars('embedding'))
        return vgg_loss, vgg_gred

    def encoder(self,input_z,name = 'encode',is_training = True):
        with tf.variable_scope(name,reuse = tf.AUTO_REUSE):

            x = tf.reshape(input_z, shape=[-1, 8, 8, 8])

            x = ly.conv2d(x, 16, name='e_conv2d_0') ### 8, 8, 16
            x = ly.batch_normal(x, name='e_bn_0', is_training=is_training)
            x = ly.relu(x)

            x = ly.conv2d(x, 32, name='e_conv2d_1') ### 8, 8, 32
            x = ly.batch_normal(x, name='e_bn_1', is_training=is_training)
            x = ly.relu(x)


            x = ly.conv2d(x, 64, name='e_conv2d_2') ### 8, 8, 64
            x = ly.batch_normal(x, name='e_bn_2', is_training=is_training)
            x = ly.relu(x)

            x = ly.conv2d(x, 128, name='e_conv2d_3') ### 8, 8, 128
            x = ly.batch_normal(x, name='e_bn_3', is_training=is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,64,name = 'e_deconv2d_0') ### 16,16, 128
            x = ly.batch_normal(x,name = 'e_bn_5',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,32,name = 'e_deconv2d_1') ### 32,32, 64
            x = ly.batch_normal(x,name = 'e_bn_6',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,16,name = 'e_deconv2d_2') ### 64,64,32
            x = ly.batch_normal(x,name = 'e_bn_7',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,8,name = 'e_deconv2d_3') ### 128,128,16
            x = ly.batch_normal(x,name = 'e_bn_8',is_training = is_training)
            x = ly.relu(x)


            x = ly.deconv2d(x,3,name = 'e_deconv2d_4') ### (256,256,3)
            x = ly.batch_normal(x,name = 'e_bn_9',is_training = is_training)
            x = tf.nn.tanh(x)

            return x


    def get_vars(self, name, scope=None):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def get_single_var(self,name):
        var_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        if(var_collection):
            return  var_collection[0]
        else:
            return []

    def make_data_queue(self,eval = False):
        images_label, image_names = load_image(self,eval = eval)
        input_queue = tf.train.slice_input_producer([images_label, image_names], num_epochs=self.epoch, shuffle=False)
        label, name = tf.train.batch(input_queue, batch_size=self.batch_size, num_threads=2,
                                      capacity=64,
                                      allow_smaller_final_batch=False)

        return label, name

    def train(self):
        # label, name = self.make_data_queue()
        self.start(pre_train = False)

    def eval(self):
        self.epoch = 1
        label, name = self.make_data_queue(eval = True)
        self.start(label, name,is_training = False)


    def start(self,is_training = True,pre_train = False):

        ## lr
        global_steps = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
                                       trainable=False)
        decay_change_batch_num = 200.0
        train_data_num = 2000 * self.epoch
        decay_steps = (train_data_num / self.batch_size / self.gpu_nums) * decay_change_batch_num

        lr = tf.train.exponential_decay(self.lr,
                                        global_steps,
                                        decay_steps,
                                        0.1,
                                        staircase=True)

        self.summaries.append(tf.summary.scalar('lr',lr))

        ## opt
        vgg_opt = tf.train.AdamOptimizer(lr)

        ## graph
        self.input_image = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.image_channels])
        self.input_index = tf.placeholder(tf.int32,shape = [self.batch_size])

        ## one_hot
        # self.embedding_vocab = tf.get_variable('embedding',initializer = tf.one_hot(tf.range(self.vocab_size),self.vocab_size))

        ## random_normal
        self.embedding_vocab = tf.get_variable('embedding',shape = [self.vocab_size,self.vocab_dim],initializer = tf.random_normal_initializer(stddev = 0.02,mean = 0))
        # self.update_vocab = tf.nn.embedding_lookup(self.embedding_vocab,self.input_index)

        ## embedding dim
        update_vocab = tf.nn.embedding_lookup(self.embedding_vocab, self.input_index)

        ### index 对应的编码
        encode = self.embedding_graph(update_vocab)

        one_moid_encode = tf.map_fn(update_embedding_mold,encode)


        ### one_moid vgg-loss
        vgg_loss, vgg_grad = self.vgg_graph(one_moid_encode,self.input_image,vgg_opt,is_training = is_training)
        self.summaries.append(tf.summary.scalar('loss',vgg_loss))
        ### grad_op
        vgg_grad_op = vgg_opt.apply_gradients(vgg_grad,global_step=global_steps)

        ### variable_op
        train_op = tf.group(vgg_grad_op)

        ### variable_summary
        for var in tf.trainable_variables():
            print( 'name: %s, shape: %s' % (var.op.name, reduce( lambda x,y:x * y, var.get_shape().as_list()) ))
            self.summaries.append(tf.summary.histogram(var.op.name, var))
        ## init
        self.init_sess(self.sess,[tf.global_variables_initializer(),tf.local_variables_initializer()])

        ## summary init
        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        # ## queue init
        # coord = tf.train.Coordinator()
        # thread = tf.train.start_queue_runners(sess = self.sess)

        ### train
        saver = tf.train.Saver(max_to_keep = 1)
        step = 1
        print('start train')
        if(is_training):
            if(pre_train):
                saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
                print('restore success')
            for _ in range(self.epoch):
                for i in range(0,self.vocab_size,self.batch_size):
                    if( (i+self.batch_size) >= (self.vocab_size - 1)):
                        continue
                    one_batch_index = range(i, i+self.batch_size)
                    one_batch_image = load_one_batch_image(self,one_batch_index)
                    print('start %s' % step)
                    _g,_str = self.sess.run([train_op,summary_op],feed_dict = {self.input_image:one_batch_image,self.input_index:one_batch_index})
                    if(step % 10 == 0):
                        print('update summary')
                        summary_writer.add_summary(_str,step)
                    if(step % 100 == 0):
                        print('update model')
                        saver.save(self.sess,os.path.join(self.model_save_path,'model_%s.ckpt' % step))
                    step += 1



        if(not is_training):
            saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
            print('restore success')
            try:
                while not coord.should_stop():
                    print('start %s' % step)
                    _fake, _eval_name = self.sess.run([self.eval_fake,self.eval_name])
                    make_image(_fake,_eval_name)

            except tf.errors.OutOfRangeError:
                print('finish thread')
            finally:
                coord.request_stop()

            coord.join(thread)
            print('thread break')

















