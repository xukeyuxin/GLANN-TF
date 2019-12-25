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
        self.train_data_generater = load_image()

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

            # x = ly.conv2d(x, 16, name='e_conv2d_0') ### 8, 8, 16
            # x = ly.batch_normal(x, name='e_bn_0', is_training=is_training)
            # x = ly.relu(x)

            # x = ly.conv2d(x, 32, name='e_conv2d_1') ### 8, 8, 32
            # x = ly.batch_normal(x, name='e_bn_1', is_training=is_training)
            # x = ly.relu(x)


            # x = ly.conv2d(x, 64, name='e_conv2d_2') ### 8, 8, 64
            # x = ly.batch_normal(x, name='e_bn_2', is_training=is_training)
            # x = ly.relu(x)

            # x = ly.conv2d(x, 128, name='e_conv2d_3') ### 8, 8, 128
            # x = ly.batch_normal(x, name='e_bn_3', is_training=is_training)
            # x = ly.relu(x)

            x = ly.deconv2d(x,hidden_num * 4,name = 'g_deconv2d_0') ### 8,8, 128
            x = ly.batch_normal(x,name = 'g_deconv_bn_0',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,hidden_num * 2,name = 'g_deconv2d_1') ### 16,16, 64
            x = ly.batch_normal(x,name = 'g_deconv_bn_1',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x,hidden_num,name = 'g_deconv2d_2') ### 32,32, 64
            x = ly.batch_normal(x,name = 'g_deconv_bn_2',is_training = is_training)
            x = ly.relu(x)

            x = ly.deconv2d(x, 3, name = 'g_deconv2d_3') ### 64,64, 64
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
        _imle_deep = tf.shape(gen)[0]
        ### content
        gen_conv3_content = self.vgg(gen,conv3 = True) ## imle_deep, h,w,c
        origin_conv3_content = self.vgg(origin,conv3 = True)  ## 1, h,w,c
        mix_origin_conv3_content = tf.concat( [ origin_conv3_content for i in range(_imle_deep) ], axis = 0 ) ## imle_deep, h,w,c
        content_distance = gen_conv3_content - mix_origin_conv3_content
        content_loss = tf.reduce_sum(tf.square(content_distance),axis = [1,2,3]) / 2. ## imle_deep

        ### style
        gen_conv3_style = gram(self.vgg(gen,conv3 = True)) ## imle_deep, c, c
        origin_conv3_style = gram(self.vgg(origin,conv3 = True)) ## 1, c, c
        mix_origin_conv3_style = tf.concat( [ origin_conv3_style for i in range(_imle_deep) ], axis = 0 ) ## imle_deep, c, c
        style_distance = gen_conv3_style - mix_origin_conv3_style ## imle_deep, c, c
        
        style_loss = tf.reduce_sum(tf.square(style_distance),axis = [1,2]) ## imle_deep

        return alpha1 * content_loss + alpha2 * style_loss

    def dense_generator(self,input,name = 'generator_z'):
        with tf.variable_scope('generator_z',reuse = tf.AUTO_REUSE):
            x = ly.fc(input,1000,name = 'generate_z_fc')
            x = ly.batch_normal(x,name = 'generate_z_bn_0')
            return x

    def glann_graph(self,z_opt,g_opt):
        
        # tf.get_variable('noise',shape = [self.imle_deep,1000],initializer=tf.random_normal_initializer(mean=0.,stddev = 0.02))

        self.z = self.dense_generator(self.input_z)  ### 16, 1000
        fake_img = self.encoder(self.z) 
        mix_input_image = tf.concat( [ self.input_image for i in range(self.imle_deep)], axis = 0 )
        img_distance = fake_img - mix_input_image
        #### l2 loss
        l2_loss = tf.reduce_sum(tf.square(img_distance),axis = [1,2,3]) / 2.
        imle_z_index = tf.argmin(l2_loss)
        imle_z_loss = tf.reduce_min(l2_loss)
        imle_z_mean_loss = tf.reduce_mean(l2_loss)
        self.summaries.append(tf.summary.scalar('z_min_loss',imle_z_loss)) 
        self.summaries.append(tf.summary.scalar('z_mean_loss',imle_z_loss)) 
        imle_choose_z = self.z[imle_z_index]
        imle_choose_img = tf.expand_dims(fake_img[imle_z_index],axis = 0)

        #### perceptual_loss
        # perceptual_loss = self.perceptual_loss(fake_img, self.input_image)
        # imle_gen_loss = tf.reduce_min(perceptual_loss)
        # imle_gen_mean_loss = tf.reduce_mean(perceptual_loss)

        perceptual_loss = self.perceptual_loss(imle_choose_img, self.input_image)
        imle_gen_loss = tf.reduce_min(perceptual_loss)
        imle_gen_mean_loss = tf.reduce_mean(perceptual_loss)
        
        self.summaries.append(tf.summary.scalar('g_min_loss',imle_gen_loss)) 
        self.summaries.append(tf.summary.scalar('g_mean_loss',imle_gen_mean_loss)) 
        
        z_grad = z_opt.compute_gradients(imle_z_loss,var_list = self.get_vars('generator_z'))
        gen_grad = g_opt.compute_gradients(imle_gen_mean_loss,var_list = self.get_vars('generate_img') )

        return z_grad, gen_grad

    def train(self):
        self.input_image = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.image_channels])
        self.input_z = tf.placeholder(tf.float32,shape = [self.imle_deep,1000] )

        z_optimizer = tf.train.AdamOptimizer(self.lr)
        gen_optimizer = tf.train.AdamOptimizer(self.lr)
        z_grad, gen_grad = self.glann_graph(z_optimizer,gen_optimizer)
        z_opt = z_optimizer.apply_gradients(z_grad)
        gen_opt = gen_optimizer.apply_gradients(gen_grad)

        ## init
        self.sess.run(tf.global_variables_initializer())

        ## summary init
        summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        summary_op = tf.summary.merge(self.summaries)

        for i in range(10000):
            img_content, name = next(self.train_data_generater)
            _img_content = np.expand_dims(img_content,axis = 0)
            _input_z = np.random.normal(size = [self.imle_deep,1000] )
            _feed_dict = {self.input_image:_img_content,self.input_z:_input_z}
            for _ in range(10):
                _z_op,_summary_str = self.sess.run([z_opt,summary_op], feed_dict = _feed_dict)
                summary_writer.add_summary(_summary_str,i)

            _g_op,_summary_op = self.sess.run([gen_opt,summary_op], feed_dict = _feed_dict)
            summary_writer.add_summary(_summary_str,i)

            






    # def start(self,is_training = True,pre_train = False):

    #     ## lr
    #     global_steps = tf.get_variable(name='global_step', shape=[], initializer=tf.constant_initializer(0),
    #                                    trainable=False)
    #     decay_change_batch_num = 200.0
    #     train_data_num = 2000 * self.epoch
    #     decay_steps = (train_data_num / self.batch_size / self.gpu_nums) * decay_change_batch_num

    #     lr = tf.train.exponential_decay(self.lr,
    #                                     global_steps,
    #                                     decay_steps,
    #                                     0.1,
    #                                     staircase=True)

    #     self.summaries.append(tf.summary.scalar('lr',lr))

    #     ## opt
    #     vgg_opt = tf.train.AdamOptimizer(lr)

    #     ## graph
    #     self.input_image = tf.placeholder(tf.float32,shape = [self.batch_size,self.image_height,self.image_weight,self.image_channels])
    #     self.input_index = tf.placeholder(tf.int32,shape = [self.batch_size])

    #     ## one_hot
    #     # self.embedding_vocab = tf.get_variable('embedding',initializer = tf.one_hot(tf.range(self.vocab_size),self.vocab_size))

    #     ## random_normal
    #     self.embedding_vocab = tf.get_variable('embedding',shape = [self.vocab_size,self.vocab_dim],initializer = tf.random_normal_initializer(stddev = 0.02,mean = 0))
    #     # self.update_vocab = tf.nn.embedding_lookup(self.embedding_vocab,self.input_index)

    #     ## embedding dim
    #     update_vocab = tf.nn.embedding_lookup(self.embedding_vocab, self.input_index)

    #     ### index 对应的编码
    #     encode = self.embedding_graph(update_vocab)

    #     one_moid_encode = tf.map_fn(update_embedding_mold,encode)


    #     ### one_moid vgg-loss
    #     vgg_loss, vgg_grad = self.vgg_graph(one_moid_encode,self.input_image,vgg_opt,is_training = is_training)
    #     self.summaries.append(tf.summary.scalar('loss',vgg_loss))
    #     ### grad_op
    #     vgg_grad_op = vgg_opt.apply_gradients(vgg_grad,global_step=global_steps)

    #     ### variable_op
    #     train_op = tf.group(vgg_grad_op)

    #     ### variable_summary
    #     for var in tf.trainable_variables():
    #         print( 'name: %s, shape: %s' % (var.op.name, reduce( lambda x,y:x * y, var.get_shape().as_list()) ))
    #         self.summaries.append(tf.summary.histogram(var.op.name, var))
    #     ## init
    #     self.init_sess(self.sess,[tf.global_variables_initializer(),tf.local_variables_initializer()])

    #     ## summary init
    #     summary_writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
    #     summary_op = tf.summary.merge(self.summaries)

    #     # ## queue init
    #     # coord = tf.train.Coordinator()
    #     # thread = tf.train.start_queue_runners(sess = self.sess)

    #     ### train
    #     saver = tf.train.Saver(max_to_keep = 1)
    #     step = 1
    #     print('start train')
    #     if(is_training):
    #         if(pre_train):
    #             saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
    #             print('restore success')
    #         for _ in range(self.epoch):
    #             for i in range(0,self.vocab_size,self.batch_size):
    #                 if( (i+self.batch_size) >= (self.vocab_size - 1)):
    #                     continue
    #                 one_batch_index = range(i, i+self.batch_size)
    #                 one_batch_image = load_one_batch_image(self,one_batch_index)
    #                 print('start %s' % step)
    #                 _g,_str = self.sess.run([train_op,summary_op],feed_dict = {self.input_image:one_batch_image,self.input_index:one_batch_index})
    #                 if(step % 10 == 0):
    #                     print('update summary')
    #                     summary_writer.add_summary(_str,step)
    #                 if(step % 100 == 0):
    #                     print('update model')
    #                     saver.save(self.sess,os.path.join(self.model_save_path,'model_%s.ckpt' % step))
    #                 step += 1


    #     if(not is_training):
    #         saver.restore(self.sess, tf.train.latest_checkpoint(self.model_save_path))
    #         print('restore success')
    #         try:
    #             while not coord.should_stop():
    #                 print('start %s' % step)
    #                 _fake, _eval_name = self.sess.run([self.eval_fake,self.eval_name])
    #                 make_image(_fake,_eval_name)

    #         except tf.errors.OutOfRangeError:
    #             print('finish thread')
    #         finally:
    #             coord.request_stop()

    #         coord.join(thread)
    #         print('thread break')

















