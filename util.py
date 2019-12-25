
import os
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams 

def rgb_float(image_content):
    return image_content / 127.5 - 1

def float_rgb(image_content):
    return (image_content + 1) * 127.5

def update_embedding_mold(cell_embed):
    return cell_embed / tf.sqrt(tf.reduce_sum(tf.square(cell_embed)))

def load_image(eval = False):
    if(eval):
        image_dir_name = os.path.join('data','eval_faces')
    else:
        # image_dir_name = os.path.join('data','random_faces')
        image_dir_name = os.path.join('data','random_faces')
    image_list = os.listdir(image_dir_name)
    image_content = []
    image_name = []
    image_n = []
    image_z = []
    for name in tqdm(image_list):
        _content = cv2.imread(os.path.join(image_dir_name,name)).astype(np.float32)
        _content = cv2.resize(_content,(self.image_height,self.image_weight))
        # image_content.append(cell_content)
        # image_name.append(name)

        yield rgb_float(_content), name

def load_one_batch_image(self,index_batch):
    image_dir_name = os.path.join('data','random_faces_1000')
    image_content = []
    for index in index_batch:
        cell_content = cv2.imread(os.path.join(image_dir_name,str(index) + '.jpg')).astype(np.float32)
        cell_content = cv2.resize(cell_content,(self.image_height,self.image_weight))
        image_content.append(cell_content)

    return rgb_float(np.array(image_content))


def make_image(input,name_list):
    write_dir = 'eval'
    image_content = float_rgb(input).astype(np.uint8)
    index = 0
    for cell in image_content:
        print(name_list[index].decode())
        cv2.imwrite(os.path.join(write_dir,name_list[index].decode()), cell)
        index += 1

