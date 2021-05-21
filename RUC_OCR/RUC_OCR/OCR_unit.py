#! /usr/bin/env python
# -*- coding: utf-8 -*-

# top 1 accuracy 0.99826 top 5 accuracy 0.99989

import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image
import cv2
from tensorflow.python.ops import control_flow_ops
import sys
from bs4 import BeautifulSoup
import requests
import math
from django.http import JsonResponse

reload(sys)
sys.setdefaultencoding('utf-8')
#os.environ["CUDA_VISIBLE_DEVICES"] = "8"
logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3524, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 50002, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 1000, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './en_model/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './en_dataset/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './en_dataset/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'validation', 'Running mode. One of {"train", "valid", "test"}')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
FLAGS = tf.app.flags.FLAGS

OCR_dict = {'firm_name':u'企业名称', 
            'firm_num':u'企业注册号',
            'firm_type':u'类型',
            'firm_address':u'住所',
            'firm_owner':u'法定代表人',
            'firm_time':u'成立时间',
            'firm_capital':u'注册资本',
            'firm_deadline':u'营业期限',
            'firm_scope':u'经营范围',
            'firm_authority':u'登记机关',
            'firm_aptime':u'核准时间'}

class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)

        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)

        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):

        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)

        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)

        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):

        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)

        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        # print 'image_batch', image_batch.get_shape()
        return image_batch, label_batch


def build_graph(top_k, charset_size):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob') 
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with tf.device('/gpu:0'):
        # network: conv2d->max_pool2d->conv2d->max_pool2d->conv2d->max_pool2d->conv2d->conv2d->
        # max_pool2d->fully_connected->fully_connected

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training}):
            conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
            max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
            conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
            max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
            conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
            max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
            conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
            conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
            max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

            flatten = slim.flatten(max_pool_4)
            fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(fc1, keep_prob), charset_size, activation_fn=None, scope='fc2')
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
        probabilities = tf.nn.softmax(logits)


        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()

        predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
        accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}



def train():
    print('Begin training')

    train_feeder = DataIterator(data_dir='./en_dataset/train/')
    test_feeder = DataIterator(data_dir='./en_dataset/test/')
    model_name = 'num-en-model'
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph(top_k=1) 
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            i = 0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8,
                             graph['is_training']: True}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0,
                                 graph['is_training']: False}
                    accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']],
                                                           feed_dict=feed_dict)
                    if step > 300:
                        test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    logger.info('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name), global_step=graph['global_step'])
        finally:

            coord.request_stop()
        coord.join(threads)


def validation():
    print('Begin validation')
    test_feeder = DataIterator(data_dir='./en_dataset/test/')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph(top_k=5)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def get_file_list(path):
    list_name=[]
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


def binary_pic(name_list):
    for image in name_list:
        temp_image = cv2.imread(image)
        #print image
        GrayImage=cv2.cvtColor(temp_image,cv2.COLOR_BGR2GRAY) 
        ret,thresh1=cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        single_name = image.split('t/')[1]
        #print single_name
        cv2.imwrite('../data/tmp/'+single_name,thresh1)


def get_label_dict():
    f=open('./chinese_labels','r')
    label_dict = pickle.load(f)
    f.close()
    return label_dict
def get_labels(labels_file):
    labels ={}
    name_label = open(labels_file, 'r')
    for line in name_label:
        tem = line.split('\n')[0]
        value = tem.split(' ')[1]
        chars = tem.split(' ')[0].decode("gbk")
        #print (value,chars)
        labels[int(value)] = chars
    return labels

def inference(name_list):
    print('inference')
    image_set=[]

    for image in name_list:
        temp_image = Image.open(image).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, 64, 64, 1])
        image_set.append(temp_image)
        

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)) as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:       
            saver.restore(sess, ckpt)
        val_list=[]
        idx_list=[]

        for item in image_set:
            temp_image = item
            predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0,
                                                         graph['is_training']: False})
            val_list.append(predict_val)
            idx_list.append(predict_index)
    #return predict_val, predict_index
    return val_list,idx_list



def return_OCR_pro():
    global OCR_pro
    return OCR_pro

def detect(checkpoint, charset_size, image_set):
    #val_dict = {'id':predict_val}
    #index_dict = {'id':predict_index}
    #image_id = 'img-line-char'
    global OCR_pro
    OCR_len = len(image_set)
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))   
    graph = build_graph(top_k=3, charset_size = charset_size)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.latest_checkpoint(checkpoint)
    if ckpt:       
        saver.restore(sess, ckpt)
    val_dict = {}
    idx_dict = {}
    for count, char_info in enumerate(image_set):
        image = char_info['img']
        image_id = char_info['id']

        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],feed_dict={graph['images']: image, graph['keep_prob']: 1.0, graph['is_training']: False})
        val_dict[image_id] = predict_val
        idx_dict[image_id] = predict_index
        OCR_pro = float(count + 1) / float(OCR_len) * 100.0
    sess.close()
    return val_dict, idx_dict

def vector_inner_peoduct(a, b):
    if len(a) != len(b):
        return 0
    else:
        length = len(a)
        res = 0
        for i in range(length):
            res = res + a[i]*b[i]
        return res
    
def vector_length(a):
    res = 0
    length = len(a)
    for i in range(length):
        res = res + a[i]*a[i]
    res = math.sqrt(res)
    return res

def vector_cos(a, b):
    if vector_length(a) != 0 and vector_length(b) != 0:    
        return vector_inner_peoduct(a, b) / (vector_length(a) * vector_length(b)) 
    else:
        return 0
    
def vector_minus(a, b):     #返回向量a-b
    length = len(a)
    res = []
    for i in range(length):
        res.append(a[i] - b[i])
    return res

def web_connect_check():
    import os
    import subprocess
     
    fnull = open(os.devnull, 'w')
    return1 = subprocess.call('ping -c 10 www.baidu.com', shell = True, stdout = fnull, stderr = fnull)
    fnull.close()
    if return1:
        #print 'ping fail'
        return 0
    else:
        #print 'ping ok'
        return 1

def web_spyder(want2search):    #从百度爬取相关结果
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:44.0) Gecko/20100101 Firefox/44.0'}
    payload = {'wd':want2search} 
    url = 'http://www.baidu.com/s'  
    result = []
    try:
        r = requests.get(url, params=payload, headers=headers, timeout=50)
        #print r.url
        # print r.status_code
    except:
        #print "ConnectionError"
        for i in range(5):
            result.append([0])
    else:
        if r.status_code == 200 and r.text != '':
  		
            # print '$$$$$$$$$$$$$$$$$$$$$'
            fp = open('search.html', 'w')
            for line in r.content:
                fp.write(line)
            fp.close()
            
            htmlfile = open('search.html', 'r')
            htmlhandle = htmlfile.read()
            soup = BeautifulSoup(htmlhandle, 'html.parser')
            
            rs_div = soup.find(attrs={"id":"1"})    #爬取百度第1个搜索结果
            if isinstance(rs_div, type(soup.a)):
                result.append(rs_div.text)
            rs_div = soup.find(attrs={"id":"2"})    #爬取百度第2个搜索结果
            if isinstance(rs_div, type(soup.a)):
                result.append(rs_div.text)
            rs_div = soup.find(attrs={"id":"3"})    #爬取百度第3个搜索结果
            if isinstance(rs_div, type(soup.a)):
                result.append(rs_div.text)
            rs_div = soup.find(attrs={"id":"4"})    #爬取百度第4个搜索结果
            if isinstance(rs_div, type(soup.a)):
                result.append(rs_div.text)
            rs_div = soup.find(attrs={"id":"5"})    #爬取百度第5个搜索结果
            if isinstance(rs_div, type(soup.a)):
                result.append(rs_div.text)
            #print rs_div.text
            
            rs_div = soup.find(attrs={"id":"rs"})   #爬取百度的底栏相关搜索
            if isinstance(rs_div, type(soup.a)):
                rs_result_list = rs_div.findAll("a")
                for rs in rs_result_list:
                    result.append(rs.string)
                    #print rs.string
    # print result
    return result
    
def modify_after_search(want2search):
    result = []
    result = web_spyder(want2search)
    
    target = want2search.decode('UTF-8')
    length = len(target)
    temp = []
    for i in range(length):
        if target[i] == u'（':
            temp.append(u'(')
        elif target[i] == u'）':
            temp.append(u')')
        else:
            temp.append(target[i])
    target = ''.join(temp)
    length = len(target)
    
    temp_result = []
    for x in result:
        temp_result.append(x[0:length])
        #print x[0:length]
    result = temp_result
    '''
    try:
        result[0] = result[0][0:length] #根据要搜索的长度截取百度的第一个搜索
    except:
        i = 1
    else:
        try:
            result[1] = result[1][0:length]
        except:
            i = 2
        else:
            try:
                result[2] = result[2][0:length]
            except:
                i = 3
            else:
                try:
                    result[3] = result[3][0:length]
                except:
                    i = 4
                else:
                    try:
                        result[4] = result[4][0:length]
                    except:
                        i = 5
                    else:
                        j = 0
    '''
    
    target_vector = []      #要搜索的词语的特征向量
    for i in range(length):
        target_vector.append(1)
        
    feature_vector = []     #从百度爬下来的结果的特征向量
    tmp = []
    
    for i in range(len(result)):        #计算10个特征向量
        for j in range(len(target)):
            if j >= len(result[i]):
                tmp.append(0)
            else:
                if result[i][j] == target[j]:
                    tmp.append(1)
                else:
                    tmp.append(0)
        feature_vector.append(tmp)
        tmp = []
    
    res_cos = []    #记录每个搜索结果的cos值
    for i in range(len(result)):
        res_cos.append(vector_cos(feature_vector[i], target_vector))
    max_index =  res_cos.index(max(res_cos))
    
    target_modify = target
    difference_vector = vector_minus(target_vector, feature_vector[max_index])
    if vector_length(difference_vector) == 1 and len(result[max_index]) >= length:
        target_modify = result[max_index][0:length]     #短的肯定不会选中
        #print modify
    return target_modify

def error_control(img_info = {}, string = '', mode = ''):
    if string != '':
        if mode == 'firm_num':
             string = string.replace(u'林', 'M')    
             string = string.replace(u'川', 'M') 
             string = string.replace(u'一', 'M') 
             string = string.replace(u'火', 'M') 
             string = string.replace(u'从', 'M') 
             string = string.replace(u'仙', 'M') 
             string = string.replace(u'灿', 'M')
        if mode == 'firm_name':
             string =  modify_after_search(string)
        return string
    for x in img_info.keys():
        if x == 'firm_num':
            img_info[x] = img_info[x].replace(u'林', 'M')    
            img_info[x] = img_info[x].replace(u'川', 'M')
            img_info[x] = img_info[x].replace(u'一', 'M')
            img_info[x] = img_info[x].replace(u'火', 'M')
            img_info[x] = img_info[x].replace(u'从', 'M')
            img_info[x] = img_info[x].replace(u'仙', 'M')
            img_info[x] = img_info[x].replace(u'灿', 'M')
        elif x != 'firm_num':
            img_info[x] = modify_after_search(img_info[x])
    return img_info

def ch_resize(src_img, img_size):
    #resize image and add side
    resize_pic = Image.fromarray((np.ones([img_size, img_size])).astype("uint8"))
    resize_pic.paste(src_img, (resize_pic.width / 2 - src_img.width/2, resize_pic.height / 2 - src_img.height / 2))
    return resize_pic

def en_resize(src_img, img_size):
    resize_pic = Image.fromarray(np.zeros([src_img.height , src_img.width + 8]).astype("uint8"))
    resize_pic.paste(src_img, (resize_pic.width / 2 - src_img.width/2, resize_pic.height / 2 - src_img.height / 2))
    return resize_pic

def std_image(src_img, std_size):
    #transform image to OCR standford size
    std_img = src_img.resize((std_size, std_size), Image.ANTIALIAS)
    std_img = np.asarray(std_img) / 255.0
    std_img = std_img.reshape([-1, std_size, std_size, 1])
    return std_img


def load_charimg(chardir, img_size = 64):
    ch_image_set = []
    en_image_set = []
    std_size = 64
    #image_set=[{'img':char_img, 'id':'img-line-char'} ...]
    for img_name in os.listdir(chardir):
        img_path = os.path.join(chardir, img_name)
        img_id = img_path.split('/')[-1]
        for line_name in os.listdir(img_path):
            line_path = os.path.join(img_path, line_name)
            line_id = line_path.split('/')[-1]
            for char_name in os.listdir(line_path):
                char_info = {}
                char_path = os.path.join(line_path, char_name)
                char_id = img_id + '-' + line_id + '-' + char_path.split('/')[-1].split('.')[0]

                char_img = Image.open(char_path).convert('L')
                width, height = char_img.size
               
                #char_img = char_img.resize((img_size, img_size), Image.ANTIALIAS)
                #char_img = np.asarray(char_img) / 255.0
                #char_img = char_img.reshape([-1, 64, 64, 1])
                #char_info['img'] = char_img
                char_info['id'] = char_id

                if width not in range(int(height * 0.8), int(height * 1.1)):
                       char_img = en_resize(char_img, img_size)
                       #char_img.save(char_path)
                       char_img = std_image(char_img, std_size)
                       char_info['img'] = char_img
                       en_image_set.append(char_info)
                else:
                       char_img = ch_resize(char_img, img_size)
                       #char_img.save(char_path)
                       char_img = std_image(char_img, std_size)
                       char_info['img'] = char_img
                       ch_image_set.append(char_info)
    return ch_image_set, en_image_set

def create_dict(pos_id, pos_dict):
    if pos_id not in pos_dict:
        return {}
    else:
        return pos_dict[pos_id]

def predict_text(final_predict_val, final_predict_index, labels_path, text):
    #val_dict = {'id':predict_val}
    #index_dict = {'id':predict_index}
    #image_id = 'img-line-char'
    labels = get_labels(labels_path)
    for x in final_predict_val.keys():
        img_id = x.split('-')[0]
        line_id = x.split('-')[1]
        char_id = int(x.split('-')[2])

        predict_index = final_predict_index[x][0][0]
        predict_char = labels[int(predict_index)]

        img_dict = create_dict(img_id, text)
        line_dict = create_dict(line_id, img_dict)

        line_dict[char_id] = predict_char
        img_dict[line_id] = line_dict
        text[img_id] = img_dict
        
    return text

def init_return_var():
    global select_pro
    global OCR_pro
    select_pro = 0
    OCR_pro = 0

def return_select_pro():
    global select_pro
    return select_pro

def select_target(text, target, saveDir):
    global select_pro
    select_len = len(text.keys())
    target_string=[]
    for count, img_id in enumerate(sorted(text.keys())):
        img_dict = text[img_id]
        img_info={}
        img_info['img_id'] = img_id
        img_info['img_path'] = os.path.join(saveDir.split('static_file/')[1], img_id + '.jpg')
        img_info['OCRtext'] = {}
        any_result = False
        for x in target:
            img_info['OCRtext'][x]={}
            img_info['OCRtext'][x]['target_text']=OCR_dict[x]
        print '==============' + str(img_id) + '=============='
        for line_id in sorted(img_dict.keys()):
            line_dict = img_dict[line_id]
            #print line_id,
            string = u''
            last_char_id = 0 
            for char_id in sorted(line_dict.keys()):
                if int(char_id) - 2 == int(last_char_id):
                    string = string + ' '
                last_char_id = char_id
                string = string + str(line_dict[char_id])
            #print string
            for x in target:               
                if OCR_dict[x] in string:
                    #print string                   
                    img_info['OCRtext'][x]['result_text'] = string.split(OCR_dict[x])[-1][1:]
                    if x in ['firm_name','firm_num']:
                        img_info['OCRtext'][x]['result_text'] = error_control(string = img_info['OCRtext'][x]['result_text'], mode = x)
                    print img_info['OCRtext'][x]['result_text']
                    any_result = True
            '''
            if target[0] in string:
                img_info['name'] = string.split(target[0])[-1][1:]
                img_info['name'] = error_control(string = img_info['name'], mode = 'name')
                print img_info['name']
            if target[1] in string:
                img_info['num'] = string.split(target[1])[-1][1:]
                img_info['num'] = error_control(string = img_info['num'], mode = 'num')
                print img_info['num']
            
        if 'name' in img_info or 'num' in img_info:
            #img_info = error_control(img_info = img_info)
            target_string.append(img_info)
            '''
        select_pro = float(count + 1) / float(select_len) * 100.0
        if any_result:
            target_string.append(img_info)
    return target_string
            

def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference':
        #label_dict = get_label_dict()
        labels = get_labels('./name_label.names')
        name_list = get_file_list('./split_char/firm_name/1/')

        final_predict_val, final_predict_index = predict(name_list, './cn_sym_model/', 3524)
        final_reco_text =[] 

        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0][0]
            candidate2 = final_predict_index[i][0][1]
            candidate3 = final_predict_index[i][0][2]
            final_reco_text.append(labels[int(candidate1)])
            logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i], 
                labels[int(candidate1)],labels[int(candidate2)],labels[int(candidate3)],final_predict_index[i],final_predict_val[i]))
        print ('=====================OCR RESULT=======================\n')

        for i in final_reco_text:
           print i, 

if __name__ == "__main__":
    tf.app.run()
