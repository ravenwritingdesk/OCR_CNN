# -*- coding: utf-8 -*-
import image_unit
import OCR_unit
import ssd_detect as ssd
import argparse
import os
import shutil
#import Chinese_OCR as OCR
import time
import frcnn


parser = argparse.ArgumentParser()
parser.add_argument('--pngdir', default='/home/zhangzm/win3/ch_rec_ssd/dataset/test/', help='pngdir')
parser.add_argument('--savedir', default='/home/zhangzm/win3/ch_rec_ssd/test/image/', help='savedir')
parser.add_argument('--splitdir', default='/home/zhangzm/win3/ch_rec_ssd/test/split_img/', help='savedir')
parser.add_argument('--chardir', default='/home/zhangzm/win3/ch_rec_ssd/test/split_char/', help='chardir')
parser.add_argument('--ch_model', default='/home/zhangzm/win3/ch_rec_ssd/model/cn_model/', help='ch_model')
parser.add_argument('--en_model', default='/home/zhangzm/win3/ch_rec_ssd/model/en_model/', help='en_model')
parser.add_argument('--ocr_model', default='/home/zhangzm/win3/ch_rec_ssd/model/all_char_model/', help='ocr_model')
parser.add_argument('--ch_label', default='/home/zhangzm/win3/ch_rec_ssd/test/ch_label.names', help='ch_label')
parser.add_argument('--en_label', default='/home/zhangzm/win3/ch_rec_ssd/test/en_label.names', help='en_label')
parser.add_argument('--ocr_label', default='/home/zhangzm/win3/ch_rec_ssd/test/all_label.names', help='ocr_label')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--labelmap_file',
                        default='/home/zhangzm/win3/ch_rec_ssd/dataset/character_label.prototxt')
parser.add_argument('--model_def',
                        default='/home/zhangzm/win3/ch_rec_ssd/model/caffemodel/firm_300x300/log/deploy.prototxt')
parser.add_argument('--image_resize', default=300, type=int)
parser.add_argument('--model_weights',
                        default='/home/zhangzm/win3/ch_rec_ssd/model/caffemodel/firm_300x300/snapshot/character_model_firm_300x300_iter_80000.caffemodel')
parser.add_argument('--mode', default='OCR', help='mode has {segment, OCR}')

arg = parser.parse_args()
pngdir = arg.pngdir
savedir = arg.savedir
gpu_id = arg.gpu_id
labelmap_file = arg.labelmap_file
model_def = arg.model_def
model_weights = arg.model_weights
image_resize = arg.image_resize
splitdir = arg.splitdir
chardir = arg.chardir
ch_model = arg.ch_model
en_model = arg.en_model
ocr_model = arg.ocr_model
ch_label = arg.ch_label
en_label = arg.en_label
ocr_label = arg.ocr_label
mode = arg.mode

def segment():
    image_unit.convert_png2jpg(pngdir, savedir)
    image_unit.add_side(savedir)

    detection = ssd.CaffeDetection(gpu_id, model_def, model_weights, image_resize, labelmap_file)

    #ssd_detect predict
    for x in os.listdir(savedir):
        result = detection.detect(savedir + x) 
        #xmin,ymin,xmax,ymax = result[0]*width, result[1]*height, result[2]*width, result[3]*height
        #object_num = result[4]   prob = result[5]   object_name = result[6]
        if len(result) != 0:
            image_unit.ssd_crop(savedir + x, result, splitdir)

    #segmentation char
    if os.path.exists(chardir):
        shutil.rmtree(chardir)
    os.mkdir(chardir)
    os.mkdir(os.path.join(chardir, 'firm_num'))
    os.mkdir(os.path.join(chardir, 'firm_name'))

    for x in os.listdir(os.path.join(splitdir, 'firm_num')):
        id = x
        img_path = os.path.join(splitdir, 'firm_num', id, 'num.jpg')
        char_path = os.path.join(chardir, 'firm_num', id)
        if os.path.exists(img_path):
            os.mkdir(char_path)
        os.system('/home/zhangzm/win3/ch_rec_ssd/test/text_split ' + img_path + ' ' + char_path)
        if os.path.exists(char_path):
            image_unit.num_handle(char_path + '/')

    for x in os.listdir(os.path.join(splitdir, 'firm_name')):
        id = x
        img_path = os.path.join(splitdir, 'firm_name', id, 'name.jpg')
        char_path = os.path.join(chardir, 'firm_name', id)
        if os.path.exists(img_path):
            os.mkdir(char_path)
        os.system('/home/zhangzm/win3/ch_rec_ssd/test/text_split ' + img_path + r' ' + char_path)
        if os.path.exists(char_path):
            image_unit.cn_handle(char_path + '/')

def segment_frcnn():
    image_unit.convert_png2jpg(pngdir, savedir)
    image_unit.add_side(savedir)

    #frcnn_detect predict
    sess, net = frcnn.init(0,0)
    for x in os.listdir(savedir):
        results = frcnn.detect(sess, net, savedir + x) 
        #xmin,ymin,xmax,ymax = result[0]*width, result[1]*height, result[2]*width, result[3]*height
        #object_num = result[4]   prob = result[5]   object_name = result[6]
        if len(results) != 0:
            image_unit.frcnn_crop(savedir + x, results, splitdir, chardir)
    sess.close()

def choose_OCR_model():
    return
'''
def ocr():
    #OCR
    num_val = {}
    num_index = {}
    for x in os.listdir(os.path.join(chardir, 'firm_num')):
        id = x
        char_path = os.path.join(chardir, 'firm_num', id)
        name_list = OCR.get_file_list(char_path)
        num_val[id], num_index[id] = OCR.predict(name_list, en_model, 36)
 
    ch_val = {}
    ch_index = {}
    for x in os.listdir(os.path.join(chardir, 'firm_name')):
        id = x
        char_path = os.path.join(chardir, 'firm_name', id)
        name_list = OCR.get_file_list(char_path)
        ch_val[id], ch_index[id] = OCR.predict(name_list, ch_model, 3514)

    #show result
    enLabel = OCR.get_labels(en_label)
    chLabel = OCR.get_labels(ch_label)

    def print_res(final_predict_val, final_predict_index, labels, isen = True):
        final=[]
        for x in range(len(final_predict_val)):
            candidate1 = final_predict_index[x][0][0]
            candidate2 = final_predict_index[x][0][1]
            candidate3 = final_predict_index[x][0][2]
            if isen == True and candidate1 == 60:
                candidate = 2
            final.append(labels[int(candidate1)])
        for x in final:
            print x,
        print


    for x in ch_val.keys():
        print x,
        print_res(ch_val[x], ch_index[x], chLabel, False)
  
    for x in num_val.keys():
        print x,
        print_res(num_val[x], num_index[x], enLabel)


if(mode == 'OCR'):
  ocr()
if(mode == 'segment'):
  segment()
'''
#segment_frcnn()

ch_image_set, en_image_set = OCR_unit.load_charimg(chardir, img_size = 64)
text = {}
val_dict, idx_dict = OCR_unit.detect(ocr_model, 3537, ch_image_set)
text = OCR_unit.predict_text(val_dict, idx_dict, ocr_label, text)
val_dict, idx_dict = OCR_unit.detect(ocr_model, 3537, en_image_set)
text = OCR_unit.predict_text(val_dict, idx_dict, ocr_label, text)
OCR_unit.select_target(text, [u'名称',u'注册号'])
'''
for img_id in sorted(text.keys()):
    img_dict = text[img_id]
    print '==============' + str(img_id) + '=============='
    for line_id in sorted(img_dict.keys()):
        line_dict = img_dict[line_id]
        print line_id, 
        for char_id in sorted(line_dict.keys()):
            print line_dict[char_id],
        print
'''
