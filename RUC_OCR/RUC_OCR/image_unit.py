# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:13:47 2018

@author: PKU
"""

import cv2 as cv
import os
from PIL import Image
import numpy as np
import shutil
import text_split
from sklearn.cluster import KMeans

def convert_png2jpg(pngdir, savedir):
    def convert(rootPath, curName):
        if curName.split('.')[1] in ['jpg', 'jpeg']:
            img = Image.open(os.path.join(rootPath, curName))
            img.save(os.path.join(savedir, curName), 'JPEG')
            
        elif curName.split('.')[1] == 'png':
            img = Image.open(os.path.join(rootPath, curName))
            if len(img.split()) == 4:
                bg = Image.new('RGB', img.size, (255,255,255))
                bg.paste(img, mask=img.split()[3])
                bg = bg.convert('RGB')
                bg.save(os.path.join(savedir, curName.split('.')[0]+'.jpg'),'JPEG')
            else:
                img.save(os.path.join(savedir, curName.split('.')[0]+'.jpg'),'JPEG')

    for x in os.listdir(pngdir):
        if os.path.isdir(os.path.join(pngdir, x)):
            for y in os.listdir(os.path.join(pngdir, x)):
                convert(os.path.join(pngdir, x), y)
        else:
            convert(pngdir, x)
    return

    if not os.path.isdir(pngdir):
        img_path, img_name = os.path.split(pngdir)
        if img_name.split('.')[1]=='png':
            img = Image.open(pngdir)
            if len(img.split()) == 4:
                bg = Image.new('RGB', img.size, (255,255,255))
                bg.paste(img, mask=img.split()[3])
                bg = bg.convert('RGB')
                
                bg.save(os.path.join(savedir, img_name.split('.')[0]+'.jpg'),'JPEG')
            else:
                
                img.save(os.path.join(savedir, img_name.split('.')[0]+'.jpg'),'JPEG')
        else:
            img = Image.open(pngdir)
            
            img.save(os.path.join(savedir, img_name.split('.')[0]+'.jpg'),'JPEG')
        return

              
    for x in os.listdir(pngdir):

        if x.split('.')[1] in ['jpg', 'jpeg']:
            img = Image.open(os.path.join(pngdir,x))
            
            img.save(os.path.join(savedir, x),'JPEG')
            continue
        elif x.split('.')[1] != 'png':
            continue
        img = Image.open(os.path.join(pngdir,x))
        if len(img.split()) == 4:
            bg = Image.new('RGB', img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            bg = bg.convert('RGB')
            #bg = bg.resize((x*512/y,512), Image.ANTIALIAS)
            #save_pic = convert2bin(bg, thresh = 100)
            #save_pic.save(savedir + x.split('.')[0]+'.jpg','JPEG')
            
            bg.save(os.path.join(savedir, x.split('.')[0]+'.jpg'),'JPEG')
        else:
            
            img.save(os.path.join(savedir, x.split('.')[0]+'.jpg'),'JPEG')

def convert2bin(pic, thresh):
    pic = pic.convert("L")
    pic_arr = np.array(pic)
    bin_arr = np.where(pic_arr < thresh, 0, 255)
    pic = Image.fromarray(bin_arr.astype("uint8"))
    return pic


def add_side(savedir):
    for x in os.listdir(savedir):
        if x.split('.')[1]!='jpg':
            continue
        img = Image.open(os.path.join(savedir, x))
        width, height = img.size
        #save_pic = Image.fromarray((255*np.ones([height + 40 , width + 40])).astype("uint8"))
        #save_pic = save_pic.convert('RGB')
        #print img.size, save_pic.size
        #save_pic.paste(img, (save_pic.width/2 - width/2, save_pic.height/2 - height/2))
        if width > height * 1.2:
            save_pic = Image.fromarray((255*np.ones([height + 40 , width + 40])).astype("uint8"))
            save_pic = save_pic.convert('RGB')
            #print img.size, save_pic.size
            save_pic.paste(img, (save_pic.width/2 - width/2, save_pic.height/2 - height/2))
            save_pic = resize(save_pic, [1300,1300])
        else:
            save_pic = Image.fromarray((255*np.ones([height + 50 , width + 50])).astype("uint8"))
            save_pic = save_pic.convert('RGB')
            #print img.size, save_pic.size
            save_pic.paste(img, (save_pic.width/2 - width/2, save_pic.height/2 - height/2))
            #save_pic = resize(save_pic, [1300,1300])
        #save_pic = resize(save_pic, [1300,1300])
        save_pic.save(os.path.join(savedir, x.split('.')[0]+'.jpg'),'JPEG')

def ssd_crop(img_path, results, savedir):
    num_dir = os.path.join(savedir, 'firm_num')
    name_dir = os.path.join(savedir, 'firm_name')
    if not os.path.exists(num_dir):
        os.mkdir(num_dir)
    if not os.path.exists(name_dir):
        os.mkdir(name_dir)
    num_save_path = os.path.join(num_dir, img_path.split('/')[-1].split('.')[0])
    name_save_path= os.path.join(name_dir, img_path.split('/')[-1].split('.')[0])
    if not os.path.exists(num_save_path):
        os.mkdir(num_save_path)
    if not os.path.exists(name_save_path):
        os.mkdir(name_save_path)

    img = Image.open(img_path)
    width, height = img.size
    point ={'firm_num':[], 'firm_name':[]}
    firm_num = []
    firm_name = []
    for res in results:
        if res[6] == 'firm_num':
            firm_num.append(res[:4])
        else:
            firm_name.append(res[:4])
    firm_num = sorted(firm_num)
    firm_name = sorted(firm_name)
    for x in firm_name:
        if x[1] <= firm_num[0][3]:
          firm_name.remove(x)
    if len(firm_num) != 0:
        img_num = img.crop((0, firm_num[0][1] * height - 5, 
                        width, firm_num[0][3] * height + 5))
        save_pic = Image.fromarray((255*np.ones([img_num.size[1] + 10 , img_num.size[0] + 10])).astype("uint8"))
        save_pic = save_pic.convert('RGB')
        save_pic.paste(img_num, (save_pic.width/2 - img_num.size[0]/2, save_pic.height/2 - img_num.size[1]/2))
        #img_num = img_num.convert('RGB')
        save_pic.save(os.path.join(num_save_path, 'num.jpg'))
    if len(firm_name) != 0:
        if(firm_name[0][1]  > 2 * firm_num[0][3] - firm_num[0][1]):
            line_height = (firm_num[0][3] - firm_num[0][1]) * height
            img_name = img.crop((0, firm_num[0][3] * height + line_height / 2, 
                        width, firm_num[0][3] * height + line_height * 1.5))
        else:
            img_name = img.crop((0, firm_name[0][1] * height - 5, 
                        width, firm_name[0][3] * height + 5))
        save_pic = Image.fromarray((255*np.ones([img_name.size[1] + 10 , img_name.size[0] + 10])).astype("uint8"))
        save_pic = save_pic.convert('RGB')
        save_pic.paste(img_name, (save_pic.width/2 - img_name.size[0]/2, save_pic.height/2 - img_name.size[1]/2))
        #img_name = img_name.convert('RGB')
        save_pic.save(os.path.join(name_save_path, 'name.jpg'))
    else:
        line_height = (firm_num[0][3] - firm_num[0][1]) * height
        img_name = img.crop((0, firm_num[0][3] * height + line_height / 2, 
                        width, firm_num[0][3] * height + line_height * 1.5))
        save_pic = Image.fromarray((255*np.ones([img_name.size[1] + 10 , img_name.size[0] + 10])).astype("uint8"))
        save_pic = save_pic.convert('RGB')
        save_pic.paste(img_name, (save_pic.width/2 - img_name.size[0]/2, save_pic.height/2 - img_name.size[1]/2))
        #img_name = img_name.convert('RGB')
        save_pic.save(os.path.join(name_save_path, 'name.jpg'))

def frcnn_crop(img_path, results, splitdir, chardir):
    img = Image.open(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]
    img_dir = os.path.join(splitdir, img_name)
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(img_dir)
    width, height = img.size
    #frcnn detect and crop each result about 'ch_char'
    for index, result in enumerate(results):
        '''
        if result[6] == 'firm_name': #把识别为企业名称、注册号的区域分别保存为图片
            crop_ax = (0, result[1] - 10, width, result[3] + 10) #裁剪区域比frcnn定位的位置略大
            save_pic = img.crop(crop_ax)
            save_name = os.path.join(img_dir, str(index) + '_name.jpg')
            save_pic.save(save_name)
        if result[6] == 'firm_num':
            crop_ax = (0, result[1] - 10, width, result[3] + 10)
            save_pic = img.crop(crop_ax)
            save_name = os.path.join(img_dir, str(index) + '_num.jpg')
            save_pic.save(save_name)
        '''
        crop_ax = (0, result[1] - 15, width, result[3] + 15)
        save_pic = img.crop(crop_ax)
        save_name = os.path.join(img_dir, str(index) + '_' + result[6] + '.jpg')
        save_pic.save(save_name)
    #crop each result(line) to single char
    char_dir = os.path.join(chardir, img_name)
    if os.path.exists(char_dir):
        shutil.rmtree(char_dir)
    os.mkdir(char_dir)
    for x in os.listdir(img_dir):
        line_num = x.split('.')[0]
        line_dir = os.path.join(chardir, img_name, line_num)
        line_img_path = os.path.join(img_dir, x)
        if not os.path.exists(line_dir):
            os.mkdir(line_dir)
        str_ = line_img_path
        if 'firm_num' in str_:
            text_split.cut_by_char_weight(line_img_path, line_dir, (48, 48))
        elif 'firm_name' in str_:
            text_split.cut_by_kmeans(line_img_path, line_dir, (48, 48))
        else:
            text_split.cut_by_kmeans_new(line_img_path, line_dir, (48, 48), 3)
        '''
        if str_[-5] == 'e':
            text_split.cut_by_kmeans(line_img_path, line_dir, (48, 48))
        elif str_[-5] == 'm':
            text_split.cut_by_char_weight(line_img_path, line_dir, (48, 48)) #按字符宽度切割出单独的文字图片
        '''
    
def resize(src_img, resize_size):
    width, height = src_img.size
    if height >= width:
        resize_height = resize_size[1]
        resize_width = int(width * 1.0 / height * 1.0 * resize_height * 1.0)
    else:
        resize_width = resize_size[0]
        resize_height = int(height * 1.0 / width * 1.0 * resize_width * 1.0)
    return src_img.resize((resize_width, resize_height), Image.ANTIALIAS) 
    


def cn_handle(char_path):
  id = 0
  img_list = {}
  for x in sorted(os.listdir(char_path)):
    img = Image.open(os.path.join(char_path, x))
    if img.width >= 1.8 * img.height:
      crop_wid = img.width / 2
      img_crop = img.crop((0,0,crop_wid,img.height))
      img_list[id] = np.array(img_crop)
      id = id + 1
      img_crop = img.crop((crop_wid,0,img.width,img.height))
      img_list[id] = np.array(img_crop)
    else:
      img_list[id] = np.array(img)
    img.close()
    id = id + 1
  new_list = []
  flag = False
  for x in range(id-1):
    img = Image.fromarray(img_list[x])
    if 1.5 * img.width <= img.height:
      img_next = Image.fromarray(img_list[x+1])
      if 1.5 * img_next.width <= img_next.height:
        new = Image.new('L',(img.width + img_next.width, img.height))
        new.paste(img, (0,0,img.width,img.height))
        new.paste(img_next, (img.width, 0, new.width, img.height))
        new_list.append(new)
        flag = True
      else:
        if flag == False:
          new_list.append(img)
        flag = False
    else:
        new_list.append(img)
        flag = False
  if len(img_list) != id+1:
    new_list.append(Image.fromarray(img_list[id-1]))
  id = 0
  shutil.rmtree(char_path)
  os.mkdir(char_path)
  for x in new_list:
    save_pic = Image.fromarray((np.ones([64, 64])).astype("uint8"))
    save_pic.paste(x, (32-x.width/2, 32-x.height/2))
    save_pic.save(os.path.join(char_path, "%06d.jpg"%id))
    id = id +1

def num_handle(char_path):
    id = 0
    img_list = {}
    for x in sorted(os.listdir(char_path))[4:]:
        '''
        img = Image.open(os.path.join(char_path, x))
        width, height = img.size
        if width * 1.2 <= height:
            save_pic = Image.fromarray(np.zeros([height , width + 8]).astype("uint8"))
            save_pic.convert('L')
            #print img.size, save_pic.size
            save_pic.paste(img, (save_pic.width/2 - width/2, save_pic.height/2 - height/2))
            save_pic = save_pic.resize((30,64))
            img_list[id] = np.array(save_pic)
            id = id + 1
        img.close()
        '''
        img = Image.open(os.path.join(char_path, x))
        if img.width >= img.height:
            crop_wid = img.width / 2
            img_crop = img.crop((0,0,crop_wid,img.height))
            save_pic = Image.fromarray(np.zeros([img_crop.height , img_crop.width + 8]).astype("uint8"))
            save_pic.convert('L')
            #print img.size, save_pic.size
            save_pic.paste(img_crop, (save_pic.width/2 - img_crop.width/2, save_pic.height/2 - img_crop.height/2))
            save_pic = save_pic.resize((30,64))
            img_list[id] = np.array(save_pic)

            id = id + 1
            img_crop = img.crop((crop_wid,0,img.width,img.height))
            save_pic = Image.fromarray(np.zeros([img_crop.height , img_crop.width + 8]).astype("uint8"))
            save_pic.convert('L')
            #print img.size, save_pic.size
            save_pic.paste(img_crop, (save_pic.width/2 - img_crop.width/2, save_pic.height/2 - img_crop.height/2))
            save_pic = save_pic.resize((30,64))
            img_list[id] = np.array(save_pic)
        else:
            save_pic = Image.fromarray(np.zeros([img.height , img.width + 8]).astype("uint8"))
            save_pic.convert('L')
            #print img.size, save_pic.size
            save_pic.paste(img, (save_pic.width/2 - img.width/2, save_pic.height/2 - img.height/2))
            save_pic = save_pic.resize((30,64))
            img_list[id] = np.array(save_pic)
        img.close()
        id = id + 1
    shutil.rmtree(char_path)
    os.mkdir(char_path)
    for x in img_list.keys():
        img = Image.fromarray(img_list[x])
        img.save(os.path.join(char_path, "%06d.jpg"%x))
   
