# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:05:27 2018

@author: PKU
"""

from PIL import Image
from PIL import ImageDraw
import numpy as np
import image_unit
import cv2 as cv    
from sklearn.cluster import KMeans
min_thresh = 2
min_range = 5

def vertical(img_arr):
    #pixdata = img.load()
    h,w = img_arr.shape
    ver_list = []
    for x in range(w):
        ver_list.append(h - np.count_nonzero(img_arr[:, x]))
    return ver_list

def horizon(img_arr):
    #pixdata = img.load()
    h,w = img_arr.shape
    hor_list = []
    for x in range(h):
        hor_list.append(w - np.count_nonzero(img_arr[x, :]))
    return hor_list

def cut_line(horz, pic):
    begin, end = 0, 0
    w, h = pic.size
    cuts=[]
    for i,count in enumerate(horz):
       if count >= min_thresh and begin == 0:
            begin = i
       elif count >= min_thresh and begin != 0:
            continue
       elif count <= min_thresh and begin != 0:
            end = i
            #print (begin, end), count
            if end - begin >= 2:
                cuts.append((end - begin, begin, end))
                begin = 0
                end = 0
                continue
       elif count <= min_thresh or begin == 0:
            continue
    cuts = sorted(cuts, reverse=True)
    if len(cuts) == 0:
        return 0, False
    else:
        if len(cuts) > 1 and cuts[1][0] in range(int(cuts[0][0] * 0.8), cuts[0][0]):
            return 0, False
        else:
            crop_ax = (0, cuts[0][1], w, cuts[0][2])
    img_arr = np.array(pic.crop(crop_ax))
    return img_arr, True

def simple_cut(vert):
    begin, end = 0,0
    cuts = []
    for i,count in enumerate(vert):
       if count >= min_thresh and begin == 0:
            begin = i
       elif count >= min_thresh and begin != 0:
            continue
       elif count <= min_thresh and begin != 0:
            end = i
            #print (begin, end), count
            if end - begin >= min_range:
                cuts.append((begin, end))
                
                begin = 0
                end = 0
                continue
       elif count <= min_thresh or begin == 0:
            continue
    return cuts

def first_cut(vert):
    begin, end = 0,0
    cuts = []
    char_wid =[]
    aver_wid = 0
    diff_count = 0
    for i,count in enumerate(vert):
        '''
        if flag == False and count > 5:
            l = i
            flag = True
        if flag and count > 5:
            r = i-1
            flag = False
            cuts.append((l,r))
        '''
        if count >= min_thresh and begin == 0:
            begin = i
        elif count >= min_thresh and begin != 0:
            continue
        elif count <= min_thresh and begin != 0:
            end = i
            begin_tem = begin
            end_tem = end
            #print (begin, end), count, aver_wid
            if (aver_wid == 0 or (end - begin) in range(aver_wid-2, aver_wid+2)):
                cuts.append((begin, end))
                char_wid.append(end - begin)
                
                aver_wid = int(sum(char_wid)/len(char_wid))
                min_range = aver_wid / 4
                begin = 0
                end = 0
                continue
            if end - begin > aver_wid * 1.1 :
                begin = 0
                end = 0
                continue
            if end - begin < 4:
                
                begin = end
                end = 0
                if sum(vert[begin_tem:end_tem])/(end_tem-begin_tem) != 0:
                    #print begin_tem, end_tem, sum(vert[begin_tem:end_tem])/(end_tem-begin_tem)
                    cuts.append((begin_tem, end_tem))
                    begin = 0
                    end = 0
        elif count <= min_thresh or begin == 0:
            continue
    return cuts
def second_cut(vert,cuts,begin_point,end_point):
    begin, end = 0,0
    for i,count in enumerate(vert[begin_point:end_point]):
       if count > min_thresh and begin == 0:
            begin = i + begin_point
       elif count > min_thresh and begin != 0:
            continue
       elif count < min_thresh and begin != 0:
            end = i + begin_point
            #print (begin, end), count
            if end - begin >= 2:
                cuts.append((begin, end))
                
                begin = 0
                end = 0
                continue
       elif count <= min_thresh or begin == 0:
            continue
    return cuts

def width_cut(cuts, aver_width, char_height): #根据字符宽度确定最终要切割的字符
    new_cuts = []
    for i, current in enumerate(cuts):
        
        #current width < aver_width
        current_width = current[1] - current[0]
        #print str(current_width) + '->',
        new_current = current
        if current_width < aver_width * 0.9: #如果当前字符宽度 < 平均字符宽度 * 0.9
                                             #认为可能是左右结构的字，或者为左中右结构的字
            for j, next in enumerate(cuts[i + 1:]):
                next_width = next[1] - next[0]
                #print next_width,
                if next[0] - new_current[1]  >= aver_width * 0.3: #如果下一个字符的和当前字符的间距 >= 平均宽度 * 0.3
                    #print 'exit1'                                #可能这两个字不是左右结构
                    break
                if current_width + next_width >= aver_width * 1.1:#如果下一个字符的宽度 + 当前字符宽度 >= 平均字符宽度 * 1.1
                    #print 'exit2'                                #也不符合左右结构字的特点
                    break
                else:
                    current_width = current_width + next_width
                    new_current = (new_current[0], next[1]) #合并两个被切割的部分
            if len(new_cuts) == 0 or new_current[0] not in range(new_cuts[-1][0], new_cuts[-1][1]):
                new_cuts.append(new_current)
        #current width > aver_width
        elif current_width > aver_width * 1.5 and current_width > char_height: #如果当前字符宽度 > 平均字符宽度 * 1.5 and 字符宽度 > 字符高度
            new_current = (current[0], current[0] + current_width / 2)         #可能是两个字叠加在一起
            new_cuts.append(new_current)
            new_current = (current[0] + current_width / 2, current[1])         #把此一分为二
            new_cuts.append(new_current)
        else:
            new_cuts.append(new_current)
        #print
    return new_cuts
            
            

def binarizing(img, threshold):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


def get_char_width(cuts): #获取字符的平均宽度，根据每个被分割下来的部分的宽度
    char_widths = {}      #上下浮动2个像素点，统计每个宽度所占的比例
    width_dis = {}        #获取占比最大的宽度，即为平均宽度
    for x in cuts:
        w = x[1] - x[0]
        if w not in char_widths.keys():
            char_widths[w] = 1
        else:
            char_widths[w] = char_widths[w] + 1
    #print char_widths
    for width in char_widths:
        width_count = char_widths[width]
        if width not in width_dis:
            width_dis[width] = 0
        for x in range(width-2, width+3):
            if x in char_widths:
                width_dis[width] = width_dis[width] + char_widths[x]
    char_widths = sorted(char_widths.items(), key = lambda x: x[1])
    #print width_dis
    if char_widths[0][1] == char_widths[-1][1]:
        return np.mean(np.array(char_widths), axis = 0)[0]
    char_width = char_widths[-1][0]
    width_dis = sorted(width_dis.items(), key = lambda x: x[1])
    char_width_by_dis = width_dis[-1][0]
    return char_width_by_dis
    
    

def get_max_gap(cuts):
    gap = 0
    pos = 0
    for x in range(len(cuts[:-1])):
        if cuts[x+1][0] - cuts[x][1] > gap:
            gap = cuts[x+1][0] - cuts[x][1]
            pos = x
    return (gap, pos+1)
    
        
def OTSU_enhance(img_gray, th_begin=0, th_end=256, th_step=1):  
    max_g = 0  
    suitable_th = 0  
    for threshold in xrange(th_begin, th_end, th_step):  
        bin_img = img_gray > threshold  
        bin_img_inv = img_gray <= threshold  
        fore_pix = np.sum(bin_img)  
        back_pix = np.sum(bin_img_inv)  
        if 0 == fore_pix:  
            break  
        if 0 == back_pix:  
            continue  

        w0 = float(fore_pix) / img_gray.size  
        u0 = float(np.sum(img_gray * bin_img)) / fore_pix  
        w1 = float(back_pix) / img_gray.size  
        u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix  
        # intra-class variance  
        g = w0 * w1 * (u0 - u1) * (u0 - u1)  
        if g > max_g:  
            max_g = g  
            suitable_th = threshold  
    return suitable_th  


def cut_single_char(pic_path, save_path, save_size):
    src_pic = Image.open(pic_path).convert('L')
    src_arr = np.array(src_pic)
    threshold = OTSU_enhance(src_arr)
    bin_arr = np.where(src_arr < 100, 0, 255)

    horz = horizon(bin_arr)
    line_arr, flag = cut_line(horz, src_pic)
    if flag == False:
        return flag
    line_arr_save = np.where(line_arr < threshold, 0, 255)
    line_arr = np.where(line_arr < 100, 0, 255)
    vert = vertical(line_arr)
    cut = first_cut(vert)
    for x in range(0, len(cut) - 1):
        if cut[x + 1][0] - cut[x][1] >= 5:
            cut = second_cut(vert, cut, cut[x][1], cut[x + 1][0])
    #second cut
    cut = sorted(cut)
    cut = second_cut(vert, cut, cut[-1][1], len(vert)-1)
    cut = sorted(cut)
    line_img = Image.fromarray((255 - line_arr_save).astype("uint8"))
    width, height = line_img.size
    for x in range(len(cut)):
        ax = (cut[x][0] - 1, 0, cut[x][1] + 1, height)
        temp = line_img.crop(ax)
        temp = image_unit.resize(temp, save_size)
        temp.save('{}/{}.jpg'.format(save_path, x))
    return flag

def cut_by_char_weight(pic_path, save_path, save_size):
    src_pic = Image.open(pic_path).convert('L') #先把图片转化为灰度
    src_arr = np.array(src_pic)
    threshold = OTSU_enhance(src_arr) * 0.9 #用大津阈值
    bin_arr = np.where(src_arr < threshold, 0, 255) #二值化图片

    horz = horizon(bin_arr) #获取到该行的 水平 方向的投影
    line_arr, flag = cut_line(horz, src_pic) #把文字（行）所在的位置切割下来
    if flag == False:
        return flag
    line_arr = np.where(line_arr < threshold, 0, 255)
    line_img = Image.fromarray((255 - line_arr).astype("uint8"))
    width, height = line_img.size

    vert = vertical(line_arr) #获取到该行的 垂直 方向的投影
    cut = simple_cut(vert) #先进行简单的文字分割（即有空隙就切割）
    if len(cut)==0:
        return False
    max_gap = get_max_gap(cut) #获取这些被切割体之前最大的距离
    if max_gap[1] == 0: #根据两个部分之前空隙最大，分为2段，每段分别进行根据字符宽度的分割
        return False
    #max_gap = (gap, pos)
    aver_width1 = get_char_width(cut[:max_gap[1]]) #获取字符宽度
    #print aver_width1
    new_cut1 = width_cut(cut[:max_gap[1]], int(aver_width1), height) #进行根据字符宽度的分割，（此处我认为要优化）
    for x in new_cut1:
        w = x[1] - x[0]
        if w >= aver_width1 * 1.2 and w >= height * 1.25:
            new_cut1 = second_cut(vert, new_cut1, x[0], x[1])
    new_cut1 = sorted(new_cut1) #再用简单分割对宽度分割没能处理的部分，进行二次处理
        

    aver_width2 = get_char_width(cut[max_gap[1]:])
    #print aver_width2
    new_cut2 = width_cut(cut[max_gap[1]:], int(aver_width2), height)
    for x in new_cut2:
        w = x[1] - x[0]
        if w >= aver_width2 * 1.2 and w >= height * 1.25:
            new_cut2 = second_cut(vert, new_cut2, x[0], x[1])
    new_cut2 = sorted(new_cut2)
    new_cut = new_cut1 + new_cut2
    new_cut = sorted(new_cut)
    

    for x in range(len(new_cut)):
        ax = (new_cut[x][0] - 1, 0, new_cut[x][1] + 1, height)
        temp = line_img.crop(ax)
        temp = image_unit.resize(temp, save_size)
        temp.save('{}/{}.jpg'.format(save_path, x))
    return flag
    
def isnormal_width(w_judge, w_normal_min, w_normal_max):
    if w_judge < w_normal_min - 1:
        return -1
    elif w_judge > w_normal_max + 1:
        return 1
    else:
        return 0

def cut_by_kmeans(pic_path, save_path, save_size):
    src_pic = Image.open(pic_path).convert('L') #先把图片转化为灰度
    src_arr = np.array(src_pic)
    threshold = OTSU_enhance(src_arr) * 0.9 #用大津阈值
    bin_arr = np.where(src_arr < threshold, 0, 255) #二值化图片
    
    horz = horizon(bin_arr) #获取到该行的 水平 方向的投影
    line_arr, flag = cut_line(horz, src_pic) #把文字（行）所在的位置切割下来
    if flag == False:
        return flag
    line_arr = np.where(line_arr < threshold, 0, 255)
    line_img = Image.fromarray((255 - line_arr).astype("uint8"))
    width, height = line_img.size
    
    vert = vertical(line_arr) #获取到该行的 垂直 方向的投影
    cut = simple_cut(vert) #先进行简单的文字分割（即有空隙就切割）
    
    #cv.line(img,(x1,y1), (x2,y2), (0,0,255),2)    
    width_data = []
    width_data_TooBig = []
    width_data_withoutTooBig = []
    for i in range(len(cut)):
        tmp = (cut[i][1] - cut[i][0], 0) 
        if tmp[0] > height * 1.8:     #比这一行的高度大两倍的肯定是连在一起的
            temp = (tmp[0], i)
            width_data_TooBig.append(temp)
        else:
            width_data_withoutTooBig.append(tmp)
        width_data.append(tmp)
    kmeans = KMeans(n_clusters=2).fit(width_data_withoutTooBig)
    #print "聚簇中心点:", kmeans.cluster_centers_
    #print 'label:', kmeans.labels_ 
    #print '方差:', kmeans.inertia_
    
    label_tmp = kmeans.labels_
    label = []
    j = 0
    k = 0
    for i in range(len(width_data)):        #将label整理，2代表大于一个字的
        if j != len(width_data_TooBig) and k != len(label_tmp):
            if i == width_data_TooBig[j][1]:
                label.append(2)
                j = j + 1
            else:
                label.append(label_tmp[k])
                k = k + 1
        elif j == len(width_data_TooBig) and k != len(label_tmp):
            label.append(label_tmp[k])
            k = k + 1
        elif j != len(width_data_TooBig) and k == len(label_tmp):
            label.append(2)
            j = j + 1
            
    label0_example = 0
    label1_example = 0
    for i in range(len(width_data)):
        if label[i] == 0:
            label0_example = width_data[i][0]
        elif label[i] == 1:
            label1_example = width_data[i][0]
    if label0_example > label1_example:    #找到正常字符宽度的label(宽度大的，防止切得太碎导致字符宽度错误)
        label_width_normal = 0
    else:
        label_width_normal = 1
    label_width_small = 1 - label_width_normal
        
    cluster_center = []
    cluster_center.append(kmeans.cluster_centers_[0][0])
    cluster_center.append(kmeans.cluster_centers_[1][0])
    for i in range(len(width_data)):
        if label[i] == label_width_normal and width_data[i][0] > cluster_center[label_width_normal] * 4 / 3: #5/4可优化，选这个的理由是洲
            label[i] = 2
            temp = (width_data[i][0], i)
            width_data_TooBig.append(temp)        
    max_gap = get_max_gap(cut) #获取这些被切割体之前最大的距离 找':', label = 3
    for i in range(len(label)):
        if i == max_gap[1]:
            label[i] = 3
    
    width_normal_data = []      #存正常字符宽度
    width_data_TooSmall = []
    for i in range(len(width_data)):
        if label[i] == label_width_normal:
            width_normal_data.append(width_data[i][0])
        elif label[i] != label_width_normal and label[i] != 2 and label[i] != 3:  #切得太碎的
            box=(cut[i][0],0,cut[i][1],height)
            region=line_img.crop(box) #此时，region是一个新的图像对象。
            region_arr = 255 - np.array(region)
            region = Image.fromarray(region_arr.astype("uint8"))
            name = "single"+str(i)+".jpg"
            #region.save(name)
            tmp = (width_data[i][0], i)
            width_data_TooSmall.append(tmp)
    width_normal_max = max(width_normal_data)
    width_normal_min = min(width_normal_data)       #得到正常字符宽度的上下限
    
    if len(width_data_TooBig) != 0:   
        for i in range(len(width_data_TooBig)):
            index = width_data_TooBig[i][1]
            mid = (cut[index][0] + cut[index][1]) / 2
            tmp1 = (cut[index][0], int(mid))
            tmp2 = (int(mid)+1, cut[index][1])
            del cut[index]
            cut.insert(index, tmp2)
            cut.insert(index, tmp1)
            del width_data[index]
            tmp1 = (tmp1[1] - tmp1[0], index)
            tmp2 = (tmp2[1] - tmp2[0], index+1)
            width_data.insert(index, tmp2)
            width_data.insert(index, tmp1)
            label[index] = label_width_normal
            label.insert(index, label_width_normal)
            
            
    if len(width_data_TooSmall) != 0:               #除':'以外有小字符,先找'('、')'label = 4                             
        for i in range(len(width_data_TooSmall)):
            index = width_data_TooSmall[i][1]
            border_left = cut[index][0] + 1
            border_right = cut[index][1]
            RoI_data = line_arr[:,border_left:border_right]
            
            #RoI_data = np.where(RoI_data < threshold, 0, 1)
            horz = horizon(RoI_data)
        
            up_down = np.sum(np.abs(RoI_data - RoI_data[::-1]))
            left_right = np.sum(np.abs(RoI_data - RoI_data[:,::-1]))
            vert = vertical(RoI_data)
        
            if up_down <= left_right * 0.6 and np.array(vert).var() < len(vert) * 2:    
                #print i, up_down, left_right,
                #print vert, np.array(vert).var()
                label[index] = 4
    
        index_delete = [] #去掉这些index右边的线
        cut_final = []
        width_untilnow = 0
        for i in range(len(width_data)):
            if label[i] == label_width_small and width_untilnow == 0:
                index_delete.append(i)
                cut_left = cut[i][0]
                width_untilnow = cut[i][1] - cut[i][0]
                #print cut_left,width_untilnow,i
            elif label[i] != 3 and label[i] != 4 and width_untilnow != 0:
                width_untilnow = cut[i][1] - cut_left
                if isnormal_width(width_untilnow, width_normal_min, width_normal_max) == -1: #还不够长
                    index_delete.append(i)
                    #print cut_left,width_untilnow,i
                elif isnormal_width(width_untilnow, width_normal_min, width_normal_max) == 0: #拼成一个完整的字
                    width_untilnow = 0
                    cut_right = cut[i][1]
                    tmp = (cut_left, cut_right)
                    cut_final.append(tmp)
                    #print 'complete',i
                elif isnormal_width(width_untilnow, width_normal_min, width_normal_max) == 1:   #一下子拼多了
                    #print 'cut error!!!!',cut_left,width_untilnow,i 
                    width_untilnow = 0
                    cut_right = cut[i-1][1]
                    tmp = (cut_left, cut_right)
                    cut_final.append(tmp)
                    #print 'cut error!!!!',i 
                    index_delete.append(i)
                    cut_left = cut[i][0]
                    width_untilnow = cut[i][1] - cut[i][0]
                    if i == len(width_data):
                        tmp = (cut[i][0], cut[i][1])
                        cut_final.append(tmp)
                        #print i
            else:
                tmp = (cut[i][0], cut[i][1])
                cut_final.append(tmp)
        i1 = len(cut_final) - 1
        i2 = len(cut) - 1
        if cut_final[i1][1] != cut[i2][1]:
            tmp = (cut[i2][0], cut[i2][1])
            cut_final.append(tmp)
                
    else:
        cut_final = cut
                             
    for x in range(len(cut_final)):
        ax = (cut_final[x][0] - 1, 0, cut_final[x][1] + 1, height)
        temp = line_img.crop(ax)
        temp = image_unit.resize(temp, save_size)
        temp.save('{}/{}.jpg'.format(save_path, x))
    return flag


def cut_by_kmeans_new(pic_path, save_path, save_size, margin_two_character): #margin_two_character 两字符的间距，法定代表人那行是2， 其余为4
    src_pic = Image.open(pic_path).convert('L') #先把图片转化为灰度
    src_arr = np.array(src_pic)
    threshold = OTSU_enhance(src_arr) * 0.9 #用大津阈值
    bin_arr = np.where(src_arr < threshold, 0, 255) #二值化图片
    
    horz = horizon(bin_arr) #获取到该行的 水平 方向的投影
    line_arr, flag = cut_line(horz, src_pic) #把文字（行）所在的位置切割下来
    if flag == False:
        return flag
    line_arr = np.where(line_arr < threshold, 0, 255)
    line_img = Image.fromarray((255 - line_arr).astype("uint8"))
    width, height = line_img.size
    
    vert = vertical(line_arr) #获取到该行的 垂直 方向的投影
    cut = simple_cut(vert) #先进行简单的文字分割（即有空隙就切割）
    
    #cv.line(img,(x1,y1), (x2,y2), (0,0,255),2)    
    width_data = []
    width_data_TooBig = []
    width_data_withoutTooBig = []
    for i in range(len(cut)):
        tmp = (cut[i][1] - cut[i][0], 0) 
        if tmp[0] > height * 1.8:     #比这一行的高度大两倍的肯定是连在一起的
            temp = (tmp[0], i)
            width_data_TooBig.append(temp)
        else:
            width_data_withoutTooBig.append(tmp)
        width_data.append(tmp)
    if width_data_withoutTooBig == []:
        return False
    kmeans = KMeans(n_clusters=2).fit(width_data_withoutTooBig)
    #print "聚簇中心点:", kmeans.cluster_centers_
    #print 'label:', kmeans.labels_ 
    #print '方差:', kmeans.inertia_
    
    label_tmp = kmeans.labels_
    label = []
    j = 0
    k = 0
    for i in range(len(width_data)):        #将label整理，2代表大于一个字的
        if j != len(width_data_TooBig) and k != len(label_tmp):
            if i == width_data_TooBig[j][1]:
                label.append(2)
                j = j + 1
            else:
                label.append(label_tmp[k])
                k = k + 1
        elif j == len(width_data_TooBig) and k != len(label_tmp):
            label.append(label_tmp[k])
            k = k + 1
        elif j != len(width_data_TooBig) and k == len(label_tmp):
            label.append(2)
            j = j + 1
            
    label0_example = 0
    label1_example = 0
    for i in range(len(width_data)):
        if label[i] == 0:
            label0_example = width_data[i][0]
        elif label[i] == 1:
            label1_example = width_data[i][0]
    if label0_example > label1_example:    #找到正常字符宽度的label(宽度大的，防止切得太碎导致字符宽度错误)
        label_width_normal = 0
    else:
        label_width_normal = 1
    label_width_small = 1 - label_width_normal
        
    cluster_center = []
    cluster_center.append(kmeans.cluster_centers_[0][0])
    cluster_center.append(kmeans.cluster_centers_[1][0])
    for i in range(len(width_data)):
        if label[i] == label_width_normal and width_data[i][0] > cluster_center[label_width_normal] * 4 / 3: #5/4可优化，选这个的理由是洲
            label[i] = 2
            temp = (width_data[i][0], i)
            width_data_TooBig.append(temp)      
            
    #max_gap = get_max_gap(cut) #获取这些被切割体之前最大的距离 找':', label = 3
    #for i in range(len(label)):
    #    if i == max_gap[1]:
    #        label[i] = 3
    
    width_normal_data = []      #存正常字符宽度
    width_data_TooSmall = []
    for i in range(len(width_data)):
        if label[i] == label_width_normal:
            width_normal_data.append(width_data[i][0])
        elif label[i] != label_width_normal and label[i] != 2:  #切得太碎的
            box=(cut[i][0],0,cut[i][1],height)
            region=line_img.crop(box) #此时，region是一个新的图像对象。
            region_arr = 255 - np.array(region)
            region = Image.fromarray(region_arr.astype("uint8"))
            name = "./test1/single"+str(i)+".jpg"
            #region.save(name)
            tmp = (width_data[i][0], i)
            width_data_TooSmall.append(tmp)
    width_normal_max = max(width_normal_data)
    width_normal_min = min(width_normal_data)       #得到正常字符宽度的上下限
    
    
    
    if len(width_data_TooBig) != 0:   
        for i in range(len(width_data_TooBig)):
            index = width_data_TooBig[i][1]
            mid = (cut[index][0] + cut[index][1]) / 2
            tmp1 = (cut[index][0], int(mid))
            tmp2 = (int(mid)+1, cut[index][1])
            del cut[index]
            cut.insert(index, tmp2)
            cut.insert(index, tmp1)
            del width_data[index]
            tmp1 = (tmp1[1] - tmp1[0], index)
            tmp2 = (tmp2[1] - tmp2[0], index+1)
            width_data.insert(index, tmp2)
            width_data.insert(index, tmp1)
            label[index] = label_width_normal
            label.insert(index, label_width_normal)
            
            
    if len(width_data_TooSmall) != 0:               #除':'以外有小字符,先找'('、')'label = 4,其他标点 label=5                             
        for i in range(len(width_data_TooSmall)):
            index = width_data_TooSmall[i][1]
            border_left = cut[index][0] + 1
            border_right = cut[index][1]
            RoI_data = line_arr[:,border_left:border_right]
            
            #RoI_data = np.where(RoI_data < threshold, 0, 1)
            horz = horizon(RoI_data)
            horz_reverse = list(reversed(horz))
            height_now = len(horz)
            flag_ = 0
            height_begin_2 = 0
            height_end_1 = 0
            height_end_2 = 0
            height_begin_1 = 0

            for j in range(height_now):                 #判断冒号，有两段连续，中间空的，两段的高度和比高度的一半小
                if horz[j] != 0 and flag_ == 0:
                    height_begin_1 = j
                    flag_ = 1
                if horz[j] == 0 and flag_ == 1:
                    height_end_1 = j
                    flag_ = 0
                    break
            flag_ = 0
            for j in range(height_now):
                if horz_reverse[j] != 0 and flag_ == 0:
                    height_end_2 = height_now - j
                    flag_ = 1
                if horz_reverse[j] == 0 and flag_ == 1:
                    height_begin_2 = height_now - j
                    flag_ = 0
                    break
            flag1 = height_begin_2 - height_end_1
            flag_maohao = 0
            if flag1 != 0:
                count0_maohao = 0
                for j in range(flag1):
                    if horz[height_end_1+j] == 0:
                        count0_maohao = count0_maohao + 1
                if count0_maohao == flag1:
                    #print "maohao!",i,(height_end_2-height_begin_2 + height_end_1-height_begin_1),height_now
                    flag_maohao = flag_maohao + 1
                    
            if height_end_2 - height_begin_1 < height / 2 and flag_maohao == 0: #不在疑似冒号集合，其他标点
                label[index] = 5
                continue
                
            #在冒号疑似集合中，两段的和比高度的一半小
            if (height_end_2-height_begin_2 + height_end_1-height_begin_1) < height / 2 and flag_maohao == 1:
                label[index] = 3
                continue
            
            up_down = np.sum(np.abs(RoI_data - RoI_data[::-1]))
            left_right = np.sum(np.abs(RoI_data - RoI_data[:,::-1]))
            vert = vertical(RoI_data)
        
            if up_down <= left_right * 0.62 and up_down >= left_right * 0.3 and np.array(vert).var() < len(vert) * 5 and len(vert) < 10:    
                #print i, up_down, left_right,
                #print vert, np.array(vert).var(),len(vert)
                label[index] = 4
    
        cut_new = cut  
        index_delete = [] #去掉这些index右边的线
        cut_final = []
        width_untilnow = 0
        for i in range(len(width_data)):
            if label[i] == label_width_small and width_untilnow == 0:
                index_delete.append(i)
                cut_left = cut[i][0]
                width_untilnow = cut[i][1] - cut[i][0]
                #print cut_left,width_untilnow,i
            elif label[i] != 3 and label[i] != 4 and label[i] != 5 and width_untilnow != 0:
                width_untilnow = cut[i][1] - cut_left
                if isnormal_width(width_untilnow, width_normal_min, width_normal_max) == -1: #还不够长
                    index_delete.append(i)
                    #print cut_left,width_untilnow,i
                elif isnormal_width(width_untilnow, width_normal_min, width_normal_max) == 0: #拼成一个完整的字
                    if cut[i][0] - cut[i-1][1] < margin_two_character:     #窄的图片间距足够小才拼接
                        width_untilnow = 0
                        cut_right = cut[i][1]
                        tmp = (cut_left, cut_right)
                        cut_final.append(tmp)
                        #print 'complete',i,tmp
                    else:
                        width_untilnow = 0              #上一个是完整的字，因为间距过大
                        cut_right = cut[i-1][1]
                        tmp = (cut_left, cut_right)
                        cut_final.append(tmp)
                        #print 'too big margin',i,tmp
                        cut_left = cut[i][0]
                        width_untilnow = cut[i][1] - cut[i][0]
                elif isnormal_width(width_untilnow, width_normal_min, width_normal_max) == 1:   #一下子拼多了
                    #print 'cut error!!!!',cut_left,width_untilnow,i 
                    if label[i-1] == 3 or label[i-1] == 4 or label[i-1] == 5:
                        width_untilnow = 0
                        cut_right = cut[i-2][1]
                        tmp = (cut_left, cut_right)
                        cut_final.append(tmp)
                        #print "1",i,tmp
                    else:
                        width_untilnow = 0
                        cut_right = cut[i-1][1]
                        tmp = (cut_left, cut_right)
                        cut_final.append(tmp)
                        #print "2",i,tmp
                    #print 'cut error!!!!',i 
                    index_delete.append(i)
                    cut_left = cut[i][0]
                    width_untilnow = cut[i][1] - cut[i][0]
                    #print width_untilnow
                    if i == len(width_data):
                        tmp = (cut[i][0], cut[i][1])
                        cut_final.append(tmp)
                        #print i
            else:
                tmp = (cut[i][0], cut[i][1])
                cut_final.append(tmp)
        i1 = len(cut_final) - 1
        i2 = len(cut) - 1
        if cut_final[i1][1] != cut[i2][1]:
            tmp = (cut[i2][0], cut[i2][1])
            cut_final.append(tmp)
                
    if len(width_data_TooSmall) == 0:
        cut_final = cut
                            
    cut_final.sort(key= lambda k:k[0])
                             
    cut_final_withlabel = []
    for i in range(len(cut_final)):
        for j in range(len(cut)):
            if cut_final[i][0] == cut[j][0]:
                tmp = (cut_final[i][0], cut_final[i][1], label[j])
                cut_final_withlabel.append(tmp)
                j = len(cut) - 1
                
    
    for x in range(len(cut_final_withlabel)):
        if cut_final_withlabel[x][2] != 5:
            ax = (cut_final_withlabel[x][0] - 1, 0, cut_final_withlabel[x][1] + 1, height)
            temp = line_img.crop(ax)
            temp = image_unit.resize(temp, save_size)
            temp.save('{}/{}.jpg'.format(save_path, x))
    return flag
'''
label:
    0 / 1:  
    2: 大于一个字
    3: ':'
    4: '('or')'
    5: 逗号、句号、顿号、星花
'''












