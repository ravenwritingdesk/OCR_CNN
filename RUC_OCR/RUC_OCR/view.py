#coding:utf-8
from django.http import HttpResponse,StreamingHttpResponse
from django.shortcuts import render_to_response, render
import os,sys,time
from subprocess import Popen,PIPE 
import imghdr
import zipfile,tarfile,rarfile
from datetime import datetime
import image_unit
import frcnn
import OCR_unit
import xlwt

baseDir = os.path.dirname(os.path.abspath(__name__))
OCRmodel = os.path.join(baseDir ,'OCRmodel', 'all_char_model2')
OCRlabel = os.path.join(baseDir ,'OCRlabel', 'all_label7.names')

def initDir(name):
    global uploadDir
    global saveDir
    global splitDir
    global charDir
    global downloadDir
    uploadDir = os.path.join(baseDir, 'static_file', 'Upload_DownloadFile', name)
    os.mkdir(uploadDir)
    saveDir = os.path.join(baseDir,'static_file', 'UntreatedImage', name)
    os.mkdir(saveDir)
    splitDir = os.path.join(baseDir,'static_file', 'SplitImage', name)
    os.mkdir(splitDir)
    charDir = os.path.join(baseDir,'static_file', 'SplitChar', name)
    os.mkdir(charDir)
    downloadDir = os.path.join(baseDir, 'static_file', 'Upload_DownloadFile', name + 'Download')
    os.mkdir(downloadDir)

def test(request):
    return render_to_response('index.html')

def savefile(src_file, save_file):
    for chunk in src_file.chunks():
        save_file.write(chunk)
    save_file.close()

def extract(f):
    if zipfile.is_zipfile(f):
        fz=zipfile.ZipFile(f,'r')
        for file in fz.namelist():
            '''
            img_type=imghdr.what(file)
            if img_type == 'jpeg':
            '''
            fz.extract(file,uploadDir)
        fz.close()
        return True
    if tarfile.is_tarfile(f):
        fz=tarfile.open(f)
        for file in fz.getnames():
            '''
            img_type = imghdr.what(file)
            if img_type == 'jpeg':
            '''
            fz.extract(file,uploadDir)
        fz.close()
        return True
    if rarfile.is_rarfile(f):
        fz=rarfile.RarFile(f)
        for file in fz.namelist():
            '''
            img_type = imghdr.what(file)
            if img_type == 'jpeg':
            '''
            fz.extract(file,uploadDir)
        fz.close()
        return True
    return False

def get_imgpath():
    for x in os.listdir(uploadDir):
        path = os.path.join(uploadDir, x)
        if os.path.isdir(path):
            return path
    return uploadDir

def OCR(filepath):
    image_unit.convert_png2jpg(filepath, saveDir)
    image_unit.add_side(saveDir)
    #frcnn_detect predict
    sess, net = frcnn.init(0,0)
    for x in os.listdir(saveDir):
        results = frcnn.detect(sess, net, os.path.join(saveDir, x)) 
        #xmin,ymin,xmax,ymax = result[0]*width, result[1]*height, result[2]*width, result[3]*height
        #object_num = result[4]   prob = result[5]   object_name = result[6]
        if len(results) != 0:
            image_unit.frcnn_crop(os.path.join(saveDir, x), results, splitDir, charDir)
    sess.close()
    ch_image_set, en_image_set = OCR_unit.load_charimg(charDir, img_size = 64)
    image_set = ch_image_set + en_image_set
    text = {}
    val_dict, idx_dict = OCR_unit.detect(OCRmodel, 3700, image_set)
    text = OCR_unit.predict_text(val_dict, idx_dict, OCRlabel, text)
    return OCR_unit.select_target(text, [u'名称',u'注册号'], saveDir)

def write2excel(target_string):
    excel_file = xlwt.Workbook(encoding = 'utf-8')
    table = excel_file.add_sheet('sheet1')
    table.write(0, 0, u"图片名")
    table.write(0, 1, u"注册号")
    table.write(0, 2, u"名称")
    for row, x in enumerate(target_string):
        if 'img_id' in x:
            table.write(row + 1, 0, x['img_id'])
        if 'num' in x:
            table.write(row + 1, 1, x['num'])
        if 'name' in x:
            table.write(row + 1, 2, x['name'])
    excel_file.save(os.path.join(downloadDir, 'download.xls'))

def OCRweb(request):
    context = {}
    if request.POST:
        if request.POST.has_key('OCRpicSubmit'):
            UntreatedImage = request.FILES.get('OCRpic')
            rand_name = datetime.now().strftime('%Y%m%d%H%M%S')
            initDir(rand_name)
            image_name = UntreatedImage.name
            imgpath = os.path.join(uploadDir, image_name)
            imgfile = open(imgpath, 'wb+')
            savefile(UntreatedImage, imgfile)
            context['target_string'] = OCR(uploadDir)
            context['image_path'] = os.path.join('UntreatedImage', rand_name, image_name.split('.')[0] + '.jpg')
            print context['image_path']
        if request.POST.has_key('OCRtarSubmit'):
            UntreatedTar = request.FILES.get('OCRtar')
            rand_name = datetime.now().strftime('%Y%m%d%H%M%S')
            initDir(rand_name)
            tar_name = rand_name + '.' + UntreatedTar.name.split('.')[-1]
            tarpath = os.path.join(uploadDir, tar_name)
            tarfile = open(tarpath, 'wb+')
            savefile(UntreatedTar, tarfile)
            extract(tarpath)
            imgpath = get_imgpath()
            context['target_string'] = OCR(uploadDir)
            context['show_table']=True
            write2excel(context['target_string'])
            context['download_path']=os.path.join(rand_name + 'Download', 'download.xls')
    return render_to_response('index.html', context)
