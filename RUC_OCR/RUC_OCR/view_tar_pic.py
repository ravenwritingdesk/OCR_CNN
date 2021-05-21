#coding:utf-8
from django.http import HttpResponse,StreamingHttpResponse, JsonResponse
from django.shortcuts import render_to_response, render
from django.template.defaulttags import register
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
@register.filter
def get_target_text(d, key_name):
    return d[key_name]['target_text']

@register.filter
def get_result_text(d, key_name):
    if 'result_text' in d[key_name]:
        return d[key_name]['result_text']
    else:
        return ''

@register.filter
def get_OCRdict(d, key_name):
    return d[key_name]

def initDir(name):
    
    global uploadDir
    global saveDir
    global splitDir
    global charDir
    global downloadDir
    

    uploadDir = os.path.join(baseDir, 'static_file', 'Upload_DownloadFile', name)
    os.mkdir(uploadDir)
    saveDir = os.path.join(baseDir, 'static_file','UntreatedImage', name)
    os.mkdir(saveDir)
    splitDir = os.path.join(baseDir, 'static_file','SplitImage', name)
    os.mkdir(splitDir)
    charDir = os.path.join(baseDir, 'static_file', 'SplitChar', name)
    os.mkdir(charDir)
    downloadDir = os.path.join(baseDir, 'static_file', 'Upload_DownloadFile', name + 'Download')
    os.mkdir(downloadDir)

def init_return_var():
    global frcnn_pro
    global OCR_pro
    global select_pro
    frcnn_pro = 0
    OCR_pro = 0
    select_pro = 0

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
    uploadList = []
    for x in os.listdir(uploadDir):
        path = os.path.join(uploadDir, x)
        if os.path.isdir(path):
            return path
    return uploadDir



def get_frcnn_pro(request):
    return JsonResponse(frcnn_pro, safe=False)

def get_OCR_pro(request):
    global OCR_pro
    OCR_pro = OCR_unit.return_OCR_pro()
    return JsonResponse(OCR_pro, safe=False)

def get_select_pro(request):
    global select_pro
    select_pro = OCR_unit.return_select_pro()
    return JsonResponse(select_pro, safe=False)


def OCR(filepath, OCRtarget):
    
    global frcnn_pro
    global OCR_pro
    global select_pro
    image_unit.convert_png2jpg(filepath, saveDir)
    image_unit.add_side(saveDir)
    frcnn_len = len(os.listdir(saveDir))
    #frcnn_detect predict
    sess, net = frcnn.init(0,0,OCRtarget)
    for count, x in enumerate(os.listdir(saveDir)):
        results = frcnn.detect(sess, net, os.path.join(saveDir, x)) 
        #xmin,ymin,xmax,ymax = result[0]*width, result[1]*height, result[2]*width, result[3]*height
        #object_num = result[4]   prob = result[5]   object_name = result[6]
        if len(results) != 0:
            image_unit.frcnn_crop(os.path.join(saveDir, x), results, splitDir, charDir)
        frcnn_pro = float(count + 1) / float(frcnn_len) * 100.0
    sess.close()
    ch_image_set, en_image_set = OCR_unit.load_charimg(charDir, img_size = 64)
    image_set = ch_image_set + en_image_set
    text = {}
    val_dict, idx_dict = OCR_unit.detect(OCRmodel, 3700, image_set)
    text = OCR_unit.predict_text(val_dict, idx_dict, OCRlabel, text)
    return OCR_unit.select_target(text, OCRtarget, saveDir)

def write2excel(target_string, OCRtarget):
    excel_file = xlwt.Workbook(encoding = 'utf-8')
    table = excel_file.add_sheet('sheet1')
    table.write(0, 0, u"图片名")
    for col, x in enumerate(OCRtarget):
        table.write(0, col + 1, OCR_dict[x])

    for row, x in enumerate(target_string):
        if 'img_id' in x:
            table.write(row + 1, 0, x['img_id'])
        for col, target in enumerate(OCRtarget):
            if target in x['OCRtext']:
                if 'result_text' in x['OCRtext'][target]:
                    table.write(row + 1, col + 1, x['OCRtext'][target]['result_text'])
  
    excel_file.save(os.path.join(downloadDir, 'download.xls'))


def show_progress(request):
    
    return JsonResponse(48 ,safe=False)


def OCRweb(request):
    context = {}
    global frcnn_pro
    init_return_var()
    OCR_unit.init_return_var()
    if request.is_ajax():
        rand_name = datetime.now().strftime('%Y%m%d%H%M%S')
        initDir(rand_name)
        OCRtarget = request.POST.getlist('OCRtarget')
        print OCRtarget
        for UploadFile in request.FILES.getlist('OCRUpload'):
            file_name = UploadFile.name
            file_type = file_name.split('.')[1]
            file_save_path = os.path.join(uploadDir, file_name)
            user_file = open(file_save_path, 'wb+')
            savefile(UploadFile, user_file)
            if file_type in ['zip', 'tar', 'rar']:
                extract(file_save_path)
        context['show_result'] = True
        context['target_string'] = OCR(uploadDir, OCRtarget)
        context['OCRtarget'] = OCRtarget
        context['OCRdict'] = OCR_dict
        write2excel(context['target_string'], OCRtarget)
        context['download_path']=os.path.join('Upload_DownloadFile', rand_name + 'Download', 'download.xls')   
        '''
    if request.POST:
        if request.POST.has_key('OCRpicSubmit'):
            UntreatedImage = request.FILES.get('OCRpic')
            rand_name = datetime.now().strftime('%Y%m%d%H%M%S')
            initDir(rand_name)
            image_name = UntreatedImage.name
            imgpath = os.path.join(uploadDir, image_name)
            imgfile = open(imgpath, 'wb+')
            savefile(UntreatedImage, imgfile)
            context['target_string'] = OCR(imgpath)
            context['image_path'] = rand_name + imgpath.split(uploadDir)[1]
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
            context['target_string'] = OCR(imgpath)
            context['show_table']=True
            write2excel(context['target_string'])
            context['download_path']=os.path.join(rand_name + 'Download', 'download.xls')
        '''
    return render_to_response('test1.html', context)
