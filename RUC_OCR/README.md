# RUC_OCR: 人大人OCR
根据中国软件杯大赛需求，这是一个针对工商信息执照进行文字提取和处理的OCR系统，可以有效处理包含“企业名称”和“企业注册号”文本中的文字信息。
本软件以网页为交互形式，给用户提供单张图片处理接口，和图片压缩包处理接口。
实验证明，对工商信息执照中目标文字的识别率可以达到98.93%，用单张1080TI的GPU处理50张图片的时间大约为48s。

### 要求系统环境
I.python2.7
II.系统要求ubuntu 14及以上
III.可支持CUDA加速计算的GPU和环境
IV.django web应用框架
### 文件路径
I.模型路径
    ./FrcnnModel --Faster RCNN模型
    ./OCRmodel --OCR模型
    ./OCRlabel --OCR标签
II.图片路径
    ./static_file/Upload_DownloadFile --用户上传下载的文件
    ./static_file/UntreadtedImage --待处理的图片
    ./static_file/SplitImage --分割后目标文本的图片
    ./static_file/SplitChar --分割后字符的图片
III.Faster RCNN库函数路径
    ./lib
IV.软件后端文件路径
    ./RUC_OCR
V.网页html路径
    ./website

### 软件部署方式
I.进入lib，编译库文件
    cd ./lib
    make
II.部署网页
    python ./manage.py runserver 0.0.0.0:8080
    runserver 后面是部署的ip地址和端口号
    例如0.0.0.0:8080，即在浏览器中输入网址 http://0.0.0.0:8080/index

### 测试地址
    同时，在服务器上已经布置好了一个可以运行的版本，大家可以访问进行测试
