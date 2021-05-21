function show_progress_bar(){
    $("#prog_out").css("display","block");
    var frcnn_pro = 0;
    var OCR_pro = 0;
    var select_pro = 0;

    var get_frcnn_pro = setInterval(function(){
        var get_frcnn_pro_url = '/get_frcnn_pro';
        $.getJSON(get_frcnn_pro_url, function(data){
            frcnn_pro = data;
            $('#frcnn_pb').width((frcnn_pro * 0.5) + '%');
            $('#frcnn_pb_text').empty();
            $('#frcnn_pb_text').append('目标定位' + frcnn_pro + '%');
            if(frcnn_pro == 100)
                clearInterval(get_frcnn_pro);
        });
    }, 1000);

    var get_OCR_pro = setInterval(function(){
        if(frcnn_pro == 100){
            var get_OCR_pro_url = '/get_OCR_pro';
            $.getJSON(get_OCR_pro_url, function(data){
                OCR_pro = data;
                $('#OCR_pb').width((OCR_pro * 0.3) + '%');
                $('#OCR_pb_text').empty();
                $('#OCR_pb_text').append('文字识别' + OCR_pro + '%');
                if(OCR_pro == 100)
                    clearInterval(get_OCR_pro);
            });
        }
    }, 1000);

    var get_select_pro = setInterval(function(){
        if(OCR_pro == 100){
            var get_select_pro_url = '/get_select_pro';
            $.getJSON(get_select_pro_url, function(data){
                select_pro = data;
                $('#select_pb').width((select_pro * 0.2) + '%');
                $('#select_pb_text').empty();
                $('#select_pb_text').append('结果整理' + select_pro + '%');
                if(select_pro == 100)
                    clearInterval(get_OCR_pro);
            });
        }

    }, 1000);
}
