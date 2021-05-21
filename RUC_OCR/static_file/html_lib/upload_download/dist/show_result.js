(function($){
    
    var img_num = $("img.TreatedImage").length;
    for(var x = Math.ceil(img_num / 4); x >= 1; x--){
        $("div.checkbox_form").after("<br><div class='img_line' id='" + x + "_line'></div>");
    }
    
    $("img.TreatedImage").each(function(){
        var img_id = parseInt($(this).attr("id"));
        var img_line_id = Math.ceil(img_id / 4);
        if(img_id % 4 == 1){
            $($('#'+img_line_id+'_line')).append($('#'+img_id+'_div'));
        }
        else{
            $($('#'+(img_id-1)+'_div')).after($('#'+img_id+'_div'));
        }
    });
    $("div.img_line").each(function(){
        var line_id_num = parseInt($(this).attr("id").split('_')[0]);
        var line_id = '#' + $(this).attr("id");
        for(var x=(line_id_num-1)*4+1; x<=line_id_num*4; x++){
            var result_div_id = "#" + x + "_show";
            console.log(result_div_id);
            $($(line_id)).after($(result_div_id));
        }
    });
    $("div.OCRresult").hide();
    $(document.body).mouseover(function(e){
        if(e.target.tagName=="IMG" && e.target.className=="TreatedImage"){
            var result_div_id = "#" + e.target.id + "_show";
            
            if($(result_div_id).css("display")=="none"){
                var pic_pos = parseInt(e.target.id);
                $("div.OCRresult").slideUp("slow");
                pic_pos = pic_pos % 4;
                if(pic_pos == 0)
                    pic_pos=4;
                pic_pos = 24*pic_pos-10 + '%';
            
                $(result_div_id).children("div.inner").css("left",pic_pos);
                $(result_div_id).slideDown("slow");
                $(result_div_id).mouseleave(function(){$(result_div_id).slideUp("slow");});
            }
        }
        if(e.target.tagName=="SECTION"){
            $("div.OCRresult").slideUp("slow");
        }
    });
}(jQuery));
