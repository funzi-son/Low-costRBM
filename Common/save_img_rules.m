function save_img_rules(T,R,vis_dir)
    rNum = size(T.r,1);    
    for i=1:rNum
        rname = strcat(vis_dir,num2str(i-1),'.BMP');        
        rind = find(T.r(i,:)>0);
        img = R.c(i)*R.r(rind,:);        
        %img(img<0) =  0;
        MN = min(min(img));        
        MX = max(max(img));                
        img = (img-MN)/(MX-MN);
        
        save_images(img,size(rind,2),28,28,rname);
    end
end