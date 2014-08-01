function model = train_rbm_l_1(conf,model1,odat,lab,otestd,testl,log_file)
% Train generative RBM with label
% sontran2013

dat = logistic(bsxfun(@plus,odat*model1.W,model1.hidB));
testd = logistic(bsxfun(@plus,otestd*model1.W,model1.hidB));

hidNum = conf.hidNum;
visNum = size(dat,2);
labNum = size(unique(lab),1);
sNum = conf.sNum;

model.W = 0.1*randn(visNum,hidNum);
model.U = 0.1*randn(labNum,hidNum);

model.visB = zeros(1,visNum);
model.hidB = zeros(1,hidNum);
model.labB = zeros(1,labNum);

WD    = zeros(size(model.W));
UD    = zeros(size(model.U));
visBD = zeros(size(model.visB));
hidBD = zeros(size(model.hidB));
labBD = zeros(size(model.labB));

lr = conf.params(1);
for e=1:conf.eNum;
    inx = randperm(size(dat,1));
    if lr>0.0001,  lr = conf.params(1)/ceil(e/10); disp(lr); end;
   disp(lr);
    res_e = 0;
    acc_e = 0;
    spr_e = 0;        
    for b=1:conf.bNum
        iiii = inx((b-1)*sNum+1:b*sNum);
        visP = dat(iiii,:);
        labP = lab(iiii) + 1;
        
       hidP = logistic(visP*model.W + model.U(labP,:) + repmat(model.hidB,sNum,1));
%         hidP = logistic(visP*model.W + repmat(model.hidB,sNum,1));
        hidPs = hidP>rand(size(hidP));
        hidNs = hidPs;
        %% gibb sampling
        for g=1:conf.gNum
            
            visN = logistic(bsxfun(@plus,hidNs*model.W',model.visB));
            visNs = visN>rand(size(visN));
            labN = softmax(exp(bsxfun(@plus,hidNs*model.U',model.labB)));
            
            hidN = logistic(visNs*model.W + model.U(labN,:) + repmat(model.hidB,sNum,1));            
%             hidN = logistic(visNs*model.W + repmat(model.hidB,sNum,1));
            hidNs = hidN>rand(size(hidN));
        end        
        res_e = res_e + sum(sqrt(sum((visP - visNs).^2,2)/visNum),1)/sNum;
        acc_e = acc_e + sum(labP == rbm_l_classify(model,visP,labNum))/sNum;
%        acc_e = acc_e + sum(sum(labP == labN))/sNum;
        %% updating
        w_diff = (visP'*hidP - visNs'*hidN)/sNum;
        WD = lr*(w_diff - conf.params(4)*model.W) + conf.params(3)*WD;
        model.W = model.W + WD;
        
        s_labP = discrete2softmax(labP,labNum);
        s_labN = discrete2softmax(labN,labNum);
        u_diff = (s_labP'*hidP - s_labN'*hidN)/sNum;        
        UD = lr*(u_diff - conf.params(4)*model.U) + conf.params(3)*UD;
        model.U = model.U + UD;
        
        visBD = lr*sum(visP - visNs,1)/sNum + conf.params(3)*visBD;
        model.visB  = model.visB + visBD;
        
        hidBD = lr*sum(hidPs - hidNs,1)/sNum + conf.params(3)*hidBD;
        model.hidB  = model.hidB + hidBD;
        
        labBD = lr*sum(s_labP - s_labN,1)/sNum + conf.params(3)*labBD;
        model.labB  = model.labB + labBD;
        
        %% Sparsity contrains
        if conf.lambda >0
           hidI = (visP*model.W +  repmat(model.hidB,sNum,1));
           hidN = logistic(hidI);
           pppp = (conf.p - sum(hidN,1)/sNum);
           model.W    = model.W   + lr*conf.lambda*(repmat(pppp,visNum,1).*(visP'*((hidN.^2).*exp(-hidI))));
           model.hidB = model.hidB + lr*conf.lambda*(pppp.*(sum((hidN.^2).*exp(-hidI),1)/sNum));           
           spr_e = spr_e + sum((conf.p - mean(hidN)).^2,2);
        end        
    end
    %% Classification with model
    output   = rbm_l_classify(model,testd,10);
    macc_e   = sum(sum(output==testl+1))/size(testl,1);  
    output   = gen_rbm_classify(model,testd);
    macc_r   = sum(sum(output==testl+1))/size(testl,1); 
        
    %% Classification with rule
     for gen_c1 = [0.3]
        if gen_c1==0
            cpen = 1;
        else 
            cpen = 0;
        end
        Rs(1) = extract_rbm(model1,[],cpen);
        for gen_c2 = [0.3]
        if gen_c2==0
            cpen = 1;
        else 
            cpen = 0;
        end
        [T R2] = extract_rbm_l(model,[],0,cpen);
        Rs(2) = R2;
        %% Generalize the rules
        [T_,Rs_] = prune_rule(T,Rs,[400 800]);
        % T_ = T; Rs_ = Rs;
        if gen_c1~=0 && gen_c2~=0, [T_,Rs_] = generalize_rule(T_,Rs_,[gen_c1 gen_c2]); end

        routput = rule_inference(Rs_,T_,otestd);
        racc = sum(sum(routput==testl+1))/size(testl,1);
        logging(log_file,[e macc_e macc_r racc]);

        fprintf('[Epoch %.3d] res_e = %.5f ||clf_e = %.5f ||macc_e=%.5f ||macc_r=%.5f ||racc=%.5f\n',e,res_e/conf.bNum,acc_e/conf.bNum,macc_e,macc_r,racc);
        end
        end     
        if ~isempty(conf.vis_dir)
            save_images(visN,100,28,28,strcat(conf.vis_dir,num2str(e),'.bmp'));
        end     
end   
end