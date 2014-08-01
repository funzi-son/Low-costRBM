function exp_mnist10K_rule_vs_model()
addpath(genpath('./'));
EXP_DIR = './';
lm = '/';

load mnist_train_dat_10k;
load mnist_train_lab_10k;


load mnist_test_dat_10k;
load mnist_test_lab_10k;
    

conf.hidNum = 1000;
conf.eNum   = 50;
conf.bNum   = 100;
conf.sNum   = 100;
conf.gNum   = 1;
conf.params = [0.2 0.2 0.01 0.00002];

conf.lambda = 0;
conf.p = 0.0001; 


 model = train_rbm_(conf,traind);
 save(strcat(EXP_DIR,'rbm_h',num2str(conf.hidNum),'_lr',num2str(conf.params(1)),'.mat'),'model');


fprintf('Extract features\n');
trn_ftrs = logistic(traind*model.W + repmat(model.hidB,size(traind,1),1));
%vld_ftrs = logistic(vldd*model.W + repmat(model.hidB,size(vldd,1),1));
tst_ftrs = logistic(testd*model.W + repmat(model.hidB,size(testd,1),1));

fprintf('Extract rules \n')
R = extract_rbm(model,'',0);

trn_ftrs_r = rule_inference_(R,traind);
%vld_ftrs_r = rule_inference_(R,vldd);
tst_ftrs_r = rule_inference_(R,testd);

c = 5;
g = 0.05;
 modelsvm = svmtrain(trainl, trn_ftrs,['-h 0 -c ' num2str(c) ' -g ' num2str(g)]);
    
%     [~, accuracy, ~] = svmpredict(vldl, vld_ftrs, model);
%     vld_acc = accuracy(1);
   [~, accuracy, ~] = svmpredict(testl, tst_ftrs, modelsvm);
   tst_acc = accuracy(1);
    
    % ----------------------
    modelsvm = svmtrain(trainl, trn_ftrs_r,['-c ' num2str(c) ' -g ' num2str(g)]);
    
    %[~, accuracy, ~] = svmpredict(vldl, vld_ftrs_r, model);
    %vld_acc_r = accuracy(1);
    
    [~, accuracy, ~] = svmpredict(testl, tst_ftrs_r, modelsvm);
    tst_acc_r = accuracy(1);
    fprintf('Model accuracy = %.5f\n',tst_acc)
    fprintf('Low-cost accuracy = %.5f\n',tst_acc_r);
    
    %logging(log_file,[conf.hidNum lr ld c g vld_acc vld_acc_r]);    


end

