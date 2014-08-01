function exp_mnist10K_rule_vs_model_()
sys_str = computer();
if ~isempty(findstr('WIN',sys_str))
     EXP_DIR = '.\';   
     addpath(genpath('..\'));
     lm = '\';    
elseif ~isempty(findstr('GLN',sys_str))
     EXP_DIR = './';         
     addpath(genpath('../'));
     lm = '/';
else
    fprintf('Cannot find paths\n');
    return;
end

for trial=1:5  
%load mnist_train_dat_10k;
%load mnist_train_lab_10k;
%traind = traind(1:5000,:);
%trainl = trainl(1:5000);
lr = 0.2;
ld = 0;
m_name = strcat(EXP_DIR,'rbm_h500_lr',num2str(lr),'_trial',num2str(trial),'.mat');
f_name = strrep(m_name,'rbm','rbmftr');

if exist(f_name,'file')
    load(f_name,'trn_ftrs','tst_ftrs');
    %load(d_name,'trn_ftrs_r','tst_ftrs_r');
else
load mnist_train_dat_60k;
load mnist_test_dat_10k;


load(m_name,'model');        

fprintf('Get layer1 features\n');

trn_ftrs = logistic(bsxfun(@plus,traind*model.W,model.hidB));
save(f_name,'trn_ftrs');
clear trn_ftrs traind;
%vldd = logistic(vldd*model.W + repmat(model.hidB,size(vldd,1),1));
tst_ftrs = logistic(testd*model.W + repmat(model.hidB,size(testd,1),1));
save(f_name,'tst_ftrs','-append');
%fprintf('Extract rules \n')
%R= extract_rbm(model,'',0);
clear model testd;;

%trn_ftrs_r = rule_inference_(R,traind);
%vld_ftrs_r = rule_inference_(R,vldd);
%tst_ftrs_r = rule_inference_(R,testd);

end
fprintf('Starting layer 2 training\n');
lr2 = 0.1;

conf.hidNum = 1000;
conf.eNum   = 50;
conf.bNum   = 600;
conf.sNum   = 100;
conf.gNum   = 1;
conf.params = [lr2 lr2 0.01 0.00002];

conf.lambda = 0;
conf.p = 0.0001; % this seem does not effect much

conf.row = 28;
conf.col = 28;

log_file = strcat(EXP_DIR,'trial_',num2str(conf.hidNum),'dbn',num2str(trial),'.mat');
d_name = strrep(m_name,'rbm',strcat('dbn2_2h1000_2lr_',num2str(lr2)));
if ~exist(d_name,'file')
    load(f_name,'trn_ftrs');
    model = train_rbm_(conf,trn_ftrs); 
    save(d_name,'model');
else
    load(d_name,'model');
end
fprintf('Extract features\n');
clearvars -except model trn_ftrs trial;
whos
trn_ftrs = logistic(bsxfun(@plus,trn_ftrs*model.W, model.hidB));
%vld_ftrs = logistic(vldd*model.W + repmat(model.hidB,size(vldd,1),1));
tst_ftrs = logistic(bsxfun(@plus,tst_ftrs*model.W,model.hidB));

%fprintf('Extract rules \n')
%R = extract_rbm(model,'',0);

%trn_ftrs_r = rule_inference_(R,trn_ftrs_r);
%vld_ftrs_r = rule_inference_(R,vldd);
%tst_ftrs_r = rule_inference_(R,tst_ftrs_r);

load mnist_train_lab_60k;
load mnist_test_lab_10k;

for c = [5]%[0.01 0.05 0.1 0.5 1 5 10]
for g = [0.05]%[0.01 0.05 0.1 0.5 1 5 10]    
    svm_model = svmtrain(trainl, trn_ftrs,['-h 0 -c ' num2str(c) ' -g ' num2str(g)]);    
%     [~, accuracy, ~] = svmpredict(vldl, vld_ftrs, model);
%     vld_acc = accuracy(1);
   [~, accuracy, ~] = svmpredict(testl, tst_ftrs, svm_model);
   tst_acc = accuracy(1);
    
    % ----------------------
%    svm_model = svmtrain(trainl, trn_ftrs_r,['-c ' num2str(c) ' -g ' num2str(g)]);
    
    %[~, accuracy, ~] = svmpredict(vldl, vld_ftrs_r, model);
    %vld_acc_r = accuracy(1);
    
 %   [~, accuracy, ~] = svmpredict(testl, tst_ftrs_r, svm_model);
    tst_acc_r = 0;%accuracy(1);
    logging(log_file,[conf.hidNum lr lr2 ld c g tst_acc tst_acc_r]);
    
    %logging(log_file,[conf.hidNum lr ld c g vld_acc vld_acc_r]);    
end % end c
end % end g
clear trn_ftrs tst_ftrs trn_ftrs_r tst_ftrs_r model svm_model;
end % trial
end