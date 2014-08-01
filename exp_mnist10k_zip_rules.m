function exp_mnist10k_zip_rules()
% Testing pruning rules with RBMs
% sontran-2014
DAT_DIR = '/home/funzi/My.Academic/My.Codes/DATA/MNIST/';
EXP_DIR = '/home/funzi/Documents/Experiments/DBN_CPRESS/MNIST/';

for trial=1:10
for i1 = [0.05 0.2]
for i2 = [0]
for i3 = [0.0000001]
 %   if i2==0 && i3 ~=0.0001, continue; end
conf.hidNum = 1000;
conf.eNum = 50;
conf.bNum = 100;
conf.sNum = 100;
conf.gNum = 1;
conf.params = [i1 i1 0.01 0.00002];

conf.lambda = i2;
conf.p      = i3;

conf.row = 28;
conf.col = 28;
load(strcat(DAT_DIR,'mnist_train_dat_10k.mat'));
rbm_name = strcat(EXP_DIR,'rbm_h_',num2str(conf.hidNum),'_lr',num2str(i1),'_trial',num2str(trial),'.mat');
rule_name = strrep(rbm_name,'rbm','rbmrule'); 
ftr_name = strrep(rbm_name,'rbm','rbmftr');
if exist(rbm_name,'file')
    load(rbm_name);
    load(rule_name);
else
    model = train_rbm_(conf,traind);
    R = extract_rbm(model,'');
    save(rbm_name,'model')
    save(rule_name,'R');
end
load(strcat(DAT_DIR,'mnist_test_lab_10k.mat')); %'yale_test_label_',num2str(TST_N),'.mat'));
load(strcat(DAT_DIR,'mnist_train_lab_10k.mat'));%'yale_train_label_',num2str(TRN_N),'.mat'));

if ~exist(ftr_name,'file')

%load ticc_alphas_vld_dat;
%load ticc_alphas_vld_lab;


trn_ftrs = logistic(bsxfun(@plus,traind*model.W, model.hidB));
save(ftr_name,'trn_ftrs');
clear traind trn_ftrs;
%vld_ftrs = logistic(alp_vld_dat*model.W + repmat(model.hidB,size(alp_vld_dat,1),1));
load(strcat(DAT_DIR,'mnist_test_dat_10k'));%'yale_test_data_',num2str(TST_N),'.mat'));
tst_ftrs = logistic(bsxfun(@plus,testd*model.W,model.hidB));
save(ftr_name,'tst_ftrs','-append');
clear testd tst_ftrs;
clear model;


load(strcat(DAT_DIR,'mnist_train_dat_10k.mat'));%'yale_train_data_',num2str(TRN_N),'.mat'));
trn_ftrs_r  = rule_inference_(R,traind);
save(ftr_name,'trn_ftrs_r','-append');
clear traind trn_ftrs_r;
%vld_ftrs_r  = rule_inference_(R,alp_vld_dat);
load(strcat(DAT_DIR,'mnist_test_dat_10k.mat'));%'yale_test_data_',num2str(TST_N),'.mat'));
tst_ftrs_r  = rule_inference_(R,testd);
save(ftr_name,'tst_ftrs_r','-append');
clear testd tst_ftrs_r;
clear R;
else
    clear model R;
end

for c = [1 5]
for g = [0.001 0.05 1]
load(ftr_name,'trn_ftrs');
model = svmtrain(trainl, trn_ftrs,['-c ' num2str(c) ' -g ' num2str(g)]);
clear trn_ftrs;
%[~, accuracy, ~] = svmpredict(alp_trn_lab, trn_ftrs, model);
%train_acc = accuracy(1);

%[~, accuracy, ~] = svmpredict(alp_vld_lab, vld_ftrs, model);
%eval_acc = accuracy(1);
load(ftr_name,'tst_ftrs');
[~, accuracy, ~] = svmpredict(testl, tst_ftrs, model);
tst_acc = accuracy(1);
clear tst_ftrs;

load(ftr_name,'trn_ftrs_r');
model = svmtrain(trainl, trn_ftrs_r,['-c ' num2str(c) ' -g ' num2str(g)]);
clear trn_ftrs_r;
%[~, accuracy, ~] = svmpredict(alp_trn_lab, trn_ftrs_r, model);
%train_acc_r = accuracy(1);
%[~, accuracy, ~] = svmpredict(alp_vld_lab, vld_ftrs_r, model);
%eval_acc_r = accuracy(1);
load(ftr_name,'tst_ftrs_r');
[~, accuracy, ~] = svmpredict(testl, tst_ftrs_r, model);
tst_acc_r = accuracy(1);

%logging('grid_s_4_cmp.mat',[i1 i2 i3 c g eval_acc eval_acc_r]);
logging(strcat(EXP_DIR,'grid_s_10k_h',num2str(conf.hidNum),'_cmp_test.mat'),[trial i1 i2 i3 c g tst_acc tst_acc_r]);

%clearvars -except trial i1 i2 i3 ftr_name c g;
end % end c
end % end g
end % end p
end % end lambda
end % end learning rate
end % end trial
end