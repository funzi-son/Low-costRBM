function get_mnist_rs()
DAT_FILE= '/home/funzi/Documents/Experiments/DBN_CPRESS/MNIST/grid_s_10k_h1000_cmp_test.mat';

load(DAT_FILE);
trials = unique(data(:,1));
max(data(:,7))
max(data(:,8))
dat = [];
mod_res = [];
rul_res = [];
for tr=trials'
    mod_res = [mod_res,data(find(data(:,1)==tr),7)];
    rul_res = [rul_res,data(find(data(:,1)==tr),8)];
    if  isempty(dat)
        dat = data(find(data(:,1)==tr),2:end);        
    else
        dat = dat + data(find(data(:,1)==tr),2:end);
    end
end

%dat = dat/size(trials,1);
[acc_mod,inx] = max(mean(mod_res,2));
std_mod = std(mod_res(inx,:));
[acc_rul,inx] = max(mean(rul_res,2));
std_rul = std(rul_res(inx,:));

fprintf('Acc model: %.5f +- %.5f v.s Acc rule %.5f +- %.5f\n',acc_mod,std_mod,acc_rul,std_rul);
%[~,inx] = max(max(mod_res,[],2));

%mod_res(inx,:)
end

