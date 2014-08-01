function [T R] = extract_rbm_l(model,vis_dir,N,cpen)
% Extract rules from rbm consisting of labels
% T: top rules for labels
% R: lower rules for intermediate proposition
TW  = [model.U model.labB'];
T.c = zeros(1,size(TW,1));
T.r = zeros(size(TW));
for i=1:size(TW,1)
    [T.c(i) T.r(i,:)] = extract_rule(TW(i,:));
end

RW = [model.W' model.hidB'];
R.c = zeros(1,size(RW,1));
R.r = zeros(size(RW));

for i=1:size(RW,1)
    [R.c(i) R.r(i,:)] = extract_rule(RW(i,:));
end

if exist('N','var') && N>0
    T = impruning(T,R,TW,N);
    %T = impruning_1(T,R);
end
if exist('cpen','var') && cpen>0
    T.c = mean(abs(TW),2)';
    R.c = mean(abs(RW),2)';
end
if ~isempty(vis_dir)
    T.r = T.r(:,1:end-1);
    R.r = R.r(:,1:end-1);
    save_img_rules(T,R,vis_dir);
end
end