function R = extract_rbm(model,vis_dir,cpen)
% Extract rules from rbm consisting of labels
% T: top rules for labels
% R: lower rules for intermediate proposition
RW = [model.W' model.hidB'];
R.c = zeros(1,size(RW,1));
R.r = zeros(size(RW));

for i=1:size(RW,1)
    [R.c(i) R.r(i,:)] = extract_rule(RW(i,:));
end

if exist('cpen','var') && cpen
    R.c = mean(abs(RW),2)';
end
    
if ~isempty(vis_dir)
    T.c = 1;
    T.r = ones(1,size(R.r,1));
    R.r = R.r(:,1:end-1);
    save_img_rules(T,R,vis_dir);
end
end