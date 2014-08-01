function [T,Rs] = prune_rule(T,Rs,pNums)
%prune rules
%sontran2014
prv_inx = 1:(size(Rs(1).r,2)-1);
for i=1:size(pNums,2)    
    [~,inx]  = sort(Rs(i).c,'descend');
    inx = inx(1:pNums(i));
    Rs(i).c = Rs(i).c(inx);
    Rs(i).r = Rs(i).r(inx,[prv_inx end]);
    prv_inx = inx;
end
T.r = T.r(:,[prv_inx end]);
end