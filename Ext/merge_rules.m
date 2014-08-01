function [TW R] = merge_rules(Ts,Rs,K)
aR  = [];
aRc = [];
aT  = [];
aTc = [];
for i=1:size(Rs,2)
    aR  = [aR;Rs(i).r];
    aRc = [aRc Rs(i).c];
    
    TW  = [TW bsxfun(@times,Ts(i).r(1:end-1),Ts(i).c)];
end

[~,inx] = sort(aRc,'descend');
inx = inx(1:K);
TW = TW(:,inx);
end

