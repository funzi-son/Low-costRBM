function output = rule_inference(Rs,T,input,no_bias)
% Inference using confidence-based rule
% sontran2013
iii = input';
for i=1:size(Rs,2)     
    if exist('no_bias','var') && no_bias
        iii = logistic(bsxfun(@times,Rs(i).r(:,1:end-1),Rs(i).c')*iii);     
    else    
        iii = logistic(bsxfun(@times,Rs(i).r,Rs(i).c')*[iii;ones(1,size(input,1))]);
    end
end
%iii

if exist('no_bias','var') && no_bias
    [~,output] = max(bsxfun(@times,T.r(:,1:end-1),T.c')*iii);
    %bsxfun(@times,T.r(:,1:end-1),T.c')*iii
else    
    [~,output] = max(bsxfun(@times,T.r,T.c')*[iii;ones(1,size(input,1))]);
end
output = output';
end

