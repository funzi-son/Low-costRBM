function [c rule] = extract_rule(w_vec)
% Extract rules from weight vectors
% sontran2013
c = 0;
o_c = -1;
www = abs(w_vec);
rule = (w_vec>0)*2-1;
while c~=o_c
    o_c = c;
    %update c    
    c = mean(www(find(rule~=0,1,'first'):end));
    %update rule
    rule(find(2*www<=c)) = 0;
end
end

