function [T,Rs] = generalize_rule(T,Rs,cs)
%T.c
T.c = cs(end);
for i=1:size(Rs,2)
    Rs(i).c = cs(i);
end
end

