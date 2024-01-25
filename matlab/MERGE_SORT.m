%% 递归
function [A] = MERGE_SORT(A,p,r)%A是数组，p在前，q在后
if p<r %判断是否到底，相等即只有一位数，不再分解
    q=floor((p+r)/2);%中间值，首尾相加除以2，再向下取整；
    A=MERGE_SORT(A,p,q);%前半段
    A=MERGE_SORT(A,q+1,r);%后半段
    A=MERGE(A,p,q,r);%排序
end
end

