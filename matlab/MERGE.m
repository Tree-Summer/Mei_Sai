%%排序
function [A] = MERGE(A,p,q,r)%A是数组，p<=q<=r
L=A(p:q);%前半段
R=A(q+1:r);%后半段
L(end+1)=inf;%前半段哨兵   哨兵的用处：避免某个半段值已用完，无法做比较。
R(end+1)=inf;%后半段哨兵
i=1;%序号
j=1;%序号
for k=p:r %进行排序
    if L(i)<=R(j) %判断前后两个半段是谁先放
        A(k)=L(i);
        i=i+1;
    else
        A(k)=R(j);
        j=j+1;
    end
end
end