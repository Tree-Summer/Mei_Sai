[sheet,str]=xlsread('E:\code\meisai\source\Problem_C_Data_Wordle_5.xlsx');
dist=sheet(:,17);
dist=sort(dist);
theta=linespace(0,1,1000)
p_x_given_theta=theta.^x.*(1-theta).^x;
p_theta_givenx=BayesRule(p_theta,p_x_given_theta);


% pd=fitdist(dist,'Normal');
% figure(1)
% h(1).FaceColor = [.8 .8 1];
% h(2).Color = [.2 .2 .2];
% disp(pd)
% x_values=0.4:0.01:0.8;
% y=pdf(pd,x_values);
% 
% plot(x_values,y);
% histfit(dist)
%正态分布画图


% %sheet是读入的excel表格的数字，str是单词
% trydistribution=sheet(:,5:11);
% Sum=sum(trydistribution,2);
% 
% 
% data1 = betarnd(4,3,100,1);
% data1=MERGE_SORT(data1,1,length(data1));
% plot(data1)
% hold on;
% [p,ci] = betafit(data1,0.01);
% x=0:0.01:1;
% y=betapdf(x,p(1,1),p(1,2));
% plot(x,y);
% xlabel('x');
% ylabel('y');


% for i=1:length (trydistribution)
%     u=0;
%     cnt=0;
%     for j=1:7
%         next=cnt+trydistribution(i,j);
%         for hh=1:trydistribution(i,j)
%             r=rand(1,1);
%             u=u+1;
%             c(u)=(j-1+r)/7;
%         end
%         % plot([(j-1)/7,j/7],[cnt/Sum(i,1),next/Sum(i,1)]);
%         % hold on;%保持在图形中
%         cnt=next;
%     end
%     [p,ci] = betafit(c(1,:),0.05);
%     ab(i,1)=p(1);
%     ab(i,2)=p(2);
%     ab(i,3)=p(1)/(p(1)+p(2));
    % x=0:0.01:1;
    % y=betacdf(x,p(1,1),p(1,2))
    % plot(x,y);
    % xlabel('x');
    % ylabel('y');
    % xlabel('x');
    % ylabel('y');
    % figure(3)
    % plot(c);
% end
