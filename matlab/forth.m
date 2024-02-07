sheet=xlsread("E:\code\meisai\source\2024\C\4_cancha.csv");
real=sheet(:,1);
predict=sheet(:,2);
predict(1)=0
predict=predict
cha=predict-real;
cha = fillmissing(cha,'constant',0);
size(cha);
h1 = histogram(cha,21,'Normalization','pdf','FaceColor','	#F0F8FF','EdgeAlpha',0.5);
hold on
% 
pd=fitdist(cha,'Normal')


x_values=-1.5:0.1:1.5;
y=pdf(pd,x_values);
xlabel('residual');
ylabel('probability');
plot(x_values,y,'linewidth',1.3,'Color','#6495ED');

mu = 0.330854;
sigma = 0.191365;
lb=mu-1.96*sigma;
ub=mu+1.96*sigma;


figure(2)
x = 7:1:117;                     
y = predict(1:111);
y=y';
xconf = [x x(end:-1:1)] ;      
yconf = [y-lb y(end:-1:1)-ub];
p = fill(xconf,yconf,'r');
hold on
p.FaceColor = [1 0.8 0.8];      
p.EdgeColor = 'none';           
x=1:1:117;

plot(x(7:117),y(1:111),'Color','#8B3A3A')
hold on
plot(x(7:117),real(1:111),'yellow','linewidth',1.6)
hold on

z=zeros(1,301);

plot(x(1:54),z(1:54),'Color','#CDC9C9','linewidth',10)
hold on
plot(x(55:86),z(55:86),'Color','	#FFF5EE','linewidth',10)
hold on
plot(x(87:117),z(87:117),'Color','	#CDC9C9','linewidth',10)
hold on
lg=legend('95% confidence interval.','predict','real','player1 won set','player2 won set');
xlabel('points');
ylabel('Hp');
hold off

% histfit(cha)