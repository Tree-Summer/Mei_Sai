sheet=xlsread("E:\code\meisai\source\2024\C\cancha.csv");
real=sheet(:,2);
predict=sheet(:,3);
cha=predict-real;
cha = fillmissing(cha,'constant',0);
size(cha);
h1 = histogram(cha,21,'Normalization','pdf','FaceColor','	#F0F8FF','EdgeAlpha',0.5);
hold on
% 
pd=fitdist(cha,'Normal')
mu = 1.44938;
sigma = 2.75218;
lb=mu-1.96*sigma;
ub=mu+1.96*sigma;

% x_values=-15:0.1:15;
% y=pdf(pd,x_values);
% xlabel('residual');
% ylabel('probability');
% plot(x_values,y,'linewidth',1.3,'Color','#6495ED');

figure(2)
x = 7:1:307;                     
y = predict(1:301);
y=y';
xconf = [x x(end:-1:1)] ;      
yconf = [y-lb y(end:-1:1)-ub];
p = fill(xconf,yconf,'red');
p.FaceColor = [1 0.8 0.8];      
p.EdgeColor = 'none';           

hold on
plot(x(7:301),y( 1:301-6),'Color','#8B3A3A')
hold on
plot(x,real(1:301),'yellow','linewidth',1.6)
z=zeros(1,308);
plot(x(1:64),z(1:64),'Color','#CDC9C9','linewidth',10)
hold on
plot(x(65:157),z(65:157),'Color','	#FFF5EE','linewidth',10)
hold on
plot(x(158:213),z(158:213),'Color','	#CDC9C9','linewidth',10)
hold on
plot(x(214:301),z(214:301),'Color','	#CDC9C9','linewidth',10)
lg=legend('95% confidence interval.','predict','real','player1 won set','player2 won set');
xlabel('points');
ylabel('Hp');
hold off

% histfit(cha)