sheet=xlsread("E:\code\meisai\source\2024\C\forth_mon.csv");
RA=sheet(:,51);
RB=sheet(:,52);
size(RA)
x=1:1:53;
figure(1)

z=zeros(1,301);
z=z+120
C2='#FFEFD5';
C1='	#F0F5EE';
w=250;
plot(x(1:4),z(1:4),'Color',C1,'linewidth',w)
hold on
plot(x(5:9),z(5:9),'Color',C1,'linewidth',w)
hold on
plot(x(10:16),z(10:16),'Color',C2,'linewidth',w);
hold on
plot(x(17:21),z(17:21),'Color',C2,'linewidth',w)
hold on
plot(x(22:25),z(22:25),'Color',C1,'linewidth',w)
hold on
plot(x(26:30),z(26:30),'Color',C2,'linewidth',w)
hold on
plot(x(31:41),z(31:41),'Color',C1,'linewidth',w)
hold on
plot(x(42:48),z(42:48),'Color',C1,'linewidth',w)
hold on
plot(x(49:53),z(49:53),'Color',C1,'linewidth',w)
hold on

plot(x,RA(1:53),'Color','#87CEFA','linewidth',3)
hold on
plot(x,RB(1:53),'Color','#fdb933','linewidth',3)
hold on
% 
% plot(32,RA(32),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(35,RA(35),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(37,RA(37),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(39,RA(39),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(41,RA(41),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(42,RA(42),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)


xlabel('points');
ylabel('momentum');
legend('player1_ won game','','','player2_ won game','','','','','','RA','RB')
hold off