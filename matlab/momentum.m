sheet=xlsread("E:\code\meisai\source\2024\C\momentum.xlsx");
RA=sheet(:,35)
RB=sheet(:,36)
size(RA)
h=63
l=92;
x=64:1:155;
figure(1)

z=zeros(1,301);
C2='#FFEFD5';
C1='	#F0F5EE';
w=250;
plot(x(64-h:67-h),z(64:67),'Color',C2,'linewidth',w)
hold on
plot(x(68-h:75-h),z(68:75),'Color',C2,'linewidth',w)
hold on
plot(x(76-h:78-h),z(76:78),'Color',C2,'linewidth',w);
hold on
plot(x(79-h:87-h),z(79:87),'Color',C1,'linewidth',w)
hold on
plot(x(88-h:97-h),z(88:97),'Color',C2,'linewidth',w)
hold on
plot(x(98-h:105-h),z(98:105),'Color',C1,'linewidth',w)
hold on
plot(x(106-h:117-h),z(106:117),'Color',C1,'linewidth',w)
hold on
plot(x(118-h:121-h),z(118:121),'Color',C1,'linewidth',w)
hold on
plot(x(122-h:126-h),z(122:126),'Color',C2,'linewidth',w)
hold on
plot(x(127-h:132-h),z(127:132),'Color',C1,'linewidth',w)
hold on
plot(x(133-h:137-h),z(133:137),'Color',C2,'linewidth',w)
hold on
plot(x(138-h:141-h),z(138:141),'Color',C1,'linewidth',w)
hold on
plot(x(142-h:155-h),z(142:155),'Color',C2,'linewidth',w)
hold on


plot(x,RA,'Color','#87CEFA','linewidth',3)
hold on
plot(x,RB,'Color','#fdb933','linewidth',3)
hold on


% plot(88,RB(88-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(89,RB(89-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(93,RB(93-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(95,RB(95-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(96,RB(96-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
% plot(97,RB(97-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)

plot(86,RA(86-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(91,RA(91-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(92,RA(92-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(93,RA(93-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(94,RA(94-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(95,RA(95-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(96,RA(96-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(111,RA(111-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(112,RA(112-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(113,RA(113-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(114,RA(114-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(115,RA(115-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(116,RA(116-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(119,RA(119-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(121,RA(121-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(123,RA(123-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(124,RA(124-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
plot(125,RA(125-63),'-p','MarkerFaceColor','#f36c21','MarkerSize',10)
xlabel('points');
ylabel('momentum');
legend('','','','player1_ won game','player2_ won game','','','','','','','','','RA','RB','predict_ turning point')
hold off