
[sheet,str]=xlsread('E:\code\meisai\source\Problem_C_Data_Wordle_5.xlsx');
y=sheet(:,3);
x=sheet(:,1);
x=(x-201)/358;
f = fittype('r1*((x-r5)^r2)*exp(-(x-r5)/r3) + r4');
disp('haha');
start_point=[1,0.1,1,0.1,1];
fit_options=fitoptions('Method','NonlinearLeastSquares','StartPoint',start_point);
disp('qwq')
[fit_result,gof]=fit(x,y,fit_options);
disp('qaq')
disp(fit_result);
plot(fot_result,x,y);
v=r1*((x-r5)^r2)*exp(-(x-r5)/r3) + r4;