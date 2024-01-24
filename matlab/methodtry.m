[tx,str]=xlsread('Problem_C_Data_Wordle.xlsx');
time=tx(:,1);
Altitude=tx(:,3);
Altitude2=smoothdata(Altitude);
SA=cumsum(Altitude2);
words=str(:,3);
fid1=fopen('E:\code\matlab\words.txt','w',','n',utf-8'); 
for i=1:length(words)
    sstr=words{i};
    %开始写入数据，hn为需要保存的数据
    count=fprintf(fid1,' %s, \n',strjoin(cellstr(sstr))); 
    %关闭文件
   
end
 fclose(fid1); 
% figure(1)
% subplot(2,1,1)
% plot(time,SA,'r');
% title('sum-results-time');
% xlabel('time');
% legend('sum_results');
% figure(2)
% subplot(2,1,1)
% plot(time,Altitude,'r');
% title('results-time');
% xlabel('time');
% legend('results');
% figure(3)
% subplot(2,1,1)
% plot(time,Altitude2,'r');
% title('results-time');
% xlabel('time');
% legend('results');
