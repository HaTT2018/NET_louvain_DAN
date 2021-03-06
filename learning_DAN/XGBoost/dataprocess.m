clear
load ../RF/V.mat
V2=V2';
pre=[];
x_train=[];
y_train=[];
x_test=[];
y_test=[];
for i=1:25
    temp1=[];
    temp2=[];
    temp3=[];
    temp4=[];
    for j=1:144*7
      temp1=[temp1;V2(i,j:j+3)];
      temp2=[temp2;V2(i,j+4)];
      temp3=[temp3;V2(i,j+144*7:j+144*7+3)];
      temp4=[temp4;V2(i,j+144*7+4)];
    end
    x_train=[x_train;temp1];
    y_train=[y_train;temp2];
    x_test=[x_test;temp3];
    y_test=[y_test;temp4];
end

x_train = x_train(1:144*3*25, :);
y_train = y_train(1:144*3*25, :);

save('data2_6.mat','x_train','y_train','x_test','y_test');