clear all
y_pre=[];
y_true=[];
for i=1:1
    eval( ['load',' result',num2str(i),'.mat'] );
    pre_temp=pre;
    true_temp=true;
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
end
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
	sqrt(mean(mean((y_pre-y_true).^2)));