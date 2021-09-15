clear all
load y.mat
load h.mat
for i=1:6
    pre_temp=reshape(y(i,:,:),[308,1003]);
    true_temp=reshape(h(i,:,:),[308,1003]);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
end