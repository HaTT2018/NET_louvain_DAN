clear all
load result.mat
pre = reshape(pre,[18*308,6]);
true = reshape(true,[18*308,6]);
for i=1:6
    pre_temp=pre(:,i);
    true_temp=true(:,i);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt((sum(sub_temp.^2)/sum(true_temp.^2)));
    rmse(i)=sqrt(mean(sub_temp.^2));
    smape(i)=mean(abs(sub_temp)./sum_temp);
end