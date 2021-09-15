clear all
load result.mat
sub=result_p-result_t;
add=result_p+result_t;
total=[];
total(1)=sqrt(sum(sum(sub.^2))/sum(sum(result_t.^2)));
total(2)=sqrt(mean(mean(sub.^2)));
total(3)=mean(mean(abs(sub)./add));
for i=1:308
    pre_temp=result_p(i,:);
    true_temp=result_t(i,:);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
end