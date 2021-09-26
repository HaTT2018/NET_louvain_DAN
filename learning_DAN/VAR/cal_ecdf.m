clear all
load result.mat
pre=reshape(permute(pre,[2,1,3]),[308,18*6]);
true=reshape(permute(true,[2,1,3]),[308,18*6]);
sub_temp=pre-true;
sum_temp=pre+true;
total(1)=sqrt(mean(mean(sub_temp.^2)));
total(2)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true.^2)));
total(3)=mean(mean(abs(sub_temp)./sum_temp));
for i=1:308
    pre_temp=pre(i,:);
    true_temp=true(i,:);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
end