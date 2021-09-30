clear all
load result.mat
pre = reshape(pre,[12*308,6]);
true = reshape(true,[12*308,6]);
for i=1:308*12
    for j=1:6
        if abs(pre(i,j))>10000
            if i==1
                pre(i,j)=pre(i+1,j);
            else if i==308*12
                pre(i,j)=pre(i-1,j);
            else
                pre(i,j)=(pre(i-1,j)+pre(i+1,j))/2;
                end
            end
        end
    end
end
for i=1:6
    pre_temp=pre(:,i);
    true_temp=true(:,i);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt((sum(sub_temp.^2)/sum(true_temp.^2)));
    rmse(i)=sqrt(mean(sub_temp.^2));
    smape(i)=mean(abs(sub_temp)./sum_temp);
end