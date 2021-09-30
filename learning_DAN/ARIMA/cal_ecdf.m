clear all
load result.mat
pre=reshape(permute(pre,[2,1,3]),[308,12*6]);
true=reshape(permute(true,[2,1,3]),[308,12*6]);
for i=1:308
    for j=1:72
        if abs(pre(i,j))>10000
            if j==1
                pre(i,j)=pre(i,j+1);
            else if j==72
                pre(i,j)=pre(i,j-1);
            else
                pre(i,j)=(pre(i,j-1)+pre(i,j+1))/2;
                end
            end
        end
    end
end
sub=pre-true;
add=pre+true;
total=[];
total(1)=sqrt(sum(sum(sub.^2))/sum(sum(true.^2)));
total(2)=sqrt(mean(mean(sub.^2)));
total(3)=mean(mean(abs(sub)./add));
for i=1:308
    pre_temp=pre(i,:);
    true_temp=true(i,:);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
end