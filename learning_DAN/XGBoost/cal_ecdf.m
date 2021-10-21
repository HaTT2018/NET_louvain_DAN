clear all
y_pre=[];
y_true=[];
for i=5:5
    eval( ['load',' result',num2str(i),'.mat'] );
    y_pre=[y_pre;pre];
    y_true=[y_true;true];
end
pre=[];true=[];
y_pre=y_pre';y_true=y_true';
for i=1:25
        pre(i,:)=reshape(y_pre(i*1008-1007:i*1008,:),[1,1008]);
        true(i,:)=reshape(y_true(i*1008-1007:i*1008,:),[1,1008]);
end
sub=pre-true;
add=pre+true;
total=[];
total(1)=sqrt(sum(sum(sub.^2))/sum(sum(true.^2)));
total(2)=sqrt(mean(mean(sub.^2)));
total(3)=mean(mean(abs(sub)./add));
for i=1:25
    pre_temp=pre(i,:);
    true_temp=true(i,:);
    sub_temp=pre_temp-true_temp;
    sum_temp=pre_temp+true_temp;
    nrmse(i)=sqrt(sum(sum(sub_temp.^2))/sum(sum(true_temp.^2)));
    rmse(i)=sqrt(mean(mean(sub_temp.^2)));
    smape(i)=mean(mean(abs(sub_temp)./sum_temp));
    mape(i)=mean(abs(sub_temp)./abs(true_temp));
    mae(i)=mean(abs(sub_temp));
end
disp(mean(mape(isinf(mape)==0)))
disp(mean(smape))
disp(mean(nrmse))
disp(mean(mae))
save(char(['C:\Users\10169\Documents\Github\NET_louvain_DAN\model\base_XGBoost_mape=', num2str(mean(mape)),'.mat']))