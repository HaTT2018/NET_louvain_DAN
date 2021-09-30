clear all
load data.mat
pre=[];
for k=1:12
    pred=[];
    for i=1:308
        mdl=arima(1,1,1);
        t=reshape(x(k,i,:), [12,1]);
        estmdl=estimate(mdl,t,'Display','off');
        temp=forecast(estmdl,6)'+mean(t);
        pred=[pred;temp];
    end
    pre(k,:,:)=pred;
    k
end
true=y(1:12,:,:);
save('result.mat','pre','true');