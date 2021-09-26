clear
load data.mat
pre=[];
for k=1:18
    pred=[];
    for i=1:25
        Spec = vgxset('n', 1, 'nAR', 1, 'Constant', true);
        EstSpec = vgxvarx(Spec, reshape(x(k,i,:),[12,1]));
        temp = vgxpred(EstSpec, 6);
        pred=[pred,temp];
    end
    pre(k,:,:)=pred';
    k
end
true=y(1:18,:,:);
save('result.mat', 'pre', 'true');