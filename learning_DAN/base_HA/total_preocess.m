clear all
load y.mat
load h.mat
y_true=y;
y_pre=h;

result_t=[];
result_p=[];
for i=1:308
    true=y_true(:,i,:);
    pre=y_pre(:,i,:);
    a=reshape(true,[1003,6]);
    b=reshape(pre,[1003,6]);
    pre=[];true=[];
    pre(1)=b(1,1);
    pre(2)=b(1,2);
    pre(3)=b(1,3);
    true(1)=a(1,1);
    true(2)=a(1,2);
    true(3)=a(1,3);
    n=4;
    for j=1:1002
        pre(n)=(b(j,4)+b(j+1,1))/2;
        true(n)=(a(j,4)+a(j+1,1))/2;
        n=n+1;
        pre(n)=(b(j,5)+b(j+1,2))/2;
        true(n)=(a(j,5)+a(j+1,2))/2;
        n=n+1;
        pre(n)=(b(j,6)+b(j+1,3))/2;
        true(n)=(a(j,6)+a(j+1,3))/2;
        n=n+1;
    end
    pre(n)=b(1003,4);
    pre(n+1)=b(1003,5);
    pre(n+2)=b(1003,6);
    true(n)=a(1003,4);
    true(n+1)=a(1003,5);
    true(n+2)=a(1003,6);
    result_t(i,:)=true;
    result_p(i,:)=pre;
    i
end
save('result.mat','result_t','result_p');