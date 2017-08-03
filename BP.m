for i=1:20 %样本个数  
    xx(i)=2*pi*(i-1)/20;  
    d(i)=0.5*(1+cos(xx(i)));  
end  
load data.txt
data=data(:,1:size(data,2)-1);
[N,S]=size(data);
d=data(:,S);
xx=data(:,1:S-1);
n=length(xx);%样本个数  
p=6; %隐层个数  
%% 初始化权值
w=rand(p,S-1);       %6*2 输入层到隐藏层    输入层神经元为2，隐藏层神经元为6  
wk=rand(1,p+1);    %1*7 隐藏层到输出层    隐藏层神经元为6，输出层神经元为1   p+1的1为阈值
max_epoch=10000;   %最大训练次数  
error_goal=0.002;  %均方误差 最终的训练误差  
q=0.09;%学习速率  
a(p+1)=-1;   %此-1为阈值 相当于隐藏层多了个值为-1 
  
%training  
%此训练网络采取1-6-1的形式，即一个输入，6个隐层，1个输出  
for epoch=1:max_epoch  
    e=0;  
    for i=1:N %样本个数  
        x=xx(i,:);     %输入层为2 第二个样本为-1
        neto=0;  
        for j=1:p   
            neti(j)=w(j,1)*x(1)+w(j,2)*x(2)+w(j,3)*x(3);  %加权求和
            a(j)=1/(1+exp(-neti(j)));      %隐层的激活函数采取s函数，f(x)=1/(1+exp(-x))   a(j)为隐藏层输出值 在[0,1]之间
            neto=neto+wk(j)*a(j);  
        end            
        neto=neto+wk(p+1)*(-1);   %阈值加权求和 
        y(i)=neto; %输出层的激活函数采取线性函数,f(x)=x  
        de=(1/2)*(d(i)-y(i))*(d(i)-y(i));   %输出层的损失函数
        e=de+e;     %损失值求和  
        dwk=q*(d(i)-y(i))*a;   
        for k=1:p  %输出层6个连接权  
            dw(k,1:S-1)=q*(d(i)-y(i))*wk(k)*a(k)*(1-a(k))*x;         
        end     
        wk=wk+dwk; %从隐层到输出层权值的更新  
        w=w+dw; %从输入层到隐层的权值的更新      
    end   
    error(epoch)=e;  
    m(epoch)=epoch;      
    if(e<error_goal)              
       break;  
    elseif(epoch==max_epoch)  
        disp('在目前的迭代次数内不能逼近所给函数，请加大迭代次数')          
    end   
end  
%simulation  
for i=1:N %样本个数  
    x=[xx(i);-1];    
    neto=0;  
    for j=1:p  
        neti(j)=w(j,1)*x(1)+w(j,2)*x(2);  
        a(j)=1/(1+exp(-neti(j)));  
        neto=neto+wk(j)*a(j);  
    end    
    neto=neto+wk(p+1)*(-1);  
    y(i)=neto; %线性函数  
end   
  
%plot  
figure(1)  
plot(m,error)  
xlabel('迭代次数')  
ylabel('均方误差')  
title('BP算法的学习曲线')  
figure(2)  
plot(xx,d)  
hold on  
plot(xx,y,'r')  
legend('蓝线是目标曲线','红线是逼近曲线') 
figure;
plot(error);