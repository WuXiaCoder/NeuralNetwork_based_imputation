%% 该代码为基于带动量项的三层BP神经网络训练过程
function net=BP_train(input_train,output_train)
%输入参数
%   input_train为训练样本  矩阵为：样本*属性
%   output_train为参考输出 矩阵为样本*属性
%输出参数
%   输出权重，阈值
%% 网络结构初始化
train_N=size(input_train,1);
innum=size(input_train,2);
midnum=8;
outnum=size(output_train,2);
 
%权值初始化
w1=rands(midnum,innum);   %输入层到隐藏层权重
b1=rands(midnum,1);       %隐藏层神经元阈值
w2=rands(midnum,outnum);  %隐藏层到输出层权重
b2=rands(outnum,1);       %输出层神经元阈值

w2_1=w2;w2_2=w2_1;
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

xite=0.1;                 %学习率
alfa=0.1;                 %动量率
loopNumber=2000;          %迭代次数
I=zeros(1,midnum);        %隐藏层输入
Iout=zeros(1,midnum);     %隐藏层输出
FI=zeros(1,midnum);       %隐藏层输出
O=zeros(1,outnum);        %输出层输入
Y=zeros(1,outnum);       %输出层输出


dw1=zeros(innum,midnum);  %输入层到隐藏层的权重一阶导数
db1=zeros(1,midnum);      %隐藏层神经元阈值的一阶导数
error_goal=0;             %迭代终止误差
E=zeros(1,loopNumber);    %每次迭代的输出误差1
Error=zeros(1,loopNumber);%每次迭代的输出误差2
%% 网络训练
for mii=1:loopNumber
    Error(mii)=0;
    E(mii)=0;
    for i=1:1:train_N
       %% 网络预测输出 
        x=input_train(i,:);
        y=output_train(i,:);
        if(~isnan(y))   %如果样本i为非缺失数据，则用该样本训练神经网络
        % 隐含层输出
        for j=1:1:midnum
            I(j)=input_train(i,:)*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        % 输出层输出
        yn=w2'*Iout'+b2;
        
       %% 权值阀值修正
        %迭代误差
        dE=(1/2)*sum((output_train(i,:)'-yn).*(output_train(i,:)'-yn)); %输出层的损失函数
        E(mii)= E(mii)+dE;    %本次迭代过程中的误差之和
        Error(mii)=Error(mii)+sum(abs(output_train(i,:)'-yn));
        %计算误差
        e=output_train(i,:)'-yn;         
        %计算权值变化率
        dw2=e*Iout;
        db2=e';       
        for j=1:1:midnum
            S=1/(1+exp(-I(j)));
            FI(j)=S*(1-S);
        end   
    
         for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(w2(j,:)*e);
                db1(j)=FI(j)*(w2(j,:)*e);
            end
        end
           
        w1=w1_1+xite*dw1'+alfa*(w1_1-w1_2);
        b1=b1_1+xite*db1'+alfa*(b1_1-b1_2);
        w2=w2_1+xite*dw2'+alfa*(w2_1-w2_2);
        b2=b2_1+xite*db2'+alfa*(b2_1-b2_2);
        
        w1_2=w1_1;w1_1=w1;
        w2_2=w2_1;w2_1=w2;
        b1_2=b1_1;b1_1=b1;
        b2_2=b2_1;b2_1=b2;
        end
    end
    if(E(mii)<=error_goal)
       break;  
    elseif(mii==loopNumber)  
        disp('在目前的迭代次数内不能逼近所给函数，请加大迭代次数')          
    end 
 end
net.w1=w1;
net.w2=w2;
net.b1=b1;
net.b2=b2;
net.E=E;
net.innum=innum;
net.midnum=midnum;
net.outnum=outnum;
 

end
