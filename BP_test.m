function [fore]=BP_test(net,input_test)
%输入参数
%   net为之前训练好的三层BP神经网络
%   input_test为测试练样本  矩阵为：样本*属性
%输出参数
%   fore为输出层最终的输出
%% 结果分析1： 测试样本输入，数据填补
    test_N=size(input_test,1);
    fore=zeros(net.outnum,test_N);
    for i=1:test_N  %1500
        %隐含层输出
         x=input_test(i,:);
         I=zeros(1,net.midnum);        %隐藏层输入
        for j=1:1:net.midnum
            for m=1:1:net.innum
                if(isnan(x(m)))
                    I(j)=I(j)+net.xw1_average(m,j)+net.b1(j);
                else
                    I(j)=I(j)+x(m)*net.w1(j,m);   
                end
            end
             I(j)=I(j)+net.b1(j);
             Iout(j)=1/(1+exp(-I(j)));
        end     
       %输出层输出
         fore(:,i)=net.w2'*Iout'+net.b2;
      
    end
   
end