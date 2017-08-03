function [fore]=BP_test_cpy(net,input_test)
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
        I=zeros(1,net.midnum);        %隐藏层输入
        for j=1:1:net.midnum
            I(j)=input_test(i,:)*net.w1(j,:)'+net.b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
       %输出层输出
         fore(:,i)=net.w2'*Iout'+net.b2;
    end 
   
end