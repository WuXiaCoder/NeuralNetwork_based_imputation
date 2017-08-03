%% 随机缺失样本属性
%输入参数
%   data为样本总体  矩阵为：样本*属性
%   nn为缺失率，取值0-1
%输出参数
%   data
function [data,miss_sample]=random_miss_cpy(data,nn)
    [N,S]=size(data);
    S=1;                             %实验初期只选取一列的数据进行随机缺失操作
    Num=floor(N*S*nn);               %Num为缺失属性的个数
    miss_sample=zeros(1,Num);        %用于记录缺少样本在整个数据集合的下标 
    m=randi([1,N],1,1);              %随机选取[1,N]范围内的一个整数，即选取缺失样本
    n=randi([1,1],1,1);              %随机选取[1,S]范围内的一个整数，即选取缺失属性，实验初期只选取一列的数据进行随机缺失操作
    for(i=1:1:Num)
          while(isnan(data(m,n)))    %while循环，知道(m,n)处的样本为不缺失数据，再赋值nan，代表该位置为缺失数据
             m=randi([1,N],1,1);
             n=randi([1,1],1,1);
          end
          data(m,n)=nan;             %样本m的n属性赋值nan，表示该属性缺失
          miss_sample(i)=m;          %记录有缺失的样本下标
         
    end        
end