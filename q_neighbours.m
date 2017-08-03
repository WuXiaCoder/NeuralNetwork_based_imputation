%% 该代码找出指定样本的q个近邻(指定样本和完整数据集都可以为缺失数据)
function [filter_sample_index,dist]=q_neighbours(pos,data,q)
%   pos为指定样本在整个样本集合的位置 整数型
%   data为整个样本集 矩阵为样本*属性 n*s
%   q为q个近邻
%输出参数
%   filter_data 为选中样本
%   dist为 q个近邻与指定样本之间的距离
 %% 部分距离公式计算样本pos和其他样本之间的距离
   P=data(pos,:);
   [n,total]=size(data);                 % n为样本个数  total为样本属性个数
   if(q>=n-1)
        error('无法找到q近邻，q表示的近邻个数已达上限');
   end 
   data_cpy=data;
   I1=zeros(1, total)+1;                 % I1用于标记样本pos缺失属性的位置 默认为1的矩阵
   index1=find(isnan(P));                %找出样本pos的缺失属性
   I1(index1)=0;                         % 如果样本pos的某一属性为NaN，则在I1相应位置赋值0
   P(index1)=0;                          % 由于nan无法进行数学计算，此处用0代替
   for i=1:1:n
      I2=zeros(1, total)+1;              % I2用于标记样本i缺失属性的位置  默认为1的矩阵
      index2=find(isnan(data(i,:)));     %找出样本i的缺失属性
      I2(index2)=0;                      % 如果样本i的某一属性为NaN，则在I2相应位置赋值0
      data(i,index2)=0;                  % 由于nan无法进行数学计算，此处用0代替
      count=total-sum((I1&I2)==0);       %count为样本pos和样本i中的完整属性的个数
      dis(1,i)= sqrt(total/count*sum((I1.*I2.*(data(i,:)-P).^2)'));                 %部分距离公式
   end
 %% 按照与样本pos的距离远近，选取q个样本
   filter_data=[ data_cpy,dis'];
   [filter_data,filter_sample_index]=sortrows(filter_data,size(filter_data,2));     %按照距离的远近排序
   dist= filter_data(1:q+1,size(filter_data,2));
   filter_sample_index=filter_sample_index(1:q+1,:); 
   index=find(filter_sample_index==pos); %过滤到指定样本P到自身的距离
   filter_sample_index(index)='';        %过滤到指定样本P到自身的距离
   dist(index)='';                       %过滤到指定样本P到自身的距离
end  
