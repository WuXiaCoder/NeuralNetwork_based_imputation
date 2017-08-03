%% �ô����ҳ�ָ��������q������(ָ���������������ݼ�������Ϊȱʧ����)
function [filter_sample_index,dist]=q_neighbours(pos,data,q)
%   posΪָ�������������������ϵ�λ�� ������
%   dataΪ���������� ����Ϊ����*���� n*s
%   qΪq������
%�������
%   filter_data Ϊѡ������
%   distΪ q��������ָ������֮��ľ���
 %% ���־��빫ʽ��������pos����������֮��ľ���
   P=data(pos,:);
   [n,total]=size(data);                 % nΪ��������  totalΪ�������Ը���
   if(q>=n-1)
        error('�޷��ҵ�q���ڣ�q��ʾ�Ľ��ڸ����Ѵ�����');
   end 
   data_cpy=data;
   I1=zeros(1, total)+1;                 % I1���ڱ������posȱʧ���Ե�λ�� Ĭ��Ϊ1�ľ���
   index1=find(isnan(P));                %�ҳ�����pos��ȱʧ����
   I1(index1)=0;                         % �������pos��ĳһ����ΪNaN������I1��Ӧλ�ø�ֵ0
   P(index1)=0;                          % ����nan�޷�������ѧ���㣬�˴���0����
   for i=1:1:n
      I2=zeros(1, total)+1;              % I2���ڱ������iȱʧ���Ե�λ��  Ĭ��Ϊ1�ľ���
      index2=find(isnan(data(i,:)));     %�ҳ�����i��ȱʧ����
      I2(index2)=0;                      % �������i��ĳһ����ΪNaN������I2��Ӧλ�ø�ֵ0
      data(i,index2)=0;                  % ����nan�޷�������ѧ���㣬�˴���0����
      count=total-sum((I1&I2)==0);       %countΪ����pos������i�е��������Եĸ���
      dis(1,i)= sqrt(total/count*sum((I1.*I2.*(data(i,:)-P).^2)'));                 %���־��빫ʽ
   end
 %% ����������pos�ľ���Զ����ѡȡq������
   filter_data=[ data_cpy,dis'];
   [filter_data,filter_sample_index]=sortrows(filter_data,size(filter_data,2));     %���վ����Զ������
   dist= filter_data(1:q+1,size(filter_data,2));
   filter_sample_index=filter_sample_index(1:q+1,:); 
   index=find(filter_sample_index==pos); %���˵�ָ������P������ľ���
   filter_sample_index(index)='';        %���˵�ָ������P������ľ���
   dist(index)='';                       %���˵�ָ������P������ľ���
end  
