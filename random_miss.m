%% ���ȱʧ��������
%�������
%   dataΪ��������  ����Ϊ������*����
%   nnΪȱʧ�ʣ�ȡֵ0-1
%�������
%   data
function [data,miss_sample]=random_miss_cpy(data,nn)
    [N,S]=size(data);
    S=1;                             %ʵ�����ֻѡȡһ�е����ݽ������ȱʧ����
    Num=floor(N*S*nn);               %NumΪȱʧ���Եĸ���
    miss_sample=zeros(1,Num);        %���ڼ�¼ȱ���������������ݼ��ϵ��±� 
    m=randi([1,N],1,1);              %���ѡȡ[1,N]��Χ�ڵ�һ����������ѡȡȱʧ����
    n=randi([1,1],1,1);              %���ѡȡ[1,S]��Χ�ڵ�һ����������ѡȡȱʧ���ԣ�ʵ�����ֻѡȡһ�е����ݽ������ȱʧ����
    for(i=1:1:Num)
          while(isnan(data(m,n)))    %whileѭ����֪��(m,n)��������Ϊ��ȱʧ���ݣ��ٸ�ֵnan�������λ��Ϊȱʧ����
             m=randi([1,N],1,1);
             n=randi([1,1],1,1);
          end
          data(m,n)=nan;             %����m��n���Ը�ֵnan����ʾ������ȱʧ
          miss_sample(i)=m;          %��¼��ȱʧ�������±�
         
    end        
end