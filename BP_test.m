function [fore]=BP_test_cpy(net,input_test)
%�������
%   netΪ֮ǰѵ���õ�����BP������
%   input_testΪ����������  ����Ϊ������*����
%�������
%   foreΪ��������յ����
%% �������1�� �����������룬�����
    test_N=size(input_test,1);
    fore=zeros(net.outnum,test_N);
    for i=1:test_N  %1500
        %���������
        I=zeros(1,net.midnum);        %���ز�����
        for j=1:1:net.midnum
            I(j)=input_test(i,:)*net.w1(j,:)'+net.b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
       %��������
         fore(:,i)=net.w2'*Iout'+net.b2;
    end 
   
end