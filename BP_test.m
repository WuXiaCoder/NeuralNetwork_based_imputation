function [fore]=BP_test(net,input_test)
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
         x=input_test(i,:);
         I=zeros(1,net.midnum);        %���ز�����
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
       %��������
         fore(:,i)=net.w2'*Iout'+net.b2;
      
    end
   
end