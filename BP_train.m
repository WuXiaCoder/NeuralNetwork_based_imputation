%% �ô���Ϊ���ڴ������������BP������ѵ������
function net=BP_train(input_train,output_train)
%�������
%   input_trainΪѵ������  ����Ϊ������*����
%   output_trainΪ�ο���� ����Ϊ����*����
%�������
%   ���Ȩ�أ���ֵ
%% ����ṹ��ʼ��
train_N=size(input_train,1);
innum=size(input_train,2);
midnum=8;
outnum=size(output_train,2);
 
%Ȩֵ��ʼ��
w1=rands(midnum,innum);   %����㵽���ز�Ȩ��
b1=rands(midnum,1);       %���ز���Ԫ��ֵ
w2=rands(midnum,outnum);  %���ز㵽�����Ȩ��
b2=rands(outnum,1);       %�������Ԫ��ֵ

w2_1=w2;w2_2=w2_1;
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

xite=0.1;                 %ѧϰ��
alfa=0.1;                 %������
loopNumber=2000;          %��������
I=zeros(1,midnum);        %���ز�����
Iout=zeros(1,midnum);     %���ز����
FI=zeros(1,midnum);       %���ز����
O=zeros(1,outnum);        %���������
Y=zeros(1,outnum);       %��������


dw1=zeros(innum,midnum);  %����㵽���ز��Ȩ��һ�׵���
db1=zeros(1,midnum);      %���ز���Ԫ��ֵ��һ�׵���
error_goal=0;             %������ֹ���
E=zeros(1,loopNumber);    %ÿ�ε�����������1
Error=zeros(1,loopNumber);%ÿ�ε�����������2
%% ����ѵ��
for mii=1:loopNumber
    Error(mii)=0;
    E(mii)=0;
    for i=1:1:train_N
       %% ����Ԥ����� 
        x=input_train(i,:);
        y=output_train(i,:);
        if(~isnan(y))   %�������iΪ��ȱʧ���ݣ����ø�����ѵ��������
        % ���������
        for j=1:1:midnum
            I(j)=input_train(i,:)*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        % ��������
        yn=w2'*Iout'+b2;
        
       %% Ȩֵ��ֵ����
        %�������
        dE=(1/2)*sum((output_train(i,:)'-yn).*(output_train(i,:)'-yn)); %��������ʧ����
        E(mii)= E(mii)+dE;    %���ε��������е����֮��
        Error(mii)=Error(mii)+sum(abs(output_train(i,:)'-yn));
        %�������
        e=output_train(i,:)'-yn;         
        %����Ȩֵ�仯��
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
        disp('��Ŀǰ�ĵ��������ڲ��ܱƽ�������������Ӵ��������')          
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
