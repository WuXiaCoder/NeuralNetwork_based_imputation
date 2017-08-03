%% �ô���Ϊ���ڴ��������BP������ȱʧ�����
function data_miss=miss()
%% ��ջ�������
clear;
clc;
%% ������ȡ
load data.txt
data=data(:,1:size(data,2)-1);              %���һ�������Ƿ������ݣ��޳�
%% �������ȱʧ���� 
n=0.10; 
[data_miss,miss_sample_index]=random_miss(data,n);

%% ����ÿһ������ 
flag=0;
for i=1:1:size(miss_sample_index,1)
  
    test_sample_index=find(miss_sample_index(i,:)==1);
    if(test_sample_index~=0)               % �����ǰ���Դ���ȷȱʧ������������������ǰ�����²�����ȱʧ���� �������һ�����Խ��в���
       %% �ҳ�ѵ��������ǣ�����������ǣ� ѵ�����Ա�ǣ�ȱʧ���Ա��
        train_sample_index=find(miss_sample_index(i,:)==0);
        attr_train=1:size(data,2);        
        attr_train(find(attr_train==i))='';   %ѵ������Ϊ��i֮�������
        attr_miss=i;
       %% ѡȡѵ��������ȱʧ����
        input_total=data(:,attr_train);
        output_total=data(:,attr_miss);
        input_train=data(train_sample_index,attr_train);
        output_train=data(train_sample_index,attr_miss);
        input_test=data(test_sample_index,attr_train);
        output_test=data(test_sample_index,attr_miss);
        %% �������ݹ�һ��
        [inputn,inputps]=mapminmax(input_train');  %mapminmax�����н��й�һ��
        inputn=inputn';
        %% ѵ��������
        if(flag==0)
            net=BP_train(inputn,output_train,0,flag);
        else
            net=BP_train(inputn,output_train,net_weight,flag);
            flag=1;
        end
        net_weight.w1=net.w1;
        net_weight.w2=net.w2;
        net_weight.b1=net.b1;
        net_weight.b2=net.b2;
        %% �������1�� �����������룬�����
        inputn_test=mapminmax('apply',input_test',inputps);
        inputn_test=inputn_test';
        fore=BP_test(net,inputn_test);
        MAPE=100/size(fore,2).*sum(abs((fore-output_test')./output_test'))
        plot_test(size(input_test,1),fore(1,:),output_test(:,1),0);%NΪ������  foreΪʵ�������  output_testΪ���������

       %% �������2��ȫ���������룬�����
        %inputn_total=mapminmax('apply',input_total',inputps);
        %inputn_total=inputn_total';
        %fore2=BP_test(net,inputn_total);
        %plot_test(size(data,1),fore2(1,:),output_total(:,1),1);%NΪ������  foreΪʵ�������  output_testΪ���������
       
     end
end
%% �����������������е����ͼ
    %figure;
   % plot(net.E)
end






%% ȷ�����������������
function [Ss,m]=ass_attr(mis_attr,data,r)
    %mis_attrΪȱʧ����  dataΪȫ���������ݣ���������+ȱʧ����  rΪ��ȡ���������ĸ���
    S=size(data,2); 
    xx=zeros(S,2);
    for(i=1:1:S)
        t=corrcoef(data(:,mis_attr),data(:,i));
        if(isnan(t(1,2))|| i==mis_attr)
            xx(i,:)=[0,0];
        else
             xx(i,:)=[t(1,2),abs(t(1,2))];
        end
    end
   [Ss,m]=sortrows(xx,-2);
   Ss=Ss(1:r,:);
   m=m(1:r,:);
   
end

%% ���ƿ�������ͼ
function  plot_test(N,actual_data,expect_data,format);
    %NΪ���ݸ��� actual_dataΪʵ���������  expect_dataΪ�����������
    %format��ʾ��ʽ 0Ϊ��  1Ϊ��
    output_fore=zeros(1,N);
    for i=1:N
        output_fore(i)=actual_data(:,i);
    end
    %BP����Ԥ�����
    error=output_fore'-expect_data;
    %����Ԥ�����������ʵ����������ķ���ͼ
    figure;
    if(format==0)
    plot(expect_data','b*')
    hold on
    plot(output_fore,'r+')
    legend('�������','ʵ�����')
    else
    plot(expect_data','b')
    hold on
    plot(output_fore,'r')
    legend('�������','ʵ�����')
    end

    %�������ͼ
    %figure;
    %plot(error)
    title('BP����������','fontsize',12)
    xlabel('�����ź�','fontsize',12)
    ylabel('�������','fontsize',12)

    %��ȷ��
    zero_index=find(error<=0.05&error>=-0.05);            
    k=length(zero_index);       
    rightridio=k/N;
    disp('��ȷ��')
    disp(rightridio);
    if(N==1)
    disp('ȱʧ���� ʵ�����');
    disp(output_fore);
    disp('ȱʧ���� �������');
    disp(expect_data);
    end    
end
