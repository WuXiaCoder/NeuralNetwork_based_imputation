%% 该代码为基于带动量项的BP神经网络缺失数据填补
function data_miss=miss()
%% 清空环境变量
clear;
clc;
%% 数据提取
load data.txt
data=data(:,1:size(data,2)-1);              %最后一列属性是分类数据，剔除
%% 随机产生缺失数据 
n=0.10; 
[data_miss,miss_sample_index]=random_miss(data,n);

%% 遍历每一个属性 
flag=0;
for i=1:1:size(miss_sample_index,1)
  
    test_sample_index=find(miss_sample_index(i,:)==1);
    if(test_sample_index~=0)               % 如果当前属性存在确缺失样本，则继续；如果当前属性下不存在缺失样本 则进入下一个属性进行查找
       %% 找出训练样本标记，测试样本标记， 训练属性标记，缺失属性标记
        train_sample_index=find(miss_sample_index(i,:)==0);
        attr_train=1:size(data,2);        
        attr_train(find(attr_train==i))='';   %训练属性为除i之外的属性
        attr_miss=i;
       %% 选取训练样本与缺失样本
        input_total=data(:,attr_train);
        output_total=data(:,attr_miss);
        input_train=data(train_sample_index,attr_train);
        output_train=data(train_sample_index,attr_miss);
        input_test=data(test_sample_index,attr_train);
        output_test=data(test_sample_index,attr_miss);
        %% 输入数据归一化
        [inputn,inputps]=mapminmax(input_train');  %mapminmax按照行进行归一化
        inputn=inputn';
        %% 训练神经网络
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
        %% 结果分析1： 测试样本输入，数据填补
        inputn_test=mapminmax('apply',input_test',inputps);
        inputn_test=inputn_test';
        fore=BP_test(net,inputn_test);
        MAPE=100/size(fore,2).*sum(abs((fore-output_test')./output_test'))
        plot_test(size(input_test,1),fore(1,:),output_test(:,1),0);%N为样本量  fore为实际输出量  output_test为期望输出量

       %% 结果分析2：全部样本输入，数据填补
        %inputn_total=mapminmax('apply',input_total',inputps);
        %inputn_total=inputn_total';
        %fore2=BP_test(net,inputn_total);
        %plot_test(size(data,1),fore2(1,:),output_total(:,1),1);%N为样本量  fore为实际输出量  output_test为期望输出量
       
     end
end
%% 画出整个迭代过程中的误差图
    %figure;
   % plot(net.E)
end






%% 确定神经网络的输入属性
function [Ss,m]=ass_attr(mis_attr,data,r)
    %mis_attr为缺失属性  data为全部样本数据：完整数据+缺失数据  r为提取输入样本的个数
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

%% 绘制可视曲线图
function  plot_test(N,actual_data,expect_data,format);
    %N为数据个数 actual_data为实际输出数据  expect_data为期望输出个数
    %format表示格式 0为点  1为线
    output_fore=zeros(1,N);
    for i=1:N
        output_fore(i)=actual_data(:,i);
    end
    %BP网络预测误差
    error=output_fore'-expect_data;
    %画出预测语音种类和实际语音种类的分类图
    figure;
    if(format==0)
    plot(expect_data','b*')
    hold on
    plot(output_fore,'r+')
    legend('期望输出','实际输出')
    else
    plot(expect_data','b')
    hold on
    plot(output_fore,'r')
    legend('期望输出','实际输出')
    end

    %画出误差图
    %figure;
    %plot(error)
    title('BP网络分类误差','fontsize',12)
    xlabel('语音信号','fontsize',12)
    ylabel('分类误差','fontsize',12)

    %正确率
    zero_index=find(error<=0.05&error>=-0.05);            
    k=length(zero_index);       
    rightridio=k/N;
    disp('正确率')
    disp(rightridio);
    if(N==1)
    disp('缺失样本 实际输出');
    disp(output_fore);
    disp('缺失样本 期望输出');
    disp(expect_data);
    end    
end
