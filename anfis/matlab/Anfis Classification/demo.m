%% ANFIS MULTIMODEL Classification
%%
%
% * Developer Er.Abbas Manthiri S
% * EMail abbasmanthiribe@gmail.com
% * Date 24-03-2017
%
%% Initialize
clc
clear
close all
warning off all
%% Load Data
%% WC Dataset day 6-9
M = csvread('w20.csv');
TrainData=M(1:end-144,1:end-1);
TrainClass=M(1:end-144,end);
%% Test data day 10
TestData=M(end-143:end,1:end-1);
TestClass=M(end-143:end,end);
%% Classification  Demo 1
epoch_n = 200;
dispOpt = zeros(1,4);
numMFs = 2;
inmftype= 'gbellmf';
outmftype= 'linear';
split_range=2;
Model=ANFIS.train(TrainData,TrainClass,split_range,numMFs,inmftype,outmftype,dispOpt,epoch_n);
disp('Model')
disp(Model)
Result=ANFIS.classify(Model,TestData);
%Performance Calculation
Rvalue=@(a,b)(1-abs((sum((b-a).^2)/sum(a.^2))));
RMSE=@(a,b)(abs(sqrt(sum((b-a).^2)/length(a))));
MAPE=@(a,b)(abs(sum(sqrt((b-a).^2)*100./a)/length(a)));
fprintf('Anfis  RMSE    MAPE\n')
r=Rvalue(TestClass,Result);
rmse=RMSE(TestClass,Result);
mape=MAPE(TestClass,Result);
fprintf('Anfis  %.4f\t%.4f\n',rmse,mape);
%% Display
% disp('TestClass Predicted ')
% disp([TrainClass(1:n),Result(1:n)])
s = size(TestClass);
x=linspace(1, 144, 144);
plot(x,TestClass, x, Result);
