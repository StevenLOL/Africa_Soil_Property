%
%steven
%

clear all
addpath('../');
load;
sprev=rng(0,'v5uniform');
addpath('./data');
addpath('./data');
addpath('./DBN');
addpath('./util');
addpath('./NN');
global hasgpu;
hasgpu=0;


%training=[training(:,1:3578),training(:,end-4:end)];
%sortedtest=sortedtest(:,1:3578);

%training=[training(:,1:2654),training(:,2669+1:end)];
%sortedtest=[sortedtest(:,1:2654),sortedtest(:,2669+1:end)];

training(:,3594)=(training(:,3594)+1)/2;
sortedtest(:,3594)=(sortedtest(:,3594)+1)/2;

[traincount dim]=size(training);



avgRMSE={};
avgPrediction={};
layersize=400;

for loopSmooth=1:101

    rmse5=[];
    predictOutput=zeros(727,5);

    if hasgpu==1
        predictOutput=gpuArray(predictOutput);
    end


    for tid=1:5

        
        randomIndex=randperm(traincount);
        train_all=training(randomIndex,:);
        testsize=int32(0.01*traincount);
        train_x=train_all(1:end-testsize,1:end-5);
        full_train_y=train_all(1:end-testsize,end-5+1:end);

        eval_x=train_all(end-testsize+1:end,1:end-5);
        full_test_y=train_all(end-testsize+1:end,end-5+1:end);
 
        [train_x, mu, sigma] = zscore(train_x);
        eval_x = normalize(eval_x, mu, sigma);
        sortedtestNormalize=normalize(sortedtest,mu,sigma);
 
        train_y=full_train_y(:,tid);
        test_y=full_test_y(:,tid);
        nn = nnsetup([3594 layersize layersize 1]);      
        nn.activation_function = 'sigm';
        nn.output              = 'linear';      %  linear is usual choice for regression problems
        nn.learningRate        = 0.001;         %  Linear output can be sensitive to learning rate
        nn.momentum            = 0.95;
        
        if hasgpu==1
            for l=1:length(nn.W)
                nn.W{l}=gpuArray(nn.W{l});
                nn.vW{l}=gpuArray(nn.vW{l});
            end
        end
        
        %opts.plot=1;
        %opts.validation=1;
        opts.numepochs = 50;   %  Number of full sweeps through data
        opts.batchsize = 100;   %  Take a mean gradient step over this many samples
        [nn, L] = nntrain(nn, train_x, train_y, opts,eval_x,test_y);
        % nnoutput calculates the predicted regression values
        predictions = nnoutput( nn, eval_x );
        RMSE = sqrt( sum( (test_y(:)-predictions(:)).^2) / numel(test_y) );

        %1) get best par via above then train on full training set
        %2)

        
        predictions = nnoutput(nn, sortedtestNormalize);

        predictOutput(:,tid)=predictions;
        rmse5(end+1)=RMSE;

    end
    
    avgPrediction{loopSmooth}=predictOutput;
    avgRMSE{loopSmooth}=rmse5;
    
    if sum([1,5,20,35,50,100]==loopSmooth)>0

        avgToWrite=mean(reshape(cell2mat(avgPrediction), [727, 5, loopSmooth]), 3);
        avgRMSEFileName=mean(mean(reshape(cell2mat(avgRMSE), [1, 5, loopSmooth]), 3));
        filename=sprintf('dnn_full_%d_%d_mean%f.csv',layersize,loopSmooth,avgRMSEFileName);
        csvwrite(filename,avgToWrite);     
        system(['python2.7 ./rewritecsv.py ./' filename  ' ../data/sample_submission.csv']);
       

    end

end








