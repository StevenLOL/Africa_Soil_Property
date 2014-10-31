%averageModel

plan_submit=[1,5,20,35,50];
csvfiles=dir('*.csv');
scores={};
system('python2.7 cleanBART.py');
for i=1:size(csvfiles)
 csvfile=csvfiles(i);
 scores{i}=csvread(csvfile.name,1,1);
 if sum(i==plan_submit)>0
     filename=sprintf('BART_AVG_%d.benchmark',i);
     avgToWrite=mean(reshape(cell2mat(scores), [727, 5, i]), 3);
     csvwrite(filename,avgToWrite);
     system(['python2.7 rewritecsv.py ./' filename  ' ../../data/sample_submission.csv']);

 end
 
end

