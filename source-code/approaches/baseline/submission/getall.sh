

for i in {2..54..4}
do 

R --slave -f bartArgs.r --args f$(($i+1)).csv   &
R --slave -f bartArgs.r --args f$(($i+2)).csv   &
R --slave -f bartArgs.r --args f$(($i+3)).csv   &
R --slave -f bartArgs.r --args f$(($i+4)).csv   &

wait

done

