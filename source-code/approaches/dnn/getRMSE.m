function rmse = getRMSE(predcitVaule,trueValue)

rmse=sqrt( sum( (trueValue(:)-predcitVaule(:)).^2) / numel(trueValue));
end
