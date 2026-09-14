function Xpre = preprocess_centerscale_new(X, flag_scale, flag_center)
% X is a numeric array: subjects x time x metabolites
% Center across subjects, then scale within metabolites mode using RMS

Xpre = X;

% Center across subjects mode
if flag_center
    temp = reshape(Xpre, size(Xpre,1), []);
    colmean = nanmean(temp, 1);
    temp_centered = temp - repmat(colmean, size(temp,1), 1);
    Xpre = reshape(temp_centered, size(X));
end

% Scale within metabolites mode
if flag_scale
    XX = zeros(size(Xpre));
    for j = 1:size(Xpre,3)
        temp = squeeze(Xpre(:,:,j));
        rmsval = sqrt(nanmean(temp(:).^2));

        if rmsval > 0
            XX(:,:,j) = temp / rmsval;
        else
            XX(:,:,j) = temp;
        end
    end
    Xpre = XX;
end