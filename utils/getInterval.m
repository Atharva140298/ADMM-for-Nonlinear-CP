function interval = getInterval(X, p, mode)
% getInterval returns [a,b] defining the censoring interval
% X    : data (vector, matrix, tensor)
% p    : proportion to KEEP (e.g., 0.8 keeps 80% of data)
% mode : 'symmetric' - censors both tails equally (default)
%        'left'      - censors left tail only (detection limit scenario)
%        'right'     - censors right tail only

    if nargin < 3
        mode = 'symmetric';
    end

    x = X(:);   % vectorize

    switch mode
        case 'symmetric'
            alpha = (1 - p) / 2;
            a = quantile(x, alpha);       % bottom tail cut
            b = quantile(x, 1 - alpha);   % top tail cut

        case 'left'
            a = quantile(x, 1 - p);       % bottom (1-p) proportion censored
            b = max(x);                   % no upper censoring

        case 'right'
            a = min(x);                   % no lower censoring
            b = quantile(x, p);           % top (1-p) proportion censored
    end

    interval = [a, b];
end