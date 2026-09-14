function A = hals_nnls(M, G, A, n_inner)
% min_{A>=0} tr(A'A G) - 2 tr(A'M), by column-wise HALS (warm-started)
    r   = size(G,1);
    eps_div = 1e-12;
    for it = 1:n_inner
        for j = 1:r
            gjj = max(G(j,j), eps_div);
            A(:,j) = max( A(:,j) + (M(:,j) - A*G(:,j))/gjj , 0 );
        end
    end
end