function [objective,residual] = compute_objective_residual(X, Xhat, T, model, error_measure, param)
switch model

    case 'ReLU'
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = norm(X - max(0, Xhat), 'fro') / normX;
            case 'L1'
                X_approx = max(0, Xhat);
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = max(0, Xhat);
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                % Construct the estimation matrix X_est
                X_est = mu_X * ones(size(X));
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                % Compute the residual betweeen Theta and Xhat=WH
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
        end
    case 'CSF'
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = norm(X - (Xhat).^2, 'fro') / normX;
            case 'L1'
                X_approx = (Xhat).^2;
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = (Xhat).^2;
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                % Construct the estimation matrix X_est
                X_est = mu_X * ones(size(X));
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
        end
    case 'MinMax'
        a = param.a;
        b = param.b;
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = norm(X - min(b, max(a, Xhat)), 'fro') / normX;
            case 'L1'
                X_approx = min(b, max(a, Xhat));
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = min(b, max(a, Xhat));
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X / mu_X) - X + mu_X);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
        end
    case 'Modulus'
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = norm(X - abs(Xhat), 'fro') / normX;
            case 'L1'
                X_approx = abs(Xhat);
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = abs(Xhat);
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                X_est = mu_X * ones(size(X));
                T = max(eps,T);
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
        end

    case 'Exponential'
        switch error_measure
            case 'Frobenius'
                normX = norm(X,'fro');
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
                objective = norm(X - exp(Xhat), 'fro') / normX;
            case 'KL divergence'
                X_approx = exp(Xhat);
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                X_est = mu_X * ones(size(X));
                T = max(eps,T);
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - Xhat, 'fro') / norm(T,'fro');
        end

end
