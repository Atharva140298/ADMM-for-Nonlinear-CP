function [T] = update_T(X,rho,T,Xhat,lambda, model, error_measure, param)
%% This file contains all the updates for the matrix T, for all different
%% cases of nonlinearities and loss-functions. The user can add more nonlinear
%% models and loss functions easily after solving one-dimensional subproblems
%% for each entry of T
switch model
    %% Case 1: ReLU with Frobenius norm
    case 'ReLU'
        switch error_measure
            case 'Frobenius'
                Y=lambda - rho*Xhat;
                p= find(Y>X);
                q= find(Y<0);
                r1= find(Y>0);
                r2= find(Y<X);
                r= intersect(r1,r2);
                t1 = (rho*Xhat - lambda + X )./(rho + 1);
                t2 = (rho*Xhat - lambda) ./ (rho);
                T(p)=t2(p);
                T(q)=t1(q);
                g1 = 1/2 * (X - t1).^2 + t1.*lambda + ((rho/2)*(t1 - Xhat).^2);
                g2 = 1/2 * (X).^2 + t2.*lambda + ((rho/2)*(t2 - Xhat).^2);
                ind1 = find(g1(r)<g2(r)) ;
                ind2 = find(g1(r)>=g2(r));
                T(r(ind1)) = t1(r(ind1));
                T(r(ind2)) = t2(r(ind2));
                %% ReLU with L1 norm
            case 'L1'

                tmin = Xhat - lambda./rho;
                t1star = min(tmin, 0);
                t21 = (rho*Xhat - lambda -1 )./(rho);
                t22 = (rho*Xhat - lambda +1 )./(rho);
                t23 = X;
                tmax  = max(min(t22,t23), t21);
                t2star =max(tmax, 0);
                g1  = X +  t1star.*lambda + ((rho/2)*(t1star - Xhat).^2);
                g2  = abs(X - t2star) + t2star.*lambda + ((rho/2)*(t2star - Xhat).^2);
                T = t1star;
                ind2better = g2 < g1; T(ind2better) = t2star(ind2better);

                %% ReLU with KL-divergence
            case 'KL divergence'
                tmax = (rho*Xhat - lambda - 1)./rho;
                t1star= max(tmax,0);
                tmin = (rho*Xhat - lambda)./rho;
                t2star=min(tmin,0);
                g1  = t1star +  t1star.*lambda + ((rho/2)*(t1star - Xhat).^2);
                g2  = t2star.*lambda + ((rho/2)*(t2star - Xhat).^2);
                T = t1star;
                ind2better = g2 < g1; T(ind2better) = t2star(ind2better);
                p = find(X>0);
                D = sqrt((lambda - rho*Xhat + 1).^2 + 4*rho*X);
                T(p) = (rho*Xhat(p) - lambda(p) - 1 + D(p))./(2*rho);
        end
        %% CSF with frobenius norm
    case 'CSF'
        [m,n,p] = size(X);
        switch error_measure
            case 'Frobenius'
                for i=1:m
                    for j=1:n
                        for k=1:p
                            a =4;
                            c =rho - 4 * X(i,j);
                            d = lambda(i,j) - rho * Xhat(i,j);
                            T(i,j,k) = cardanoatharva(a,c, d);
                        end
                    end
                end
                %% CSF with L1 norm
            case 'L1'
                if rho>2
                    tmin = (rho*Xhat - lambda)./(rho-2);
                    sqrtX=sqrt(X);
                    t1star= min(sqrtX, max(-sqrtX,tmin));
                    tmax = (rho*Xhat - lambda)./(rho+2);
                    t2star = max(tmax,sqrtX);
                    t3star = min(tmax, -sqrtX);
                    g1 = X - t1star.^2 + t1star.*lambda + (rho/2)*(t1star - Xhat).^2 ;
                    g2 = t2star.^2 - X + t2star.*lambda + (rho/2)*(t2star - Xhat).^2 ;
                    g3 = t3star.^2 - X + t3star.*lambda + (rho/2)*(t3star - Xhat).^2;
                    T = t1star;
                    ind2better = g2 < g1 & g2 <= g3; T(ind2better) = t2star(ind2better);
                    ind3better = g3 < g1 & g3 < g2; T(ind3better) = t3star(ind3better);
                else
                    t1 = sqrt(X);
                    t2 = -sqrt(X);
                    g1 = abs(X - (t1).^2) + lambda.*t1 + ((rho/2).*(t1 - Xhat).^2);
                    g2 = abs(X - (t2).^2) + lambda.*t2 + ((rho/2).*(t2 - Xhat).^2);
                    T = t1;
                    ind2better = g2<g1 ; T(ind2better) = t2(ind2better);
                end

                %% CSF with KL-divergence
            case 'KL divergence'
                q  = find(X==0);
                D  = sqrt(((lambda - rho.*Xhat).^2) + (8*X.*(rho+2)));
                t1 = (rho.*Xhat - lambda + D)./(2*(rho+2));
                t2 = (rho.*Xhat - lambda - D)./(2*(rho+2));
                g1 = t1.^2 - X.* log(t1.^2) + t1.*lambda + ((rho/2).*(t1-Xhat).^2);
                g2 = t2.^2 - X.* log(t2.^2) + t2.*lambda + ((rho/2).*(t2-Xhat).^2);
                T = t1;
                ind2better = g2 < g1; T(ind2better) = t2(ind2better);
                T(q) = (rho.*Xhat(q) - lambda(q))./(rho+2);
        end
        %% MinMax with Frobenius norm
    case 'MinMax'
        a = param.a;
        b = param.b;
        switch error_measure
            case 'Frobenius'
                tmin = Xhat - lambda./rho;
                t1star = min(tmin, a);
                t3star = max(tmin, b);
                t_med = (X + rho*Xhat - lambda) ./ (rho+1);
                t2star = min(max(t_med, a), b);
                g1 = 0.5*(X - a).^2 + t1star.*lambda + (rho/2)*(t1star - Xhat).^2 ;
                g2 = 0.5*(X - t2star).^2 + t2star.*lambda + (rho/2)*(t2star - Xhat).^2 ;
                g3 = 0.5*(X - b).^2 + t3star.*lambda + (rho/2)*(t3star - Xhat).^2;
                T = t1star;
                ind2better = g2 < g1 & g2 <= g3; T(ind2better) = t2star(ind2better);
                ind3better = g3 < g1 & g3 < g2; T(ind3better) = t3star(ind3better);
                %% MinMax with L1-norm
            case 'L1'

                tmin = Xhat - lambda./rho;
                t1star = min(tmin, a);
                t3star = max(tmin, b);
                t_med = max(min(tmin + 1./rho, X), tmin - 1./rho);
                t2star = min(max(t_med, a), b);
                g1 = abs(X - a) + t1star.*lambda + (rho/2)*(t1star - Xhat).^2 ;
                g2 = abs(X - t2star) + t2star.*lambda + (rho/2)*(t2star - Xhat).^2 ;
                g3 = abs(X - b) + t3star.*lambda + (rho/2)*(t3star - Xhat).^2;
                T = t1star;
                ind2better = g2 < g1 & g2 <= g3; T(ind2better) = t2star(ind2better);
                ind3better = g3 < g1 & g3 < g2; T(ind3better) = t3star(ind3better);
                %% MinMax with KL-divergence
            case 'KL divergence'
                p = find(X>0);
                q = find(X==0);
                %% For X>0
                tmin = Xhat - lambda./rho;
                t1star = min(tmin, a);
                t3star = max(tmin, b);
                D   = sqrt((lambda - rho*Xhat + 1).^2 + 4*rho*X);
                t_med = (rho*Xhat - lambda - 1 + D)./(2*rho);
                t2star = min(max(t_med, a), b);
                g1 = a - X.*log(a) + t1star.*lambda + ((rho/2) .* (t1star-Xhat).^2);
                g2 =t2star - X.* log(t2star) + t2star.*lambda + ((rho/2) .* (t2star-Xhat).^2);
                g3 = b - X.*log(b) + t3star.*lambda + ((rho/2) .* (t3star-Xhat).^2);
                T(p) = t1star(p);
                ind2better = g2(p) < g1(p) & g2(p) <= g3(p); T(ind2better) = t2star(ind2better);
                ind3better = g3(p) < g1(p) & g3(p) < g2(p); T(ind3better) = t3star(ind3better);
                %% For X=0
                tmin0 = Xhat - lambda./rho;
                t1star0 = min(tmin0, a);
                t3star0 = max(tmin0, b);
                t_med0 = (rho*Xhat - lambda - 1) ./ rho;
                t2star0 = min(max(t_med0, a), b);
                g10 = a + t1star0.*lambda + ((rho/2) .* (t1star0-Xhat).^2);
                g20 =t2star0  + t2star0.*lambda + ((rho/2) .* (t2star0-Xhat).^2);
                g30 = b + t3star0.*lambda + ((rho/2) .* (t3star0-Xhat).^2);
                T(q) = t1star0(q);
                ind2better0 = g20(q) < g10(q) & g20(q) <= g30(q); T(ind2better0) = t2star0(ind2better0);
                ind3better0 = g30(q) < g10(q) & g30(q) < g20(q); T(ind3better0) = t3star0(ind3better0);
        end
        %% Modulus with Frobenius norm
    case 'Modulus'
        switch error_measure
            case 'Frobenius'
                Y  = rho*Xhat - lambda;
                p  = find(Y>X);
                q  = find(Y<(-X));
                r1 = find(Y>(-X));
                r2 = find(Y<X);
                r  = intersect(r1,r2);
                t1 = (rho*Xhat - lambda + X )./(rho + 1);
                t2 = (rho*Xhat - lambda - X )./(rho + 1);
                T(p)=t1(p);
                T(q)=t2(q);
                g1 = 1/2 * (X - t1).^2 + t1.*lambda + ((rho/2)*(t1 - Xhat).^2);
                g2 = 1/2 * (X + t2).^2 + t2.*lambda + ((rho/2)*(t2 - Xhat).^2);
                ind1 = find(g1(r)<g2(r)) ;
                ind2 = find(g1(r)>=g2(r));
                T(r(ind1)) = t1(r(ind1));
                T(r(ind2)) = t2(r(ind2));
                %% Modulus with L1-norm
            case 'L1'
                Y  =rho*Xhat - lambda;
                p  = find(Y>1);
                q  = find(Y<(-1));
                r1 = find(Y>(-1));
                r2 = find(Y<1);
                r  = intersect(r1,r2);
                t11 = (rho*Xhat - lambda +1)./(rho);
                t22 = (rho*Xhat - lambda -1 )./(rho);
                t1  = max(min(t11,X), t22);
                t2  = max(min(t11,-X), t22);
                g1  = abs(X -t1) +  t1.*lambda + ((rho/2)*(t1 - Xhat).^2);
                g2  = abs(X + t2) + t2.*lambda + ((rho/2)*(t2 - Xhat).^2);
                T(p)=t1(p);
                T(q)=t2(q);
                ind1 = find(g1(r)<g2(r)) ;
                ind2 = find(g1(r)>=g2(r));
                T(r(ind1)) = t1(r(ind1));
                T(r(ind2)) = t2(r(ind2));
                %% Modulus with KL-divergence
            case 'KL divergence'
                Y  = rho.*Xhat - lambda;
                p  = (X > 0);

                D1 = sqrt( max((1 - Y).^2 + 4*rho.*X, 0) );
                t1 = (Y - 1 + D1) ./ (2*rho);

                D2 = sqrt( max((1 + Y).^2 + 4*rho.*X, 0) );
                t2 = (1 + Y - D2) ./ (2*rho);

                f1 = t1;
                f2 = -t2;

                g1 = f1 - X.*log( max(f1, eps) ) + t1.*lambda + 0.5*rho.*(t1 - Xhat).^2;
                g2 = f2 - X.*log( max(f2, eps) ) + t2.*lambda + 0.5*rho.*(t2 - Xhat).^2;

                ind1better = (g1 <= g2);
                T(p &  ind1better) = t1(p &  ind1better);
                T(p & ~ind1better) = t2(p & ~ind1better);

                Y1  = (Y >= 1);
                Ym1 = (Y <= -1);
                mid = ~Y1 & ~Ym1;

                t1 = (Y - 1) ./ rho;
                t2 = (Y + 1) ./ rho;

                q  = ~p; % X==0
                T(q & Y1)  = t1(q & Y1);
                T(q & Ym1) = t2(q & Ym1);
                T(q & mid) = 0;

        end

    case 'Exponential'
        switch error_measure
            case 'Frobenius'
                parfor idx = 1:numel(T)
                    f = @(t) 0.5*(X(idx) - exp(t)).^2 + (rho/2)*(t - Xhat(idx) + lambda(idx)).^2;
                    T(idx) = fminsearch(f,0);
                end
            case 'KL divergence'
                parfor idx = 1:numel(T)

                    x  = X(idx);
                    xhat = Xhat(idx);
                    l  = lambda(idx);
                    if x > 0
                        T(idx) = fminsearch(@(t) exp(t) - x*t + (rho/2)*(t-xhat+l).^2,0);
                    else
                        T(idx) = fminsearch(@(t) exp(t) + (rho/2)*(t-xhat+l).^2,0);
                    end
                end
        end
end