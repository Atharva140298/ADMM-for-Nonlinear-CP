function x=cardanoatharva(c3,c1,c0)
    % this function outputs the root we want (we hope so)
     % of c3*x^3+c1*x+c0 = 0
    a     = (c1)/(c3);
    b     = (c0)/(c3);
    delta = 4*(a^3)+27*(b^2);
    if delta<=0
        r   = 2*sqrt(-a/3);
        th  = atan2(sqrt(-delta/27),-b)/3;
        t0  = r*cos(th);
        t02 = t0*t0;
        t1  = r*cos(th+((2*3.141592653589793238)/3));
        t12 = t1*t1;
        if t12*t12/4+a*t12/2+b*t1<t02*t02/4+a*t02/2+b*t0
            x  = t1;
        else
            x=t0;
        end
    else
        x = nthroot(0.5*(-b+sqrt(delta/27)),3)+nthroot(0.5*(-b-sqrt(delta/27)),3);
    end
end
