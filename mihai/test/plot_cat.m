I_f = csvread('cat2d_f.csv');

I_f = I_f(1:272,1:272);

I = csvread('cat2d.csv');

I=I(1:272,1:272);

figure();

imshow([I, I_f'], [0, 1]);
