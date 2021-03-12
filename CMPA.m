
i_s = 0.01e-12;
i_b = 0.1e-12;
v_b = 1.3;
g_p = 0.1;

v = linspace(-1.95, 0.7, 200)';

I = i_s*(exp((1.2/0.025)*v)-1) + g_p*v - i_b*(exp((-1.2/0.025)*(v+v_b))-1);

random = (0.2.*randn(200,1)+1);
I_rand = I.*random;

figure(1)
plot(v, I)
figure(2)
semilogy(v, I)

figure(3)
plot(v, I_rand)
figure(4)
semilogy(v, I_rand)

figure(5)
p1 = polyfit(v,I,4);
p11 = polyval(p1, v);
plot(v, p11)
title("4th order regular")
figure(6)
p2 = polyfit(v,I_rand,4);
p22 = polyval(p2, v);
plot(v, p22)
title("4th order rand")
figure(7)
p3 = polyfit(v,I,8);
p33 = polyval(p3, v);
plot(v, p33)
title("8th order regular")
figure(8)
p4 = polyfit(v,I_rand,8);
p44 = polyval(p4, v);
plot(v, p44)
title("8th order rand")

fo = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');

ff = fit(v, I, fo);

If = ff(v);

figure(9)
plot(v, If)

inputs = v';
targets = I';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;







