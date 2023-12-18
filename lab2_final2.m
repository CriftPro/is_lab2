close all;
clear
clc;
% 1. Duomenų paruošimas
 x1 = 0.1: 1/22 :1;
 x2 = 0.1: 1/22 :1;

N = size(x1,2)
K = size(x2,2)
f = @(x1,x2) (1 + 0.6*sin(2*pi*x1/0.7) + 0.3*sin(2*pi*x2))/2;
for index1 = 1:N
    for index2 = 1:K
        desired_output(index1,index2) = f(x1(index1),x2(index2));
    end
end



figure(1)
surf(desired_output)


% 2. Tinklo struktūros pasirinkimas
% 1 įėjimas; 4 neuronai paslėptajame sluoksnyje, 2 išėjimai
% paslėptajame sluoksnyje sigmoidė, išėjimo sluoksnyje – tiesinė aktyvavimo
% funkcija

% 3. Pradinių parametrų reikšmių pasirinkimas

% I sluoksnis
w11_1 = randn(1); w12_1 = randn(1); b1_1 = randn(1);
w21_1 = randn(1); w22_1 = randn(1); b2_1 = randn(1);
w31_1 = randn(1); w32_1 = randn(1); b3_1 = randn(1);
w41_1 = randn(1); w42_1 = randn(1); b4_1 = randn(1);

% II sluoksnis
w11_2=randn(1); w12_2=randn(1); w13_2=randn(1); w14_2=randn(1); b1_2=randn(1); 


n = 0.01; % mokymo žingsnis
epoch = 90000; % Mokymo epochų skaičius

% Sigmoidinė aktyvacijos funkcija
sigmoid = @(x) 1./(1 + exp(-x));



% 4. Tinklo atsako skaičiavimas
for k = 1:epoch
    for index1 = 1:N
        for index2 = 1:K
            % I sluoksnis
            v1_1 = x1(index1)*w11_1 + x2(index2)*w12_1 + b1_1; y1_1 = sigmoid(v1_1);
            v2_1 = x1(index1)*w21_1 + x2(index2)*w22_1 + b2_1; y2_1 = sigmoid(v2_1);
            v3_1 = x1(index1)*w31_1 + x2(index2)*w32_1 + b3_1; y3_1 = sigmoid(v3_1);
            v4_1 = x1(index1)*w41_1 + x2(index2)*w42_1 +b4_1; y4_1 = sigmoid(v4_1);
            % II sluoksnis
            v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2; y1 = v1_2;
            %         y2 musu output
            %
            % 5. Klaidos skaičiavimas E = 1/2*e1^2 + 1/2*e2^2;
            e1 = desired_output(index1,index2) - y1;
            %
            % 6. Tinklo koeficientų (ryšių svorių) atnaujinimas
            % w = w + n*delta*IN;
            % delta_out = aktyvavimo_funkcijos_išvestinė|v * tikslo_funkcijos_išvestinė|e
            % delta_hidden = aktyvavimo_funkcijos_išvestinė|v * (w1*delta_out1 + w2*delta_out2 + ...)
            delta_out1 = 1 * 1/2*2*e1^1; % delta_out1 = e1;
            delta_hidden1 = y1_1*(1 - y1_1)*(w11_2*delta_out1);
            delta_hidden2 = y2_1*(1 - y2_1)*(w12_2*delta_out1);
            delta_hidden3 = y3_1*(1 - y3_1)*(w13_2*delta_out1);
            delta_hidden4 = y4_1*(1 - y4_1)*(w14_2*delta_out1);

            w11_2 = w11_2 + n*delta_out1*y1_1; w12_2 = w12_2 + n*delta_out1*y2_1;
            w13_2 = w13_2 + n*delta_out1*y3_1; w14_2 = w14_2 + n*delta_out1*y4_1;
            b1_2 = b1_2 + n*delta_out1*1;



            w11_1 = w11_1 + n*delta_hidden1*x1(index1); w12_1 = w12_1 + n*delta_hidden1*x2(index2);
            b1_1 = b1_1 + n*delta_hidden1;
            w21_1 = w21_1 + n*delta_hidden2*x1(index1); w22_1 = w22_1 + n*delta_hidden2*x2(index2);
            b2_1 = b2_1 + n*delta_hidden2;
            w31_1 = w31_1 + n*delta_hidden3*x1(index1); w32_1 = w32_1 + n*delta_hidden3*x2(index2);
            b3_1 = b3_1 + n*delta_hidden3;
            w41_1 = w41_1 + n*delta_hidden4*x1(index1); w42_1 = w42_1 + n*delta_hidden4*x2(index2);
            b4_1 = b4_1 + n*delta_hidden4;
        end
    end
end

% 7. Tinklo testavimas

x3 = 0.1:0.05:1;
x4 = 0.1:0.05:1;

N = size(x3,2);
K = size(x4,2)

for index1 = 1:N
    for index2 = 1:K
        desired_output2(index1,index2) = f(x3(index1),x4(index2));
        
    end
end

Y1 = zeros(N,K);


e1=0;

for index1 = 1:N
   for index2 = 1:K
       % I sluoksnis

       v1_1 = x3(index1)*w11_1  + x4(index2)*w12_1 + b1_1; y1_1 = sigmoid(v1_1);
       v2_1 = x3(index1)*w21_1  + x4(index2)*w22_1 + b2_1; y2_1 = sigmoid(v2_1);
       v3_1 = x3(index1)*w31_1  + x4(index2)*w32_1 + b3_1; y3_1 = sigmoid(v3_1);
       v4_1 = x3(index1)*w41_1  + x4(index2)*w42_1 + b4_1; y4_1 = sigmoid(v4_1);
       % II sluoksnis
       v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2; y1 = v1_2;

       % 8. Klaidos skaičiavimas E = 1/2*e1^2 + 1/2*e2^2;
       Y1(index1,index2) = y1;

       e1 = e1 + abs(desired_output2(index1,index2) - y1);
   end
    
end
e1 = e1/N;

disp("Error "+e1)

figure(2)
surf(desired_output2)
hold on
surf(Y1)
hold off

