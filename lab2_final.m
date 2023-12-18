clear
clc;
% 1. Duomenų paruošimas
x1 = 0.1: 1/22: 1;
N = size(x1,2)

f = @(x) (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;
desired_output = f(x1);

%    end
%end
figure(1)
plot(x1,desired_output,'kx')
% 2. Tinklo struktūros pasirinkimas
% 1 įėjimas; 4 neuronai paslėptajame sluoksnyje, 2 išėjimai
% paslėptajame sluoksnyje sigmoidė, išėjimo sluoksnyje – tiesinė aktyvavimo
% funkcija

% 3. Pradinių parametrų reikšmių pasirinkimas

% I sluoksnis
w11_1 = randn(1);  b1_1 = randn(1);
w21_1 = randn(1);  b2_1 = randn(1);
w31_1 = randn(1);  b3_1 = randn(1);
w41_1 = randn(1);  b4_1 = randn(1);

% II sluoksnis
w11_2=randn(1); w12_2=randn(1); w13_2=randn(1); w14_2=randn(1); b1_2=randn(1); 


n = 0.1; % mokymo žingsnis
epoch = 90000; % Mokymo epochų skaičius

% Sigmoidinė aktyvacijos funkcija
sigmoid = @(x) 1./(1 + exp(-x));

% 4. Tinklo atsako skaičiavimas
for k = 1:epoch
    for indx = 1:N
        % I sluoksnis
          v1_1 = x1(indx)*w11_1 + b1_1; y1_1 = sigmoid(v1_1);
          v2_1 = x1(indx)*w21_1 + b2_1; y2_1 = sigmoid(v2_1);
          v3_1 = x1(indx)*w31_1 + b3_1; y3_1 = sigmoid(v3_1);
          v4_1 = x1(indx)*w41_1 + b4_1; y4_1 = sigmoid(v4_1); 
        % II sluoksnis
        v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2; y1 = v1_2;
%         y2 musu output
%        
        % 5. Klaidos skaičiavimas E = 1/2*e1^2 + 1/2*e2^2;
        e1 = desired_output(indx) - y1;
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
        
        
        
        w11_1 = w11_1 + n*delta_hidden1*x1(indx); 
        b1_1 = b1_1 + n*delta_hidden1;
        w21_1 = w21_1 + n*delta_hidden2*x1(indx); 
        b2_1 = b2_1 + n*delta_hidden2;
        w31_1 = w31_1 + n*delta_hidden3*x1(indx); 
        b3_1 = b3_1 + n*delta_hidden3;
        w41_1 = w41_1 + n*delta_hidden4*x1(indx); 
        b4_1 = b4_1 + n*delta_hidden4;
    end
end

% 7. Tinklo testavimas

x3 = 0.1:1/30:1;
% x3 = randn(1,20)
N = size(x3,2);
Y1 = zeros(1,N);

desired_output = f(x3);

e1=0;

for indx = 1:N
   
    % I sluoksnis

    v1_1 = x3(indx)*w11_1  + b1_1; y1_1 = sigmoid(v1_1);
    v2_1 = x3(indx)*w21_1  + b2_1; y2_1 = sigmoid(v2_1);
    v3_1 = x3(indx)*w31_1  + b3_1; y3_1 = sigmoid(v3_1);
    v4_1 = x3(indx)*w41_1  + b4_1; y4_1 = sigmoid(v4_1);
    % II sluoksnis
    v1_2 = y1_1*w11_2 + y2_1*w12_2 + y3_1*w13_2 + y4_1*w14_2 + b1_2; y1 = v1_2;
    
    % 8. Klaidos skaičiavimas E = 1/2*e1^2 + 1/2*e2^2;
    Y1(indx) = y1;
    
    e1 = e1 + abs(desired_output(indx) - y1);
    
    
end
e1 = e1/N;

disp("Error "+e1)


figure(2)
plot(1:N,Y1,'r',1:N,Y1,'ko')
hold on 
plot(1:N,desired_output,'b',1:N,desired_output,'kx')
legend('predicted','predicted points','real','real points')