#---------- ЛР 2. Дослідження алгоритмів згладжування за накопиченою вибіркою
#---------- ІТ-03 Карванський Нестор
#---------- Варіант 5

import numpy as np
import math as mt
import matplotlib.pyplot as plt

#глобальні змінні:
n = 10000 #ВВ
anomaly_percentage = 10 #Аномальні виміри складають 10% від загальної кількості вимірів у експериментальної вибірки
anomaly_amount = int((n*anomaly_percentage)/100) #к-ть АВ
mean, sd = 0, 5 #вип. сер., похибка

#обчислення похибки
def get_errors(S, mean):
    errors = []
    for i in S:
        error=i-mean
        errors.append(error)
    return errors

#обчислення мат. сподівання
def get_msp(S):
    prb = 1/n
    sum = 0
    for i in range (0, n):
        sum += (S[i] * prb)
    return float(sum)

#метод для виводу статистичних характеристик масивів (медіана, дисперсія, СКВ)
def print_stats(S):
    #print('матриця реалізацій = ', S) #матриця реалізацій
    #print('мат. сподівання ВВ = ', get_msp(S)) #мат. сподівання
    print('медіана ВВ = ', np.median(S)) #медіана
    print('дисперсія ВВ = ', np.var(S)) #дисперсія
    print('СКВ ВВ = ', mt.sqrt(np.var(S))) #середньоквадратичне відхилення
    print('\n')



#метод stat_data з return, але без виводу даних
def get_stats(S):
    median = np.median(S)  #медіана
    dispers = np.var(S)    #дисперсія
    scv = mt.sqrt(dispers) #середньоквадратичне відхилення
    return median, dispers, scv


#метод генерація моделей (ВВ похибки, рівн. з нормальним шумом, адитивна)
def generate_models():
    #похибки
    S = np.zeros((n))
    S = ((np.random.randn(n)) * sd) + mean
    print("Вхідна вибірка")
    print_stats(S)

    #розрахунок похибки
    err = np.zeros((n))
    err = get_errors(S, mean)

    plt.hist(S, bins=20, alpha=0.5, label="похибка")
    plt.title('Закон розподілу вхідних даних')
    plt.show()

    uniform = np.zeros((n))
    uniform = np.random.uniform(-10, 10, size=n)

    #трендова модель
    add_model = []
    plus_error = np.zeros((n))
    plus_error = np.add(err, uniform)

    for i in range(n):
        S[i] = (0.0000005 * i * i) #кв. закон
    add_model = np.add(plus_error, S)

    print('Вхідна вибірка без аномальних вимірів')
    print_stats(add_model)
    
    plt.hist(add_model)
    plt.title('Вхідна вибірка')
    plt.show()

    plt.title('Аномалії відсутні')
    plt.plot(add_model)
    plt.plot(S)
    plt.ylim([-60, 90])
    plt.show()
    return add_model



#метод генерації аномальних вимірів у вибірці
def generate_anomalies(S):
    anom=np.zeros((n))
    for i in range(n):
        anom[i]=np.random.randint(0, n)
    #stat_data(anom)
    for i in range(anomaly_amount):
        S[i]=mt.ceil(np.random.randint(1, n))
    return S

#метод створення адитивної моделі з присутніми аномальними вимірами
S0 = np.zeros((n))
def generate_addtitional_WA(AM, AV):
    SV0=np.zeros((n))
    S2AV=np.zeros((anomaly_amount))
    sWA=np.zeros((n))
    
    for i in range(n):
        S0[i] = (0.0000005 * i * i) #qua
        SV0[i] = abs(AM[i] - S0[i]) #ideal
        sWA[i] = AM[i] #rl
    S2AV = np.random.normal(mean, (3 * sd), anomaly_amount)
    for i in range(anomaly_amount):
        k = int(AV[i])
        sWA[k] = S0[k] + S2AV[i]

    print('Вибірка з аномаліями')
    print_stats(sWA)

    plt.plot(sWA)
    plt.plot(S0)
    plt.title('Аномалії присутні')
    plt.ylim([-60, 90])
    plt.show()
    return sWA, S0


#метод, який виявляє аномалії
def detect_anomalies(sWA):
    idxs=[]
    threshold = 3
    temp_mean = np.mean(sWA)
    temp_sd = np.std(sWA)

    for i in sWA: #zscore
        score = (i - temp_mean) / temp_sd
        if (np.abs(score) > threshold):
            idxs.append(sWA.index(i))

    return idxs

#метод, який відкидає знайдені detect_anomalies() аномалії
def correct_anomalies(sWA):
    anomaly_idxs = detect_anomalies(sWA)

    temp = []
    for i in sWA: #correct
        if(not sWA.index(i) in anomaly_idxs):
            temp.append(i)
    sWA=temp

    plt.plot(sWA)
    plt.plot(S0)
    plt.title('Аномалії усунені')
    plt.ylim([-60, 90])
    plt.show()


#метод згладжування
def ensmoothen(S0, sWN, sWA):
    Y_in = np.zeros((n, 1))
    F = np.ones((n, 3))

    for i in range(n): #матриці без аномалій
        Y_in[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)

    def MNK(Y_in, F):
        FT = F.T
        FTdF = FT.dot(F)
        FTDFinv = np.linalg.inv(FTdF)
        FTDFinv_dFT = FTDFinv.dot(FT)
        C = FTDFinv_dFT.dot(Y_in)
        Y_out = F.dot(C)
        return Y_out
    
    #буз шуму
    smooth_NN = MNK(Y_in, F) #no noise 

    #з шумом
    for i in range(n):
        Y_in[i, 0] = float(sWN[i])
    smooth_WN = MNK(Y_in, F) #with noise
    
    #з аномаліями
    for i in range(np.size(sWA)): #np.size(VV_AV) < n
        Y_in[i, 0] = float(sWA[i])
    smooth_WA = MNK(Y_in, F) #with anomalies

    sNN_C= np.zeros((n)); sWN_C = np.zeros((n)); sWA_C = np.zeros((n))
    for i in range(n):
         sNN_C[i] = abs(smooth_NN[i] - S0[i])
         sWN_C[i] = abs(smooth_WN[i] - S0[i])
         sWA_C[i] = abs(smooth_WA[i] - S0[i])

    print('Згладжена вибірка:')
    sNN_msp, sNN_var, sNN_sd = get_stats(sNN_C)
    sWN_msp, sWN_var, sWN_sd = get_stats(sWN_C)
    sWA_msp, sWA_var, sWA_sd = get_stats(sWA_C)
    print('                    БЕЗ ПОХИБОК            ПОХИБКИ НОРМАЛЬНІ              ПОХИБКИ АНОМАЛЬНІ -----')
    print('Мат. сподівання | ', sNN_msp, ' | ', sWN_msp, ' | ', sWA_msp)
    print('Дисперсія       | ', sNN_var, ' | ', sWN_var, ' | ', sWA_var)
    print('СК відхилення   | ', sNN_sd, ' | ', sWN_sd, ' | ', sWA_sd)
    return smooth_NN, smooth_WN, smooth_WA




def main():
    AM = generate_models()
    AV=np.zeros((anomaly_amount))
    AV = generate_anomalies(AV)
    sWA, S0 = generate_addtitional_WA(AM, AV)

    sWA = sWA.tolist()
    correct_anomalies(sWA)
    NN, WN, WA = ensmoothen(S0, AM, sWA)

    plt.title('Вхідна вибірка значень', loc = 'center', fontsize = 18)
    plt.ylim([-60, 90])
    plt.plot(S0)
    plt.plot(AM)
    plt.plot(sWA)
    plt.plot(NN)
    plt.plot(WN)
    plt.plot(WA)
    plt.legend()
    plt.show()

    plt.hist(AM, bins=20, alpha=0.5, label='AM')
    plt.hist(S0, bins=20, alpha=0.5, label='S0')
    plt.hist(sWA, bins=20, alpha=0.5, label='sWA')
    plt.hist(NN, bins=20, alpha=0.5)
    plt.hist(WN, bins=20, alpha=0.5)
    plt.hist(WA, bins=20, alpha=0.5)
    plt.legend()
    plt.show()

main()