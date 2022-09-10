import numpy as np
import math as mt
import matplotlib.pyplot as plt

n = 10000 #ВВ
anomaly_percentage = 10 #Аномальні виміри складають 10% від загальної кількості вимірів у експериментальної вибірки
anomaly_amount = int((n*anomaly_percentage)/100) #к-ть АВ
mean, sd = 0, 5 #вип. сер., похибка

#обчислення похибки
def get_errors(S):
    errors = []
    for i in S:
        error = 0.0000005 * i
        errors.append(error)
    return errors

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
    S = np.random.uniform(0, 100, size=n)

    print("Вхідна вибірка")
    print_stats(S)

    #розрахунок похибки
    err = np.zeros((n))
    err = get_errors(S)

    plt.hist(S, bins=20, alpha=0.5, label="похибка")
    plt.title('Закон розподілу вхідних даних')
    plt.show()

    normal = np.zeros((n))
    normal = ((np.random.randn(n)) * sd) + mean

    #трендова модель
    add_model = []
    plus_error = np.zeros((n))
    plus_error = np.add(err, normal)

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

#метод альфа бета фільтру
def alpha_beta_filter(Yin, iter):
    Yout = np.zeros((iter, 1)) #вихідні значення
    t0 = 1 #період оновлення інформації
    Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / t0 #швидкість n-1
    Yextra = Yin[0, 0] + Yspeed_retro #екстрапольоване значення

    alpha = 1  # 2 *(2*1 - 1)/(1*(1+1))
    beta = 3  # 6 / (1*(1+1))

    Yout[0, 0] = Yin[0, 0] + alpha * (Yin[0, 0])
    for i in range(1, iter):
        Yout[i, 0] = Yextra + alpha * (Yin[i, 0] - Yextra)  # координати
        Yspeed = Yspeed_retro + (beta / t0) * (Yin[i, 0] - Yextra)  # швидкість
        Yspeed_retro = Yspeed
        Yextra = Yout[i, 0] + Yspeed_retro
        alpha = 2 * (2 * i - 1) / (i * (i + 1))
        beta = 6 / (i * (i + 1))
    #print('Yin = ', Yin, 'Yout = ', Yout)
    return Yout


#метод згладжування (а-б)
def ensmoothenAB(S0, sWN, sWA):
    Yin = np.zeros((n, 1))
    for i in range(n):
        Yin[i, 0] = float(S0[i]) #без шуму
        Yin[i, 0] = float(sWN[i]) #нормальний шум
        Yin[i, 0] = float(sWA[i]) #норм. шум з аномаліями
    result = alpha_beta_filter(Yin, n)

    plt.plot(Yin)
    plt.plot(result)
    plt.title('Норм. шум + АВ (alpha-beta):')
    plt.show()
    Yin = np.zeros((n, 1))

    # незашумлені
    for i in range(n):
        Yin[i, 0] = float(S0[i])
        Yin[i, 0] = float(sWN[i])
    zgl1 = alpha_beta_filter(Yin, n)
    # зашумлені
    for i in range(n):
        Yin[i, 0] = float(sWN[i])

    zgl2 = alpha_beta_filter(Yin, n)
    # з АВ
    for i in range(n):
        Yin[i, 0] = float(sWA[i])

    zgl3 = alpha_beta_filter(Yin, n)
    zgl11 = np.zeros((n));
    zgl22 = np.zeros((n));
    zgl33 = np.zeros((n))
    for i in range(n):
        zgl11[i] = abs(zgl1[i] - S0[i])
        zgl22[i] = abs(zgl2[i] - S0[i])
        zgl33[i] = abs(zgl3[i] - S0[i])

    print('\n   Згладжена вибірка (AB):')
    zgl111, zgl112, zgl113 = get_stats(zgl11)
    zgl221, zgl222, zgl223 = get_stats(zgl22)
    zgl331, zgl332, zgl333 = get_stats(zgl33)
    print('                    БЕЗ ПОХИБОК            ПОХИБКИ НОРМАЛЬНІ              ПОХИБКИ АНОМАЛЬНІ')
    print('Мат. сподівання | ', zgl111, ' | ', zgl221, ' | ', zgl331)
    print('Дисперсія       | ', zgl112, ' | ', zgl222, ' | ', zgl332)
    print('СК відхилення   | ', zgl113, ' | ', zgl223, ' | ', zgl333)

    return zgl1, zgl2, zgl3


#метод альфа бета гама фільтру
def alpha_beta_gama_filter(Zin, iter):

    Zout = np.zeros((iter, 1)) #вихідні значення
    t0 = 1 #період оновлення інформації
    Yspeed_retro = (Zin[1, 0] - Zin[0, 0]) / t0 #швидкість n-1
    Zacln=1 #прискорення
    Zextra = Yspeed_retro + Zin[0, 0]*t0 + Zacln*t0/2 #екстрапольоване значення

    alpha = 1 # 3*(3*1^2 - 3*1 +2) / (1*(1+1)*(1+2))
    beta = 3 # 18(2*1 - 1) / (t0 * (1+1)*(1+2)*1)
    gamma = 10 # 60 / (t0^2 * (1+1)*(1+2)*1)

    Zout[0, 0] = Zin[0, 0] + alpha * (Zin[0, 0])
    for i in range(1, iter):
        Zout[i, 0] = Zextra + alpha * (Zin[i, 0] - Zextra)  #координати
        Zspeed = (Yspeed_retro+Zacln*t0) + (beta / t0) * (Zin[i, 0] - Zextra)  #швидкість

        Zacln = Zacln + (gamma/pow(t0, 2))*(Zin[i, 0] - Zextra) #прискорення
        
        Yspeed_retro = Zspeed # швидкість n-1
        Zextra = Zout[i, 0] + Yspeed_retro

        alpha = 3*(2*(pow(i, 2)) - 3*i + 2) / (i*(i+1)*(i+2))
        beta = 18*(2*i - 1) / (t0 * (i+1)*(i+2)*i)
        gamma = 60 / (pow(t0, 2) * (i+1)*(i+2)*i)
    #print('Zin = ', Zin, 'Zout = ', Zout)

    return Zout

#метод згладжування (а-б-г)
def ensmoothenABG(S0, sWN, sWA):
    Zin = np.zeros((n, 1))

    for i in range(n):
        Zin[i, 0] = float(S0[i])
        Zin[i, 0] = float(sWN[i])
        Zin[i, 0] = float(sWA[i])
    result = alpha_beta_gama_filter(Zin, n)

    plt.plot(Zin)
    plt.plot(result)
    plt.title('Норм. шум + АВ (alpha-beta-gamma):')
    plt.show()
    
    Zin = np.zeros((n, 1))

    #бeз шуму
    for i in range(n):
        Zin[i, 0] = float(S0[i])
        Zin[i, 0] = float(sWN[i])
    smooth_NN = alpha_beta_gama_filter(Zin, n)

    #з шумом
    for i in range(n):
        Zin[i, 0] = float(sWN[i])
    smooth_WN = alpha_beta_gama_filter(Zin, n)
    
    #з аномаліями
    for i in range(np.size(sWA)): #np.size(VV_AV) < n
        Zin[i, 0] = float(sWA[i])
    smooth_WA = alpha_beta_gama_filter(Zin, n)

    sNN_C= np.zeros((n)); sWN_C = np.zeros((n)); sWA_C = np.zeros((n))
    for i in range(n):
         sNN_C[i] = abs(smooth_NN[i] - S0[i])
         sWN_C[i] = abs(smooth_WN[i] - S0[i])
         sWA_C[i] = abs(smooth_WA[i] - S0[i])

    print('\n   Згладжена вибірка (ABG):')
    sNN_msp, sNN_var, sNN_sd = get_stats(sNN_C)
    sWN_msp, sWN_var, sWN_sd = get_stats(sWN_C)
    sWA_msp, sWA_var, sWA_sd = get_stats(sWA_C)
    print('                    БЕЗ ПОХИБОК            ПОХИБКИ НОРМАЛЬНІ              ПОХИБКИ АНОМАЛЬНІ')
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

    ensmoothenAB(S0, AM, sWA)

    ensmoothenABG(S0, AM, sWA)

main()