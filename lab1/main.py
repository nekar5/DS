#---------- ЛР 1. Дослідження характеристик експериментальних даних
#---------- ІТ-03 Карванський Нестор
#---------- Варіант 5

import numpy as np
import math as mt
import matplotlib.pyplot as plt

#обчислення похибки
def get_errors(S, mean):
    errors = []
    for i in S:
        error=i-mean
        #errors.append(abs(error))
        errors.append(error)
    return errors

#обчислення мат. сподівання
def get_msp(S, n):
    prb = 1/n
    sum = 0
    for i in range (0, n):
        sum += (S[i] * prb)
    return float(sum)

#вивід інформації
def print_stats(S, n):
    #print('матриця реалізацій = ', S) #матриця реалізацій
    print('мат. сподівання ВВ = ', get_msp(S, n)) #мат. сподівання
    print('медіана ВВ = ', np.median(S)) #медіана
    print('дисперсія ВВ = ', np.var(S)) #дисперсія
    print('СКВ ВВ = ', mt.sqrt(np.var(S))) #середньоквадратичне відхилення
    print('\n')

#генерація нормального розподілу похибки
def generate_normal_error(n):
    mean, sd = 0, 5 #50, 0.1
    pdf = ((np.random.randn(n)) * sd) + mean

    err = np.zeros((n))
    err = get_errors(pdf, mean)

    """
    print("Нормальний та похибка:")
    print_stats(pdf, n)
    print_stats(err, n)
    plt.hist(pdf, bins=25, alpha=0.5, label="Нормальний")
    plt.hist(err, bins=25, alpha=0.5, label="Похибка")
    plt.title('Рівномірний')
    plt.legend()
    plt.show()
    """
    return err, pdf

#генерація рівномірного розподілу
def generate_uniform(n):
    S = np.random.uniform(0, 100, size=n)

    """
    print("Рівномірний:")
    print_stats(normal_error, n)
    plt.hist(normal_error, bins=25, alpha=0.5, label="Рівномірний")
    plt.title('Рівномірний')
    plt.legend()
    plt.show()
    """
    return S

#генерація адитивної моделі 
def generate_additional(S, normal_error, uniform, n):
    add_model = []
    corrected = np.zeros((n))
    for i in range(n):
        corrected[i] = normal_error[i] + uniform[i]

    for i in range(n):
        S[i] = (0.0000005 * i * i)
        addit = corrected[i] + S[i]
        add_model.append(addit)
    """
    print("Адитивна:")
    print_stats(S, n)
    print_stats(corrected, n)
    print_stats(add_model, n)
    plt.hist(S, bins=25, alpha=0.5, label="Квадратична")
    plt.hist(corrected, bins=25, alpha=0.5, label="Рівномірний з нормальним шумом")
    plt.hist(add_model, bins=25, alpha=0.5, label="Адитивна")
    plt.title('Адитивна')
    plt.legend()
    plt.show()
    """
    return S, corrected, add_model


def main():
    n=10000
    for i in range(100):
        #print(i) #test
        normal_error, S = generate_normal_error(n)
        uniform = generate_uniform(n)
        S, corrected, addit_model = generate_additional(S, normal_error, uniform, n)

    print("Похибка:")
    print_stats(normal_error, n)
    plt.hist(normal_error, bins=25, alpha=0.5, label="Похибка")
    plt.title("Похибка")
    plt.legend()
    plt.show()

    """
    plt.plot(normal_error, label="Похибка")
    plt.plot(uniform, label="Рівномірний")
    plt.title("Рівномірний та похибка")
    plt.legend()
    plt.show()
    """

    print("Рівномірний:")
    print_stats(uniform, n)
    plt.hist(uniform, bins=25, alpha=0.5, label="Рівномірний")
    plt.title("Рівномірний розп. дос. процесу")
    plt.legend()
    plt.show()

    print("Адитивна модель експ. даних:")
    print_stats(addit_model, n)
    plt.title("Адитивна модель експ. даних")
    plt.hist(S, bins=25, alpha=0.5, label="Квадратична")
    plt.hist(corrected, bins=25, alpha=0.5, label="Рівномірна з нормальним шумом")
    plt.hist(addit_model, bins=25, alpha=0.5, label="Адитивна")
    plt.legend()
    plt.show()
main()