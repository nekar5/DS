import pandas as pd
import numpy as np

def get_file(path):
    data = pd.read_excel(path)
    print("data=", data) #вивід

    #"посунути" масив наліво на 1
    data.set_index("Критерії", inplace=True) # зміна індексу датафрейму
    print(data) #повторний вивід
    
    return data

def transform(data, F):
    #text -> float
    for j in range(len(F)):
        for i in range(len(F)):
            F[j][i] = float((data.iloc[j, i]).replace(',', '.'))

    for i in F:
        print(i)
    
    return F

def Voronin(F, G, data):
    integrated = np.zeros(len(G))
    sum_F = [0 for y in range(10)]

    #масив нормативних частинних критерій:
    Fn=[[np.zeros((len(G))) for f in range(10)] for fn in range(10)]

    for i in range(len(F)):
        sum_F[i] =  np.sum(F[i])

    for j in range(len(F)):
        for i in range(len(F)):
            # нормативні частинні критерії
            if j == 6:
                Fn[j][i] = 1 / F[j][i] / sum_F[j]
            else:
                Fn[j][i] = F[j][i] / sum_F[j]

    for i in range(len(F)):
        def calculate():
            temp=0
            for j in range(len(F)):
                temp+=G[j]*(1-Fn[j][i])**(-1)
                #print(Fn[j][i])
            #print(str(temp)+"\n")
            return temp

        integrated[i] = calculate()


    min=10000; optimal=0
    for i in range(len(integrated)):
        if min > integrated[i]:
            min = integrated[i]
            optimal=i

    print("integrated:", integrated, sep='\n')
    print("оптимальний маршрут :   № " + str(optimal))
    print(data.iloc[:, optimal])

def main():
    data = get_file("C:\\Users\\Nestor\\Desktop\\lab4\\data.xlsx")
    
    criteria_amount = data.shape[0]
    F=[[np.zeros((criteria_amount)) for f in range(10)] for fn in range(10)]

    F = transform(data, F)

    #масив вагових критерій
    G = np.ones(len(F))
    #масив нормованих вагових критерій
    Gnorm = np.zeros(len(F))

    Gsum = np.sum(G)+G[6]+1

    #нормування вагових коефіцієнтів
    for i in range(len(F)):
        Gnorm[i] = G[i]/Gsum

    Voronin(F, Gnorm, data)

main()