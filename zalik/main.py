import pandas as pd
from datetime import date
import requests

#координати Києва
latitude="50.450001" #N
longitude="30.523333" #E
response = requests.get("https://api.openweathermap.org/data/2.5/onecall?lat="+latitude+"&lon="+longitude+"&exclude=current,minutely,hourly,alerts&units=metric&appid=6a6da2419cc44cad040cded0613f6276")

data = pd.read_excel('kyiv-ukraine.xlsx')

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31]) #останні дні у місяцях

def get_avg(d, m): #видає середні температури за день
    min_avg = 0
    max_avg = 0
    index = 0
    while index < len(data.index):
        if(data.iloc[index, 1] == m):
            min_avg += data.iloc[index + d - 1, 6]
            max_avg += data.iloc[index + d - 1, 5]

            while index < len(data.index) and data.iloc[index, 1] == m:
                index = index + 1 #без повторювань, ітеруємо заново
        index = index + 1
    min_avg /= 10
    max_avg /= 10
    return pd.Series({'min':min_avg, 'max':max_avg}) #серії: 0 - мінімальна, 1 максимальна


day = date.today().day
month = date.today().month
print("Date: "+str(month)+"/"+str(day))

#"пустишки", далі будуть заповнюватися
min_temps = pd.Series(0, index=['-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7'])
max_temps = pd.Series(0, index=['-6','-5','-4','-3','-2','-1','0','1','2','3','4','5','6','7'])

curday = day
curmonth = month
for i in range(7): #заповнення середніми даними
    averages = get_avg(curday, curmonth)
    min_temps.iloc[6 - i] = averages.iloc[0]
    max_temps.iloc[6 - i] = averages.iloc[1]

    curday -= 1
    if(curday == 0): #петля місяців (12->1)
        if(curmonth == 0): #петля днів
            curmonth = 13
        curmonth -= 1
        curday = days.iloc[curmonth - 1]
curday = day
curmonth = month
for i in range(8):  #0 - 7 заповнення середніми
    averages = get_avg(curday, curmonth)
    min_temps.iloc[6 + i] = averages.iloc[0]
    max_temps.iloc[6 + i] = averages.iloc[1]

    curday += 1
    if(curday > days.iloc[curmonth - 1]): #12 -> 1 перехід місяців
        if(curmonth == 12): #31 -> 1
            curmonth = 0
        curmonth += 1
        curday = 1

#константи
sum_x = 91
sum_sqrx = 819  

min_sum_y = 0; min_sum_xy = 0
max_sum_y = 0; max_sum_xy = 0


for item in range(14): #використовується в регресії
    min_sum_y += min_temps.iloc[item]
    min_sum_xy += (item * min_temps.iloc[item])
    max_sum_y += max_temps.iloc[item]
    max_sum_xy += (item * max_temps.iloc[item])

#мін макс
min_a = ((min_sum_y * sum_sqrx) - (sum_x * min_sum_xy)) / ((14 * sum_sqrx) - (sum_x**2))
min_b = ((14 * min_sum_xy) - (sum_x * min_sum_y)) / ((14 * sum_sqrx) - (sum_x**2))
max_a = ((max_sum_y * sum_sqrx) - (sum_x * max_sum_xy)) / ((14 * sum_sqrx) - (sum_x**2))
max_b = ((14 * max_sum_xy) - (sum_x * max_sum_y)) / ((14 * sum_sqrx) - (sum_x**2))

#похибка
var_min = 0
var_max = 0

#підрахунок
for x in range(14):
    var_min += ((min_temps.iloc[x] - ((min_b * x) + min_a))**2)
    var_max += ((max_temps.iloc[x] - ((max_b * x) + max_a))**2)

#+-похибки:
var_min /= 12 
var_max /= 12


print("\nformula:")
print("high: {0:.1f}x + {1:.1f}, sd: {2:.1f}".format(max_b, max_a, var_max))
print("low: {0:.1f}x + {1:.1f}, sd: {2:.1f}".format(min_b, min_a, var_min))

print("\n\tAfter {}/{} (linear regression):".format(str(month), str(day)))
curday = day
curmonth = month
for p in range(7): #7 днів
    curday += 1
    if(curday > days.iloc[curmonth - 1]):
        if(curmonth == 12):
            curmonth = 0
        curmonth += 1
        curday = 1 #вивід наступних 7 днів (прогноз по регресії)
    #фаормула регресії +- помилка
    print("{}/{}: ".format(str(curmonth), str(curday))+"\thigh: {0:.1f}-{1:.1f} C\tlow: {2:.1f}-{3:.1f} C".format((max_b * (p + 7)) + max_a - var_max, (max_b * (p + 7)) + max_a + var_max, (min_b * (p + 7)) + min_a - var_min, (min_b * (p + 7)) + min_a + var_min))


owt_data = response.json()['daily']
owt_temp = []; owt_date = []

for d in owt_data:
    owt_date.append(d['dt'])
    owt_temp.append(d['temp'])

owt_min_temp = []; owt_max_temp = []
for t in owt_temp:
    owt_min_temp.append(t['min'])
    owt_max_temp.append(t['max'])

print("\n\tAfter {}/{} (openweathermap.org):".format(str(month), str(day)))
for i in range(1,8):
    print("{}\thigh: {} C\tlow: {} C".format(date.fromtimestamp(owt_date[i]) , owt_max_temp[i], owt_min_temp[i]))