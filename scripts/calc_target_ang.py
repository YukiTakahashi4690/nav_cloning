import csv
import math

path = '/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/rm/straight/ang.csv'
pos_list = []

with open(path, 'r') as csvfile:
        for row in csvfile:
            pos_list.append(row)

def calc_avg():
    no = 11
    while no < len(pos_list):  
        cur_pos = pos_list[no]
        pos = cur_pos.split(',')
        print(pos)
        value = float(pos[1])
        value = [value]
        no += 15
        average = sum(value) / len(value)
    print(f"平均値: {average}")

calc_avg()

