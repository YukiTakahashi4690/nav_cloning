import csv

csv_file_path = '/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/cp_00_01_02/ang.csv' 
ang_list = []
i = 0

with open(csv_file_path, "r") as fs:
    for row in csv.reader(fs):
        ang_list.append(row)
    print(ang_list[0])