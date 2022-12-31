#!/usr/bin/env python3
import os
import re
import shutil

if __name__ == '__main__':
    arr = []
    child_dir = '/home/y-takahashi/catkin_ws/src/nav_cloning/data/result/analysis/img/corner2/'
    i = 3231
    for file_name in range(513):
        regex = re.compile(str(i))
        for file_name in os.listdir(child_dir):
            if file_name.endswith('.jpg'):
                arr.append(file_name)
                new_name = regex.sub(str(i - 456), file_name)
                print(new_name)
                shutil.move(child_dir + file_name, child_dir + new_name)
        i += 1