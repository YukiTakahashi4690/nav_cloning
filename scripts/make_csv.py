import csv

class maka_csv_node:
        def __init__(self):
            self.path = "/home/y-takahashi/catkin_ws/src/nav_cloning/data/ang/test2/ang.csv"
            self.write_path = "/home/y-takahashi/catkin_ws/src/nav_cloning/data/analysis/-02m_5deg.csv"
            self.pos_list = []
            with open(self.path, "r") as csvfile:
                for row in csvfile:
                    self.pos_list.append(row)

        def write_csv(self):
            start_no = 8
            with open(self.write_path, "a") as f:
                writer = csv.writer(f, lineterminator="")
                while start_no < len(self.pos_list):
                    cur_list = self.pos_list[start_no]
                    self.list = cur_list.split(",")
                    # print(self.list)
                    writer.writerow(self.list)
                    start_no += 9
                print("complete")

node = maka_csv_node()
node.write_csv()