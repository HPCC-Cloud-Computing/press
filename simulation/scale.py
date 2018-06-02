import math


class Scale(object):
    def __init__(self, nbox, start_box, box_capacity, warm_up_time, cost):
        """
        nbox: number of boxes
        start_box
        box_capacity: capacity of a box
        cost: cost of a box
        """
        self.nbox = nbox
        self.start_box = start_box
        self.warm_up_time = warm_up_time
        self.box_capacity = box_capacity
        self.cost = cost

    def scale(self, data, interval=1):
        """
        Input:
            data: array
            interval
        Output:
            A tupple of:
            - number of boxes system used
            - cost for run
            - overload time
            - number of request lost
        """
        box = []
        overload_time = 0
        lost_request = 0
        cost = 0
        for i in range(len(data)):
            need_box = math.ceil(data[i] / self.box_capacity)
            if i < self.warm_up_time / interval:
                b = self.start_box
            else:
                b = self.nbox if need_box > self.nbox else need_box
            box.append(b)
            cost += b * self.cost
            if b < need_box:
                overload_time += interval
                lost_request += data[i] - b * self.box_capacity

        return (box, cost, overload_time, lost_request)
