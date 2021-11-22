"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 25 Feb 2021
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# ================================ Imports ================================ #
import sys

sys.dont_write_bytecode = True
import os
from pprint import pprint
import time
from ipdb import set_trace
import shutil
from parameters import DISCOUNT, HORIZON, TINY, TOTAL_AGENTS
import pickle
import torch as tc
import numpy as np
from functools import reduce
from numpy import dot
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use("seaborn")


# plt.grid(False)
#p

# ============================================================================ #


def loadDataStr(x):

    file2 = open(x+'.pkl', 'rb')
    ds = pickle.load(file2)
    file2.close()
    return ds

def dumpDataStr(x,obj):

    afile = open(x+'.pkl', 'wb')
    pickle.dump(obj, afile)
    afile.close()


def listFiles(path):

    return os.listdir(path)

def splitList(n=None, list=None):

    '''
    n : #items per sublist
    '''
    my_list = list
    final = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]
    return final

def joinPNG(files, append="b", output="Out.png", n=3):


    t1 = files[0]

    t1 = t1.split("/")[1:-1]
    path = reduce(lambda v1, v2:v1+"/"+v2, t1)


    path = "/" + path

    tmp1 = splitList(n=n, list=files)

    if append =="b":
        h1 = 1
        tmp2 = []
        for item in tmp1:
            ap = "+append"
            tstr = "convert "+ap+" "
            tstr += reduce(lambda v1, v2 : v1+" "+v2, item)
            tstr += " "+path+"/"+str(h1)+".png"

            os.system(tstr)
            rmv = reduce(lambda v1, v2 : v1+" "+v2, item)
            os.system("rm "+rmv)
            tmp2.append(path+"/"+str(h1)+".png")
            h1 += 1
        ap = "-append"
        tstr = "convert "+ap+" "
        tstr += reduce(lambda v1, v2 : v1+" "+v2, tmp2)
        tstr += " "+path+"/"+output
        os.system(tstr)
        rmv = reduce(lambda v1, v2 : v1+" "+v2, tmp2)
        os.system("rm "+rmv)
        return
    elif append == "v":
        ap = "-append"
    elif append == "h":
        ap = "+append"
    tstr = "convert "+ap+" "
    tstr += reduce(lambda v1, v2 : v1+" "+v2, files)
    tstr += " "+output
    os.system(tstr)
    rmv = reduce(lambda v1, v2 : v1+" "+v2, files)
    os.system("rm "+rmv)

class subgoal_model:

    def __init__(self, sg_list=None, goal=70):

        self.sg_list = sg_list
        self.goal = goal
        self.sg_model = {}
        self.trajCount = 0

        # ---- Initial Update
        for x in sg_list:
            self.sg_model[x] = {}
            self.sg_model[x]['count'] = [ 0 for _ in range(len(self.sg_list))]
            self.sg_model[x]['prob'] = [ 0 for _ in range(len(self.sg_list))]

    def update_sgList(self, new_sg):

        for x in new_sg:
            if x not in self.sg_list:
                self.sg_list.append(x)

        for x in self.sg_list:
            if x not in list(self.sg_model.keys()):
                self.sg_model[x] = {}
                self.sg_model[x]['count'] = [ 0 for _ in range(len(self.sg_list))]
                self.sg_model[x]['prob'] = [ 0 for _ in range(len(self.sg_list))]

    def update_sg_trans(self, traj):

        self.trajCount += 1
        for x in self.sg_list:
            for y in self.sg_list:
                if x != y:
                    if self.sg_in_traj(traj, x, y):
                        y_indx = self.sg_list.index(y)
                        self.sg_model[x]['count'][y_indx] += 1

        for x in self.sg_list:
            for j in range(len(self.sg_list)):
                y = self.sg_list[j]
                if x != y:
                    self.sg_model[x]['prob'][j] = self.sg_model[x]['count'][j] / float(self.trajCount)

        # ---- Normalize

        for x in self.sg_list:
            Z = sum(self.sg_model[x]['prob'])
            if Z > 0:
                for j in range(len(self.sg_list)):
                    y = self.sg_list[j]
                    if x != y:
                        self.sg_model[x]['prob'][j] = self.sg_model[x]['prob'][j] / Z

    def sg_in_traj(self, traj, sg1, sg2):

        if sg1 == self.goal:
            return False

        if sg1 not in traj:
            return False
        elif sg2 not in traj:
            return False
        elif sg1 in traj and sg2 in traj:
            i1 = traj.index(sg1)
            i2 = traj.index(sg2)
            if i2 > i1:
                if sg2 == self.goal:
                    t1 = traj[i1:i2 + 1]
                    t1 = set(list(set(t1)))
                    t2 = self.sg_list.copy()
                    t2.remove(sg1)
                    t2.remove(self.goal)
                    t2 = set(t2)
                    t3 = len(t1.intersection(t2))
                    if t3 > 0:
                        return False
                    else:
                        return True
                return True
            else:
                return False

    def sg_tran_plot(self, ppath):

        sg_list = [22, 20, 24, 40, 58, 56, 60, 70]
        keys = {22: "LM", 20: "LT", 24: "LB", 40: "M", 58: "RM", 56: "RT", 60: "RB", 70: "G"}

        labels = list(keys.values())
        x = np.arange(len(labels))


        for sg in sg_list:
            plt.clf()
            y = self.sg_model[sg]['prob']
            plt.bar(x, y, label="Subgoal: "+str(keys[sg]))
            plt.xticks(x, labels, fontsize=20)
            plt.xlabel("Subgoals", fontsize=10)
            plt.title("CPT for "+str(keys[sg]), fontsize=20)
            plt.savefig(ppath+"/"+str(sg)+".png")
            # plt.legend()
            # plt.show()

        t1 = listFiles(ppath)
        t1 = list(filter(lambda x : ".png" in x, t1))
        t1 = list(map(lambda x: ppath+"/"+x, t1))

        joinPNG(t1, n=3)

        # set_trace()

        # joinPNG()

class subgoal_model_BK:

    def __init__(self, sg_list=None):

        self.sg_list = sg_list
        self.sg_model = {}
        self.trajCount = 0

        # ---- Initial Update
        t1 = orderedPair(self.sg_list)
        for (sg1, sg2) in t1:
            self.sg_model[(sg1, sg2)] = {}
            self.sg_model[(sg1, sg2)]['count'] = 0
            self.sg_model[(sg1, sg2)]['prob'] = 0

    def update_sgList(self, new_sg):

        self.sg_list.extend(new_sg)
        t1 = orderedPair(self.sg_list)
        for (sg1, sg2) in t1:
            self.sg_model[(sg1, sg2)] = {}
            self.sg_model[(sg1, sg2)]['count'] = 0
            self.sg_model[(sg1, sg2)]['prob'] = 0

    def update_sg_trans(self, traj):

        self.trajCount += 1
        for (sg1, sg2) in list(self.sg_model.keys()):
            if self.sg_in_traj(traj, sg1, sg2):
                self.sg_model[(sg1, sg2)]['count'] += 1
                self.sg_model[(sg1, sg2)]['prob'] = float(self.sg_model[(sg1, sg2)]['count']) / self.trajCount

    def sg_in_traj(self, traj, sg1, sg2):

        if sg1 not in traj:
            return False
        elif sg2 not in traj:
            return False
        elif sg1 in traj and sg2 in traj:
            i1 = traj.index(sg1)
            i2 = traj.index(sg2)
            if i2 > i1:
                return True
            else:
                return False

def orderedPair(tmp_list):

    orderedList = []
    for x in tmp_list:
        for y in tmp_list:
            if x != y:
                orderedList.append((x, y))
    return  orderedList

class logRunTime:

    def __init__(self, init_time=None):

        self.internal_time = init_time

    def now(self):

        return time.time()

    def logTime(self):

        runtime = self.getRuntime(self.internal_time, self.now())
        self.internal_time = self.now()
        return str(round(runtime, 3))+" sec"

    def getRuntime(self, st, en):

        return round(en - st, 3)

def loadDataStr(x):

    file2 = open(x+'.pkl', 'rb')
    ds = pickle.load(file2)
    file2.close()
    return ds

def dumpDataStr(x,obj):

    afile = open(x+'.pkl', 'wb')
    pickle.dump(obj, afile)
    afile.close()

def deleteDir(dir):

    if os.path.isdir(dir): shutil.rmtree(dir, ignore_errors=False, onerror=None)

def get_one_hot(n_classes):
    target = tc.tensor([[ _ for _ in range(n_classes)]])
    y = tc.zeros(n_classes, n_classes).type(tc.int)
    y[range(y.shape[0]), target] = 1
    y = y.data.numpy()
    return y

class log:

    def __init__(self, fl):
        self.opfile = fl
        if os.path.exists(self.opfile):
            os.remove(self.opfile)
        # f = open(self.opfile, 'w')
        # f.write("test")
        # f.close()

    def writeln(self, msg):
        file = self.opfile
        print(str(msg))
        with open(file, "a") as f:
            f.write("\n"+str(msg))

    def write(self, msg):
        file = self.opfile
        print(str(msg),)
        with open(file, "a") as f:
            f.write(str(msg))

def discounted_return(reward_list):
    return_so_far = 0.0
    tot_time = reward_list.shape[0]
    tmpReturn = np.zeros(tot_time)
    k = 0
    for t in range(tot_time - 1, -1, -1):
        return_so_far = reward_list[t] + DISCOUNT * return_so_far
        tmpReturn[k] = return_so_far
        k += 1
    tmpReturn = np.flip(tmpReturn)
    return tmpReturn

def main():

    t1 = subgoal_model(sg_list=[1, 2])

    l1 = [1, 2, 3, 4, 5]

    set_trace()
    pass


# =============================================================================== #

if __name__ == '__main__':
    main()
