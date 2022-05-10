# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 09:45:47 2022

@author: mthff
"""
import pandas as pd
from collections import namedtuple
from operator import itemgetter
from pprint import pformat
import math
from queue import PriorityQueue
from sklearn.model_selection import train_test_split


#The value of K in KNN
k=2

#Delcared Priority Queue to store the K nearest nodes
pq = PriorityQueue()

#Node class of KD-tree
class Node(namedtuple("Node", "location left_child right_child")):
    def __repr__(self):
        return pformat(tuple(self))
    

#build KD-Tree    
def kdtree(point_list, depth: int=0):
    if not point_list:
        return None
    
    size = len(point_list[0])
    
    axis = depth % size
    
    point_list.sort(key = itemgetter(axis))
    median = len(point_list) // 2
    
    return Node(
        location = point_list[median],
        left_child = kdtree(point_list[:median], depth+1),
        right_child = kdtree(point_list[median+1:], depth+1)
    )
        

def push_pq(dist, point1, point2):
    #print("\n",-dist)
    #insert dist in minus because pq sort in increasing order
    pq.put((-dist, (point1, point2)))
    if(pq.qsize()>k):
        pq.get()
        #print("pq ",pq.get()," pq\n")
        
#eucledean distance between two nodes 
def eu_distance(x,y,p,q):
    return math.sqrt((x-p)**2+(y-q)**2)


def KNN(recur_tree, p, q, layer):
    #print(recur_tree)
    
    global pq
    
    if(recur_tree==None):return (1e9,(-1,-1))
    
    #print(recur_tree)
    #save data of a node's x and y coordinates.
    x = recur_tree[0][0] 
    y = recur_tree[0][1]
    #print("x y: ",x,y)
    #calculate eucledean distance between the given point and a node.
    dis = eu_distance(x,y,p,q)
    #push the distance and coordinates into the priority qu to save the nearest k points from the searching point.
    push_pq(dis,x,y)
    
    #base point of recursion
    if (recur_tree.left_child == None and recur_tree.right_child == None): 
        #print(dis)
        return (dis, (x, y))
    
    #take data of left child and right child
    left = recur_tree.left_child
    right = recur_tree.right_child
    #print((p,q),recur_tree[0])
    
    #level X
    dist = 0
    if(layer%2==0):
        comp = x
       # print ("0 ",comp)
        if(comp<p):
            (dist, (point1, point2)) = KNN(right, p, q, layer+1)
            #check the special case of kdtree
            if(dist > abs(y-q)):
                (dist_prime, (point1, point2)) = KNN(left, p, q, layer+1)
                return (min(dist, dist_prime), (point1, point2))
            else:
                return (dist, (point1, point2))
        else:
            (dist, (point1, point2)) = KNN(left, p, q, layer+1)
            #check the special case of kdtree
            if(dist > abs(y-q)):
                (dist_prime, (point1, point2)) = KNN(right, p, q, layer+1)
                return min(dist, dist_prime), (point1, point2)
            return (dist, (point1, point2))
    #level Y    
    else:
        comp = y
        #print ("1 ", comp)
        if(comp<q):
            (dist, (point1, point2)) = KNN(right, p, q, layer+1)
            #check the special case of kdtree
            if(dist > abs(x-p)):
                (dist_prime, (point1, point2)) = KNN(left, p, q, layer+1)
                return (min(dist, dist_prime), (point1, point2))
            else:
                return (dist, (point1, point2))
                        
        else:
            dist, (point1, point2) = KNN(left, p, q, layer+1)
            #check the special case of kdtree
            if(dist > abs(x-p)):
                (dist_prime, (point1, point2)) = KNN(right, p, q, layer+1)
                return min(dist, dist_prime), (point1, point2)
            return (dist, (point1, point2))

    
    
    
df = pd.read_csv('H:\diabetes.csv')
target = df['Outcome']
df.drop('Outcome',axis=1, inplace=True)

df.drop(['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','Age'],axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)



records = X_train.to_records(index=False)
points = list(records)

#points = [(5,4), (2,6), (13, 3), (3, 1), (10, 2), (8,7)]

def main():
    tree = kdtree(points)
    #print(tree)
    
    p = 2
    q =  5
    KNN(tree, p, q, 0)
    
    # print("\n\n\n")
    
    print(pq.qsize())
    while pq.qsize():
        dist, (n1, n2) = pq.get()
        print(n1,n2)
    
    
if __name__=="__main__":
    main()