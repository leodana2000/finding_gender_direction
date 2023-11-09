'''
Run this file to create the Trin and Test datasets.
'''
from Data.Train_Data import generate_trainset_v1, generate_trainset_v2
from Data.Test_Data import generate_testset

generate_testset()
generate_trainset_v1()
generate_trainset_v2()