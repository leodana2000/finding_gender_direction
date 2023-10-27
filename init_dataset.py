'''
Run this file to create the Trin and Test datasets.
'''
from Data.Train_Data import generate_trainset
from Data.Test_Data import generate_testset

generate_testset()
generate_trainset()