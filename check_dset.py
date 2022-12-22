# coding=utf-8

''' 查看各个教学楼的数量
目前(2022/12/05):
1教：57张
2教：28张
3教：65张
4教：34张
5教：1张
6教：15张
7教：27张
8教：11张
9教：8张
10教：51张
'''

import os

def print_count(dir: str) -> None:
    '''
    '''
    imgs = os.listdir(dir)
    print(dir)
    for i in range(1, 11):
        count = len([v for v in imgs if v.startswith(f"J{i:02d}-")])
        print(f"{i}教：{count}张")


if __name__ == "__main__":
    print_count('train')
    print_count('test')