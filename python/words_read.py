# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:25:41 2024

@author: wandering_leaf
"""
def dh_list(file):
    
    with open(file,"r",encoding='ISO-8859-1') as f: 
        f.seek(0) #把指针移到文件开头位置
        result=[]
        for line in f.readlines():#readlines以列表输出文件内容
            line=line.replace(",","").replace("\n","")#改变元素，去掉，和换行符\n,tab键则把逗号换成"/t",空格换成" "
            result.append(line)
    return result

file_location = r"E:\code\meisai\source\words.txt"
words_data=dh_list(file_location)
print(words_data)