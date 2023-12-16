import random
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def string2array(s):
    arr = []

    for i in range(0,len(s)):
        arr.append(int(s[i]))
    return arr

def array2string(arr):
    # print("Hi")
    # print(len(arr))
    s = ""
    for ele in arr:
        s += chr(ele)
    return s
    # arr = arr.astype('string')
    # return "".join([chr(item) for item in arr])