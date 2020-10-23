#!/usr/bin/python3
import sys
n = int(sys.argv[1])
b = [0.0]*n*n*n
for i in range (n):
    for j in range (n):
        for k in range (n):
            b[i*n*n+j*n+k] = 1.1*(i+j+k)
a = [0.0]*n*n*n
for i in range (1,n-1):
    for j in range (1,n-1):
        for k in range (1,n-1):
            a[i*n*n + j*n + k] = 0.8*(b[(i-1)*n*n+j*n+k]+b[(i+1)*n*n+j*n+k]+b[i*n*n+(j-1)*n+k]+b[i*n*n+(j+1)*n+k]+b[i*n*n+j*n+(k-1)]+b[i*n*n+j*n+(k+1)]);
result = 0.0
for i in range (1,n-1):
    for j in range (1,n-1):
        for k in range (1,n-1):
            if((i+j+k)%2):
                result += a[i*n*n + j*n + k]
            else:
                result -= a[i*n*n + j*n + k]
print("Result is ", result)
