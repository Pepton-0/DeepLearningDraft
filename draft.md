# How the mechanics works

A(n,m) = double[n][m]

Input Layer(A00 ～ A0j)
Hidden Layer(A10 ～ A1k)
...
Output Layer(An0 ～ Anl)

Aij = ReLU(w(1)A(i-1, 0) + w(2)A(i-1, 1) + ... + w(n)A(i-1, n) - biass(i))

# Learning Test

0.8 of data -> learning
0.2 of data -> test

# Learning

Loss Function
C =  1/n * Σ (Ank - T(k))²
where T is the target value

Find the minimum C using gradient descent