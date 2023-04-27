

A = TT([randn(1,3,3,2),randn(2,3,3,2),randn(2,3,3,1)])
B = TT([randn(1,3,3,2),randn(2,3,3,2),randn(2,3,3,1)])
Amat = ttm2mat(A)
Bmat = ttm2mat(B)

C = TT([A[1].*B[1],A[2].*B[2],A[3].*B[3]])
Cmat = ttm2mat(C)