function GCV(hyp,y,ΦWd,evol)
    
    λ       = hyp[1]
    push!(evol,λ)
    N,K     = size(ΦWd)
    F       = svd(ΦWd,full=true)
    U       = F.U 
    S       = zeros(N)
    S[1:K]  = F.S

    z       = U'*y

    nom     =  1/N * sum( (N*λ ./ (S   .+ N*λ)).^2 .* z.^2 )
    den     = (1/N * sum(  N*λ ./ (F.S .+ N*λ)) + N - K)^2

    return nom/den

end