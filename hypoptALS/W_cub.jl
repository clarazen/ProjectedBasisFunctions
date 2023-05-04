function cubicKernel(s::Float64)
    if 0 ≤ abs(s) < 1
        u = 1.5*abs(s)^3 - 2.5*abs(s)^2 + 1
    elseif 1 ≤ abs(s) < 2
        u = -0.5*abs(s)^3 + 2.5*abs(s)^2 - 4*abs(s) + 2
    else
        u = 0.0
    end
    return u
end

function cubicWeights(point::Float64, xInterp::AbstractArray{Float64}, nInterp::Int64, h::Float64)
    k = findInterpIndex(point, xInterp)
    s = (point - xInterp[k])/h
    sdim = [s+1, s, s-1, s-2]
    u = cubicKernel.(sdim)
    # if-else with boundary conditions
    if k == 1
        j = [k k+1 k+2 k+3]
        u = [u[2]+3u[1] u[3]-3u[1] u[1]+u[4] 0.0]
    elseif k == nInterp-1
        j = [k-2 k-1 k k+1]
        u = [0.0 u[1]+u[4] u[2]-3u[4] u[3]+3u[4]]
    elseif k == nInterp
        j = [k-3 k-2 k-1 k]
        u = [0.0 0.0 0.0 u[2]]
    else
        j = [k-1 k k+1 k+2]
        u = [u[1] u[2] u[3] u[4]]
    end
    return j, u
end

function findInterpIndex(point::Float64, xInterp::AbstractArray{Float64})
    xDifference = point .- collect(xInterp)
    minVal = Inf
    k = undef
    for i in eachindex(xDifference)
        if 0 <= xDifference[i] < minVal
            minVal = xDifference[i]
            k = i
        end
    end
    return k
end

function interpMatrixDim(Xvar, Xinducing, deg)
    h = (maximum(Xinducing) - minimum(Xinducing)) / (length(Xinducing) .- 1)
    Jdim = Array{Int64}(undef, length(Xvar), deg+1)
    Cdim = Array{Float64}(undef, length(Xvar), deg+1)
    if deg == 3
        for i in eachindex(Xvar)
            Jdim[i,:], Cdim[i,:] = cubicWeights(Xvar[i], Xinducing, length(Xinducing), h)
        end
    elseif deg == 5
        for i in eachindex(Xvar)
            Jdim[i,:], Cdim[i,:] = quinticWeights(Xvar[i], Xinducing, length(Xinducing), h)
        end
    end
    return Jdim, Cdim
end

function interpMatrix(Xvar, Xinducingdim, deg)
    JCdim = [interpMatrixDim(Xvar[:,d], Xinducingdim[d], deg) for d in eachindex(Xinducingdim)]
    Jdim = [JCdim[d][1] for d in eachindex(JCdim)]
    Cdim = [JCdim[d][2] for d in eachindex(JCdim)]
    Idim = [repeat(1:size(Xvar,1), 1, size(Cdim[d],2)) for d in eachindex(Cdim)]
    inducingdims = length.(Xinducingdim)
    #J = undef
    #C = undef
    #for d in eachindex(Xinducingdim)
    #    Jd = Jdim[d]
    #    Cd = Cdim[d]
    #    if d == 1
    #        J = Jd
    #        C = Cd
    #    else
    #        pd = prod(inducingdims[1:d-1])
    #        J = repeat(J, 1, deg+1) + repeat((Jd .- 1) .* pd, inner = (1,size(J,2)))
    #        C = repeat(C, 1, deg+1) .* repeat(Cd, inner = (1,size(C,2)))
    #    end
    #end
    Wdim = [sparse(vec(Idim[d]), vec(Jdim[d]), vec(Cdim[d]), size(Xvar,1), inducingdims[d]) for d in eachindex(Xinducingdim)]
    #Isp = repeat(1:size(Xvar,1), 1, size(C,2))
    #W = sparse(vec(Isp), vec(J), vec(C), size(Xvar,1), prod(inducingdims))
    return Wdim
end
