include("misc.jl")

function leastSquaresGauss(X, y, s)
    Z = gaussBasis(X, s)
    w = Z'Z \ Z'y

    function predict(X, Xhat, s)
        Zhat = gaussBasisPred(X, Xhat, s)
        return Zhat * w
    end
    return GenericModel(predict)
end

function gaussBasis(X, s)
    n = length(X)
    Z = ones(n, n)
    for i in 1:n
        for j in 1:n
            Z[i, j] = exp(-((X[i] - X[j])^2) / (2 * (s^2)))
        end
    end
    return Z
end

function gaussBasisPred(X, Xhat, s)
    n = length(X)
    m = length(Xhat)
    Zhat = ones(m, n)
    for i in 1:m
        for j in 1:n
            Zhat[i, j] = exp(-((Xhat[i] - X[j])^2) / (2 * (s^2)))
        end
    end
    return Zhat
end