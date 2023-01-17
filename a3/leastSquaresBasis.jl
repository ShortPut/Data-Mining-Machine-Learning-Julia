include("misc.jl")

function leastSquaresBasis(x, y, p)
    Z = polyBasis(x, p)

    w = Z'Z \ Z'y

    function predict(Xhat, p)
        Z = polyBasis(Xhat, p)
        return Z * w
    end
    return GenericModel(predict)
end

function polyBasis(x, p)
    n = length(x)
    A = ones(n, 1)
    Z = A
    for i in 1:p
        Z = hcat(A, x .^ i)
        A = Z
    end
    return Z

end