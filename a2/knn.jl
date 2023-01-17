include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat, X, y, k)
  (n, d) = size(X)
  (t, d) = size(Xhat)
  k = min(n, k) # To save you some debuggin

  d = distancesSquared(X, Xhat)
  yhat = ones(t)

  for i in 1:t
    di = d[:, i]
    sortedDistances = sortperm(di)
    knn = sortedDistances[1:k]
    knnLabels = y[knn]
    yhat[i] = mode(knnLabels)
  end

  return yhat
end

function knn(X, y, k)
  # Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat, X, y, k)
  return GenericModel(predict)
end
