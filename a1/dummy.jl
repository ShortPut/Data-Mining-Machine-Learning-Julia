if longditude > 37.695
    if lattitude > -96.03
        return 1
    else
        return 2
    end
else
    if lattitude < -112.548
        return 1
    else
        return 2
    end
end
