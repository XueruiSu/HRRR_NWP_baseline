

def online_mean_variance(data):  
    n = 0  
    mean = 0  
    M2 = 0  
  
    for x in data:  
        n += 1  
        delta = x - mean  
        mean += delta / n  
        delta2 = x - mean  
        M2 += delta * delta2  
  
    if n < 2:  
        return float('nan'), float('nan')  
    else:  
        variance = M2 / (n - 1)  
        return mean, variance  
  
data = [2, 4, 6, 8, 10]  
mean, variance = online_mean_variance(data)  
print("Mean:", mean)  
print("Variance:", variance)  
