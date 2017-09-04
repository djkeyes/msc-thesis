from scipy.stats import norm

def main():

  mean = 0.5
  sigma = 0.01
  n = 1000

  data = norm.rvs(mean, sigma, size=n)
  data = 1./data

  mu, std = norm.fit(data)
  print('mean of reciprocal: ', mu)
  print('sigma of reciprocal: ', std)
  print('orig sigma / mean^2', sigma/(mean*mean))

if __name__=="__main__":
  main()