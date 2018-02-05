import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt

def generate_train_data(no_data_points, dim):
    """
    Generate random samples from the Normal distribution with zero-mean and unit coavriance
    for illustration purposes.
    """
    return numpy.random.randn(no_data_points, dim)

def multivariate_t_distribution(x,mu,Sigma,df,d):
    '''
    Multivariate t-student density:
    output:
        the density of the given element
    input:
        x = parameter (d dimensional numpy array or scalar)
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        d: dimension
    Ref: https://stackoverflow.com/questions/29798795/multivariate-student-t-distribution-with-python
    '''
    Num = scipy.special.gamma(1. * (d+df)/2)
    Denom = (scipy.special.gamma(1.*df/2) * numpy.power(df * numpy.pi,1.*d/2) * numpy.power(numpy.linalg.det(Sigma),1./2) * numpy.power(1 + (1./df)*numpy.dot(numpy.dot((x - mu),numpy.linalg.inv(Sigma)), (x - mu)),1.* (d+df)/2))
    res = 1. * Num / Denom 
    return res


def p_NIW(x, X_mean, X_cov, N, N0, dim):
    """
    Uses Eq 3 to compute p_NIW.
    """
    phi = numpy.eye(dim, dtype=numpy.float)
    mu_0 = numpy.zeros(dim, dtype=numpy.float)
    phi_prime = phi + X_cov + ((N0 * N) / float(N0 + N)) * numpy.dot((X_mean - mu_0).T, X_mean - mu_0)
    mu_prime = (N0 / float(N0 + N)) * mu_0 + (N / float(N0 + N)) * X_mean
    phi_prime *= (N0 + 1) / float(N0 * (N0 - dim + 1))
    df = N0 + N - dim + 1
    p = multivariate_t_distribution(x, mu_prime, phi_prime, df, dim)
    return p

def main():
    dim = 2
    N = 10
    N0 = 20
    X = generate_train_data(N, dim)
    X_cov = numpy.cov(X.T) # each column must represent a datum.
    X_mean = numpy.mean(X, axis=0)
    #print(D)
    #count, bins, ignored = plt.hist(D, 30, normed=True)
    #plt.show()
    x_test = numpy.array([2,2], dtype=numpy.float)
    prob_x = p_NIW(x_test, X_mean, X_cov, N, N0, dim)
    print(prob_x)

    pass


if __name__ == '__main__':
    main()

