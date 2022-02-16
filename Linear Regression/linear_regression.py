def model_lin_reg(x, y): # x and y are arrays of the same length
    n=len(x)

    # Error handling
    if len(x) != len(y):
        print("Array x and y must be the same length.")
        return (0,0)
    
    # Summation
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sqr = sum(k**2 for k in x)
    sum_xy = sum(x[k]*y[k] for k in range(0, n))
 
    #slope formula derived from least squares method
    m=(n*sum_xy-sum_x*sum_y)/(n*sum_x_sqr-(sum_x)**2)
    b = (sum_y-m*sum_x)/(n)

    return (m, b)
