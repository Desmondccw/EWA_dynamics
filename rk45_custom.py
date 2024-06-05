import numpy as np

# Custom RK_45 Implementation with adaptive step size near the boundary of the simplex

def rk45(deriv, y0, t_span, tol=1e-6, h_max=0.1):
    t0, tf = t_span
    t = t0
    y = np.array(y0)
    h = h_max
    t_values = [t]
    y_values = [y]

    while t < tf:
        if t + h > tf:
            h = tf - t

        k1 = h * deriv(t, y)
        k2 = h * deriv(t + h / 4, y + k1 / 4)
        k3 = h * deriv(t + 3 * h / 8, y + 3 * k1 / 32 + 9 * k2 / 32)
        k4 = h * deriv(t + 12 * h / 13, y + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197)
        k5 = h * deriv(t + h, y + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104)
        k6 = h * deriv(t + h / 2, y - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40)

        y4 = y + 25 * k1 / 216 + 1408 * k3 / 2565 + 2197 * k4 / 4104 - k5 / 5
        y5 = y + 16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55

        error = np.linalg.norm(y5 - y4) + 1e-20

        if np.all((y5 > 0) & (y5 <= 1)) == False : 
            dyy= np.array(y/k1 * h) 
            neg_dyy = dyy[dyy < 0]
            
            h = -np.max(neg_dyy)/ 5
        
        if error < tol:
            t += h
            y = y5
            t_values.append(t)
            y_values.append(y)
        
        h = h * min(max(0.1, 0.8 * (tol / error)**0.2), 5.0) 

        if h > h_max:
            h= h_max

    return np.array(t_values), np.array(y_values)