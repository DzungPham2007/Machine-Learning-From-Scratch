def cost_function(x,y,w,b):
    m = x.shape[0]
    cost=0
    for i in range(m):
        f_wb=np
        cost = cost + (w*x[i] + b - y_train[i])**2
    return (1 / (2 * m)) * cost

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    d_dw = 0
    d_db = np.array
    for i in range(m):
        d_db += w*x[i] + b - y[i]
        d_dw += (w*x[i] + b - y[i])*x[i]
    d_db /= m
    d_dw /= m
        
    return d_dw, d_db

def gradient_descent(a,x,y,w,b,num_iters):
    for _ in range(num_iters):
        d_dw, d_db = compute_gradient(x,y,w,b)
        w = w - a*d_dw
        b = b - a*d_db
    return w,b
