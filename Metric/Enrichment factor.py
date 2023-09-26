
def enrichment_factor(x,y,top,decreasing=True):
    if len(x) != len(y):
        raise('The number of scores must be equal to the number of labels.')
        
    N = float(len(y))
    n = float(sum(y==1))
    
    x_prev = np.inf
    area = 0
    fp = tp = fp_prev = tp_prev = 0
    o = x.argsort()[::-1]
    for i in o:
        if (x[i] != x_prev):
            if (fp+tp)>=(N*top):
                n_right = (fp - fp_prev)+(tp - tp_prev)
                rat = (N * top - (fp_prev + tp_prev))/n_right
                tp_r = tp_prev + rat *(tp-tp_prev)
                return ((tp_r/(N*top))/(n/N))
            
            x_prev = x[i]
            fp_prev = fp
            tp_prev = tp
            
        if (y[i]==1):
            tp = tp+1
        else:
            fp = fp+1
    n_right = (fp - fp_prev)+(tp - tp_prev)
    rat = (N * top - (fp_prev + tp_prev))/n_right
    tp_r = tp_prev + rat *(tp-tp_prev)
    return ((tp_r/(N*top))/(n/N))
