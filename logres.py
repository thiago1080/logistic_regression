
def linear(x,w,b):
    lin = np.dot(x.T, w) + b
    #print(f'{lin.shape = }')
    return lin
def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    #print(f'{sig.shape = }')
    return sig
def a(x,w,b):
    return sigmoid(linear(x,w,b))
def cross_entropy_loss(y_true, y_pred):
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
def dz(x,w,b, y):
    return a(x,w,b)-y
def dw(w,b,x,y):
    #print(f'{x.shape = }')
    #print(f'{w.shape = }')
    #print(f'{b.shape = }')
    #print(f'{y.shape = }')
    dz_ = dz(x,w,b,y)
    #print(f'{dz_.shape = }')
    dw_ = x.T * dz_
    #print(f'{dw_ = }')
    return dw_
def db(w,b,x,y):
    return dz(x,w,b,y).T
def gradient_descent_step(w,b,alpha, x,y):
    #print(w.shape)
    #print(x.shape)
    #print(y.shape)

    termo =  dw(w,b,x,y)*alpha
    #print(f'{termo.shape = }')
    tmp_w = w - termo.reshape(4,1)
    tmp_b = b - db(w,b,x,y) * alpha
    #print(f'{tmp_w = }')
    #print(f'{tmp_b = }')
    return tmp_w,tmp_b
# x = np.linspace(0, 1, 5)
# y = 3 * x + 2 + np.random.randn(5) * 0.1
# Ensure matplotlib is installed
w = np.array([[0.1], [0.2], [0.3], [0.4]])
b = np.array([[0.5]])
alpha = 0.01
sum_w = np.array([[0.0], [0.0], [0.0], [0.0]])
sum_b = np.array([[0.0]])
for epoch in range(10000):
    losses = 0  # Reset losses at the start of each epoch
    for i in range(len(X)):
        losses += cross_entropy_loss(y[i], a(X[i].reshape(-1, 1), w, b))
        w_, b_ = gradient_descent_step(w, b, alpha, X[i].reshape(-1, 1), y[i].reshape(1, 1))
        sum_w += w_
        sum_b += b_
    losses /= len(X)
    print(f'{epoch} : {losses[0][0] = :.2f}')
    w_ = sum_w / len(X)
    b_ = sum_b / len(X)
    w = w - w_ * alpha
    b = b - b_ * alpha
    if losses < 0.5:
        break
print(f'{losses = }')
print(f'{w = }')
print(f'{b = }')

def classify(x,w,b):
    pred = a(x,w,b)
    pred = np.where(pred > 0.5, 1, 0)
    return pred
for x in iris.data:
    pred = classify(x, w, b)
    print(f'Predicted: {pred}, Actual: {iris.target[iris.data.tolist().index(x.tolist())]}')