## Linear Regression Model
### Multiple Linear function:

#### $$\hat{y} = f_{\vec{w},b} (\vec{x}) = \vec{w} \times \vec{x} + b = w_1 \times x_1 + w_2 \times x_2 + ... + w_n \times x_n+b$$


Prameter:  $\quad \vec{w} = [w_1 \; w_2 \; w_3 \; ... \; w_n]$ , b <br>
Input: $\quad \vec{x} = [x_1 \; x_2 \; x_3 \; ... \; x_n]$

### Square error cost function:
#### $$J_{\vec{w},b} = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)}-y^{(i)})^2$$
Number of training: $m$ <br>
Predict Output: $\hat{y}$ <br>
Actual Output: $y$

### Goal:
#### $$\min_{\vec{w},b} J_{\vec{w},b}$$

### Gradient Descent
#### $$w_j \; := \; w_j - \alpha.\frac{\partial}{\partial w_j} J_{\vec{w},b}$$
#### $$b \; := \; b - \alpha.\frac{\partial}{\partial b} J_{\vec{w},b}$$
$j \; = \; 1 \; .. \;m$ <br>
$\alpha$ : Learning rate $\; (0<\alpha<1)$ <br>
"a := b"  change the value of a to b

### Compute the derivative
#### $$\frac{\partial}{\partial w_j} J_{\vec{w},b} \; = \; \frac{1}{m} \sum_{i=1}^{m}[(\hat{y}^{(i)} - y^{(i)}) \times x_j^{(i)}] $$
#### $$\frac{\partial}{\partial b} J_{\vec{w},b} \; = \; \frac{1}{m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})$$
