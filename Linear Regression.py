# LinearRegression Session 1
import numpy as np
import matplotlib.pyplot as plt
def estimate_coef(x, y):     # Recieve the value of x & y
# no. of observation /points.
 n = np.size(x)
# means of x & y vector
 m_x = np.mean(x)
 m_y= np.mean(y)
# calculating cross-deviation and deviation about x
 SS_xy =np.sum(x*y)-n*m_x*m_y
 SS_xx= np.sum(x*x)- n*m_x * m_x
# calculation regression coefficients
 m = SS_xy/SS_xx
 c = m_y-m*m_x
 return[m,c]
def plot_regression_line(x, y, b):
# plotting the actual points scatter plot
 plt.scatter(x, y, color='m', marker='o', s=30)
# predicted response vector
 y_pred = b[0] + b[1] * x
# plotting the regression line
 plt.scatter(x, y_pred, color='g')
 plt.plot(x, y_pred, color='b')
# putting labels

 plt.xlabel('-----x---->')
 plt.ylabel('-----y---->')

 plt.title('Linear Regression')
# function to show plot
 plt.show()
def startAIAlgorithm():
 # observations
 x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])      # x is input historic training data & y is historic output training data
 y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

 # estimating coefficients
 m, c= estimate_coef(x, y)  #Sender Values or function calling
 print("Estimate coefficients:\n")
 print("slop m=",m)
 print("y -intercept c=",c)
 print("y=",m,"*x+",c)

 # plotting regression line
 plot_regression_line(x, y, [c, m])
if __name__=="__main__":    # in ML main function of program
  startAIAlgorithm()