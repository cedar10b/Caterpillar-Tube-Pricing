### **Kaggle Caterpillar Tube Pricing Competition**

# *Summary*

The goal of this project is to predict the price of industrial tube assemblies that Caterpillar is using in its construction and mining equipment. Currently, Caterpillar relies on a variety of suppliers to manufacture these tube assemblies, each having their own unique pricing model. The competition provides detailed information about the physical characteristics of the tubes and their components, as well as annual usage data. The goal of the competition is to develop a machine learning model that minimizes the Root Mean Squared Logarithmic Error (RMSLE):

\begin{equation}
\sqrt(\frac{1}{n} \displaystyle\sum_{i=1}{n} \left(log(p_{i} + 1) - log(a_{i} + 1) \right)^{2})
\end{equation}

where:

$p_{i}$ is the predicted price
$a_{i}$ is the actual price
$n$ is the number of price quotes in the test test (about 30000)


