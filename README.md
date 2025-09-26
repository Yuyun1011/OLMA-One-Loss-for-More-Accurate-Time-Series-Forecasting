# OLMA: One Loss for More Accurate Time Series Forecasting
⚡ OLMA can replace the original loss function of any supervised  time series forecasting model to improve their performance!
## ✨ Why OLMA?
1. The entropy of random noise imposes a fundamental limit on the theoretical lower bound of time series forecasting error.
2. Most models suffer from *frequency bias* when predicting time series.  
**OLMA** changes the game:
- 🎯 Both theoretical analysis and empirical results demonstrate the existence of a unitary transformation that can reduce the marginal entropy of multivariate time series.
- 📉 Direct supervision in the frequency domain alleviates the spectral bias inherent in neural networks.
- 🛠️ Plug-and-play: works with Transformer, CNN, MLP, Mamba, Linear and even LLM-based forecasters.

> OLMA is not just a loss. It’s a **guarantee** for stronger time series forecasting.
## Abstract
Time series forecasting faces two important but often overlooked challenges. Firstly, the inherent random noise in the time series labels sets a theoretical lower bound for the forecasting error, which is positively correlated with the entropy of the labels. Secondly, neural networks exhibit a frequency bias when modeling the state-space of time series, that is, the model performs well in learning certain frequency bands but poorly in others, thus restricting the overall forecasting performance. To address the first challenge, we prove a theorem that there exists a unitary transformation that can reduce the marginal entropy of multiple correlated Gaussian processes, thereby providing guidance for reducing the lower bound of forecasting error. Furthermore, experiments confirm that Discrete Fourier Transform (DFT) can reduce the entropy in the majority of scenarios. Correspondingly, to alleviate the frequency bias, we jointly introduce supervision in the frequency domain along the temporal dimension through DFT and Discrete Wavelet Transform (DWT). This supervision-side strategy is highly general and can be seamlessly integrated into any supervised learning method. Moreover, we propose a novel loss function named OLMA, which utilizes the frequency domain transformation across both channel and temporal dimensions to enhance forecasting. Finally, the experimental results on multiple datasets demonstrate the effectiveness of OLMA in addressing the above two challenges and the resulting improvement in forecasting accuracy. The results also indicate that the perspectives of entropy and frequency bias provide a new and feasible research direction for time series forecasting.
