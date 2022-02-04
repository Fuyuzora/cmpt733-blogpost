import pickle
import matplotlib.pyplot as plt
import pandas as pd

acc_dict = {}
with open('accuracy.pkl', 'rb') as f:
    acc_dict = pickle.load(f)
data = pd.DataFrame(acc_dict)
cols = data.columns.values
data = data.explode([col for col in cols])
data.plot(legend=True).set_title("Comparions between basic CNN architectures")
plt.grid(color = 'grey', ls = '--', lw = 0.5, alpha=0.5)
plt.show()