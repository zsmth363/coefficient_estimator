import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

results_dir = "C:\\Users\\zhsmith\\OneDrive - Clemson University\\IM_runs\\outputs_10-29_drop\\param_index_0"

df = pd.read_excel(os.path.join(results_dir,"results.xlsx"))

cols = df.columns

cols_pred = cols[0:6]
cols_target = cols[6:]

zipped_cols = list(zip(cols_pred,cols_target))

stats = {}
for coef in zipped_cols:
    stats[coef[1]] = {}
    mae = mean_absolute_error(df[coef[0]],df[coef[1]])
    mse = mean_squared_error(df[coef[0]],df[coef[1]])
    r2 = r2_score(df[coef[0]],df[coef[1]])
    stats[coef[1]].update({"mae":mae,"mse":mse,"r2":r2})

out_df = pd.DataFrame(stats).T
out_df.to_excel(os.path.join(results_dir,"stats.xlsx"),index=True)


x_diag = [0,1,2,3,4,5,6]
y_diag = [0,1,2,3,4,5,6]


ax = df.plot(x="R1",y="R1_pred",kind="scatter",grid=True)
ax.plot(x_diag,y_diag,linestyle="--",alpha=0.5,color="k",zorder=-1)
plt.xlim([0,0.05])
plt.ylim([0,0.05])
plt.xlabel("Target Coefficient")
plt.ylabel("Predicited Coefficient")
plt.title("CIM5BL Model - Rotor Resistance Constant - $R_1$")
plt.savefig(os.path.join(results_dir,"R1.png"))

ax = df.plot(x="X1",y="X1_pred",kind="scatter",grid=True)
ax.plot(x_diag,y_diag,linestyle="--",alpha=0.5,color="k",zorder=-1)
plt.xlim([0,0.2])
plt.ylim([0,0.2])
plt.xlabel("Target Coefficient")
plt.ylabel("Predicited Coefficient")
plt.title("CIM5BL Model - Rotor Reactance Constant - $X_1$")
plt.savefig(os.path.join(results_dir,"X1.png"))

ax = df.plot(x="H",y="H_pred",kind="scatter",grid=True)
ax.plot(x_diag,y_diag,linestyle="--",alpha=0.5,color="k",zorder=-1)
upper_lim = np.round(df["H"].max())
plt.xlim([0,5])
plt.ylim([0,5])
plt.xlabel("Target Coefficient")
plt.ylabel("Predicited Coefficient")
plt.title("CIM5BL Model - Inertia Constant - $H$")
plt.savefig(os.path.join(results_dir,"H.png"))

ax = df.plot(x="a",y="a_pred",kind="scatter",grid=True)
ax.plot(x_diag,y_diag,linestyle="--",alpha=0.5,color="k",zorder=-1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Target Coefficient")
plt.ylabel("Predicited Coefficient")
plt.title("ZIP Model - Constant Current Coefficient - $a$")
plt.savefig(os.path.join(results_dir,"a.png"))

ax = df.plot(x="b",y="b_pred",kind="scatter",grid=True)
ax.plot(x_diag,y_diag,linestyle="--",alpha=0.5,color="k",zorder=-1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Target Coefficient")
plt.ylabel("Predicited Coefficient")
plt.title("ZIP Model - Constant Impedance Coefficient - $b$")
plt.savefig(os.path.join(results_dir,"b.png"))

ax = df.plot(x="c",y="c_pred",kind="scatter",grid=True)
ax.plot(x_diag,y_diag,linestyle="--",alpha=0.5,color="k",zorder=-1)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel("Target Coefficient")
plt.ylabel("Predicited Coefficient")
plt.title("ZIP Model - Constant Power Coefficient - $c$")
plt.savefig(os.path.join(results_dir,"c.png"))


