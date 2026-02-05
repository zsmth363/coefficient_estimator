import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path_to_file = ""

# df = pd.read_excel("")
df0 = pd.read_excel(os.path.join(path_to_file,"0_outputs.xlsx"))
df1 = pd.read_excel(os.path.join(path_to_file,"1_outputs.xlsx"))
df2 = pd.read_excel(os.path.join(path_to_file,"2_outputs.xlsx"))
df3 = pd.read_excel(os.path.join(path_to_file,"3_outputs.xlsx"))

# ax = df.plot(x="hidden_dim",y="R^2",kind="bar",zorder=2)
# ax.grid(True,which="both",zorder=0)
# plt.xlabel("DNN Hidden Dimension")
# plt.ylabel("Average $R^2$ Score")
# plt.ylim([0.7,1])
# plt.legend(["$R^2$"])
# plt.title("$R^2$ Score as a Function of DNN Hidden Dimension")
# plt.show()

plt.figure()
plt.plot(df3["Time(s)"],df3["PLOD 9[LOAD B 230.00]1"],label="Actual (ZIP)")
plt.plot(df0["Time(s)"],df0["PLOD 9[LOAD B 230.00]1"],linestyle='--',label="Predicted (ZIP) - Noise")
plt.plot(df1["Time(s)"],df1["PLOD 9[LOAD B 230.00]1"],linestyle='--',label="Predicted (ZIP) - Sampling")
plt.plot(df2["Time(s)"],df2["PLOD 9[LOAD B 230.00]1"],linestyle='--',label="Predicted (ZIP) - Default")

plt.plot(df3["Time(s)"],df3["PLOD 9[LOAD B 230.00]2"],label="Actual (CIM5BL)")
plt.plot(df0["Time(s)"],df0["PLOD 9[LOAD B 230.00]2"],linestyle='--',label="Predicted (CIM5BL) - Noise")
plt.plot(df1["Time(s)"],df1["PLOD 9[LOAD B 230.00]2"],linestyle='--',label="Predicted (CIM5BL) - Sampling")
plt.plot(df2["Time(s)"],df2["PLOD 9[LOAD B 230.00]2"],linestyle='--',label="Predicted (CIM5BL) - Default")
# plt.xlim([0.95,1.25])
plt.title("Predicted and Actual Transient Stability Simulation")
plt.xlabel("Time [s]")
plt.ylabel("Real Power [p.u.]")
plt.legend()
plt.show()


# df["Pagg"] = df["PLOD 9[LOAD B 230.00]2"] + df["PLOD 9[LOAD B 230.00]1"]


# ax = df.plot(x="VOLT 9 [LOAD B 230.00]",y="Pagg",kind="scatter",grid=True)
# # plt.xlim([0,0.05])
# # plt.ylim([0,0.05])
# plt.xlabel("Voltage [p.u]")
# plt.ylabel("Power [p.u.]")
# plt.title("Aggregated Load PV Curve")
# plt.savefig("C:\\Users\\zhsmith\\OneDrive - Clemson University\\IM_runs\\figures\\agg_PV_curve.png")
# plt.show()