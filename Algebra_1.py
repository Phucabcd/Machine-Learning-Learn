import numpy as np
import matplotlib.pyplot as plt

# Consider the following example:
# - Sheriff has 180km/h car
# - Bank robbers has 150km/h car anmd five-minute head start
# - How long does it take the sheriff to catch up the robber?
# (For simplicity, let's ignore, acceleration and traffic,...)
# (Note 180km/h = 3km/min & 150km/h = 2.5km/min)

t = np.linspace(0, 40, 1000) # start, finish, n points
d_robber = 2.5 * t
d_sheriff = 3 * (t - 5)

# figure, axes  
fig, ax = plt.subplots()
plt.title("The road sheriff catch robber")
plt.xlabel("time (in minutes)")
plt.ylabel("distance (in km)")

ax.set_xlim([0, 40])
ax.set_ylim([0, 100])

ax.plot(t, d_robber, c='green')
ax.plot(t, d_sheriff, c='red')

plt.axvline(x = 30, c='purple', linestyle='--')
plt.axhline(y = 75, c='purple', linestyle='--')

plt.show()