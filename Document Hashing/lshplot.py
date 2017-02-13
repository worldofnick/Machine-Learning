import matplotlib.pyplot as plt

xvals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
yvals = []

for x in xvals:
	yvals.append(1-(1-(x)**(5.538))**(28.891))

plt.plot(xvals,yvals, linewidth=2.0)
plt.xlabel("JS(D1, D2)")
plt.ylabel("LSH S-Curve")
plt.draw()
plt.show()