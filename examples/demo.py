"""plotlibs demo — same code style as matplotlib."""
import numpy as np
import plotlibs as pl

x = np.linspace(0, 10, 1000)
pl.close()
pl.plot(x, np.sin(x), label="sin")
pl.plot(x, np.cos(x), label="cos")
pl.xlabel("x"); pl.ylabel("y"); pl.title("plotlibs demo")
pl.legend(); pl.grid(True)
pl.savefig("demo.png")
print("saved demo.png")
