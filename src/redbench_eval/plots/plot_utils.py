import matplotlib.pyplot as plt
import seaborn as sns


def config_plotting():
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
