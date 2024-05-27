import numpy as  np
import matplotlib.pyplot as plt

def plot_bar_blue():
    names = ['dx,dy', 'dx,dz', 'H1', 'H2', 'H3']

    acqr_coverages = [ 0.82352, 0.85714,  0.676470, 0.70588, 0.61764705]
    qr_coverages = [0.67647, 0.08571, 0.0294117, 0.0882, 0.0]

    # multiply by 100
    acqr_coverages = [x * 100 for x in acqr_coverages]
    qr_coverages = [x * 100 for x in qr_coverages]

    # plot as paired bar graph
    fig, ax = plt.subplots(figsize=(8, 3))
    barWidth = 0.25
    r1 = np.arange(len(acqr_coverages))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # plot with shaded color
    # plt.bar(r1, ensemble_coverages, color="grey", width=barWidth, edgecolor='grey', label='Ensemble')

    plt.bar(r1, qr_coverages, color="#ef38ac", width=barWidth, edgecolor='grey', label='QR')

    plt.bar(r2, acqr_coverages, color="#f4b400", width=barWidth, edgecolor='grey', label='ACQR')
    plt.xlabel('Annotator of Intuitive Input')
    plt.ylabel('Coverage')
    plt.ylim(0, 102)
    plt.title('Coverage of ACQR and QR on each annotator of human input')
    plt.xticks([r + barWidth for r in range(len(acqr_coverages))], names)
    # plt.legend()

    # remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # remove border around legend
    # ax.legend(frameon=False)

    plt.savefig('blue_coverage_acqr_qr_ensemble.png')
    plt.close()
def plot_bar_red():
    names = ['dx,dy', 'dx,dz', 'H1', 'H2', 'H3']

    acqr_coverages = [0.8888888, 0.8, 0.818181, 0.787878, 0.79797979]
    qr_coverages = [0.373737, 0.29, 0.1616, 0.20202, 0.0303]

    # multiply by 100
    acqr_coverages = [x * 100 for x in acqr_coverages]
    qr_coverages = [x * 100 for x in qr_coverages]

    # plot as paired bar graph
    fig, ax = plt.subplots(figsize=(8, 3))
    barWidth = 0.25
    r1 = np.arange(len(acqr_coverages))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # plot with shaded color
    # plt.bar(r1, ensemble_coverages, color="grey", width=barWidth, edgecolor='grey', label='Ensemble')

    plt.bar(r1, qr_coverages, color="#ef38ac", width=barWidth, edgecolor='grey', label='QR')

    plt.bar(r2, acqr_coverages, color="#f4b400", width=barWidth, edgecolor='grey', label='ACQR')
    plt.xlabel('Annotator of Intuitive Input')
    plt.ylabel('Coverage')
    plt.ylim(0, 102)
    plt.title('Coverage of ACQR and QR on each annotator of human input')
    plt.xticks([r + barWidth for r in range(len(acqr_coverages))], names)
    # plt.legend()

    # remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # remove border around legend
    # ax.legend(frameon=False)

    plt.savefig('red_coverage_acqr_qr_ensemble.png')
    plt.close()

if __name__ == '__main__':
    plot_bar_blue()
    plot_bar_red()





