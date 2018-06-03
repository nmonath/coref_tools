import argparse
import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot number of nodes vs. num nodes explored.')
    parser.add_argument('infile', type=str, help='Pruned log file.')
    parser.add_argument('k', type=int, help='k param of nsw.')
    parser.add_argument('r', type=int, help='r param of nsw.')
    args = parser.parse_args()

    num_nodes = []
    num_explored = []
    nodes_to_explored = defaultdict(list)
    with open(args.infile, 'r') as fin:
        for line in fin:
            splits = line.split('\t')
            node_stat = splits[1]
            explored_stat = splits[3]
            num_nodes.append(int(node_stat.split('=')[1]))
            num_explored.append(int(explored_stat.split('=')[1]))
            nodes_to_explored[num_nodes[-1]].append(num_explored[-1])

    plt.subplot(111)
    xs = []
    ys = []
    for nn, nes in nodes_to_explored.items():
        xs.append(nn)
        ys.append(np.mean(nes))
    plt.scatter(xs, ys, label='nsw')
    plt.scatter(xs, xs, label='exact')
    plt.legend()
    plt.title('Num Nodes vs. Num Explored, k=%s, r=%s' % (args.k, args.r))
    plt.xlabel('Num Nodes')
    plt.ylabel('Num Explored')
    plt.savefig('/tmp/num_nodes_vs_num_explored.png')
