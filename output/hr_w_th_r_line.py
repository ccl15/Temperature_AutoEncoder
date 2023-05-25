"""
Singal station different hour and weight line plot.
"""


def recall_rate(good, bad, weight):
    mean = np.mean(good)
    std = np.std(good)
    th = mean + weight * std
    # th = mean*weight
    r =  sum(bad <= th)/len(bad)
    # print(weight, mean, std, th, r)
    return th


def recall_plot():
    weight = [0,0.1,0.2]
    # weight = [1,1.5]
    MAE_good = [ [MAE(pred_g[i][-n:], obs_g[i][-n:]) for i in range(len(obs_g))]  for n in range(1,11)] 
    MAE_bad  = [ [MAE(pred_b[i][-n:], obs_b[i][-n:]) for i in range(len(obs_b))]  for n in range(1,11)] 
    colors = ['#0D47A1', '#1E88E5', '#64B5F6']    
    for w in range(len(weight)):
        recall = [recall_rate(MAE_good[i], MAE_bad[i], weight[w]) for i in range(10)]
        plt.plot(recall, c=colors[w], label=f'+{weight[w]} std')
    
    plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.9))
    plt.xticks(range(10), range(1,11) ,fontsize = 10)
    plt.savefig('fig/threshold_std_24.png', bbox_inches='tight', dpi = 200)
    
# recall_plot()