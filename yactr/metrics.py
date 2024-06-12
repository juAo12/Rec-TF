from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict


def evaluate_metrics(y_true, y_pred, metrics, group_id=None):
    return_dict = OrderedDict()
    group_metrics = []

    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            return_dict[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            return_dict[metric] = roc_auc_score(y_true, y_pred)
        elif metric in ["gAUC", "avgAUC", "MRR"] or metric.startswith("NDCG"):
            return_dict[metric] = 0
            group_metrics.append(metric)
        else:
            raise ValueError("metric={} not supported.".format(metric))

    # 针对 "gAUC", "avgAUC", "MRR"
    if len(group_metrics) > 0:
        assert group_id is not None, "group_index is required."
        metric_funcs = []
        for metric in group_metrics:
            try:
                metric_funcs.append(eval(metric))
            except:
                raise NotImplementedError('metrics={} not implemented.'.format(metric))
        score_df = pd.DataFrame({"group_index": group_id,
                                 "y_true": y_true,
                                 "y_pred": y_pred})
        results = []
        pool = mp.Pool(processes=mp.cpu_count() // 2)
        for idx, df in score_df.groupby("group_index"):
            results.append(pool.apply_async(evaluate_block, args=(df, metric_funcs)))
        pool.close()
        pool.join()
        results = [res.get() for res in results]
        sum_results = np.array(results).sum(0)
        average_result = list(sum_results[:, 0] / sum_results[:, 1])
        return_dict.update(dict(zip(group_metrics, average_result)))

    return return_dict

def evaluate_block(df, metric_funcs):
    res_list = []
    for fn in metric_funcs:
        v = fn(df.y_true.values, df.y_pred.values)
        if type(v) == tuple:
            res_list.append(v)
        else: # add group weight
            res_list.append((v, 1))
    return res_list