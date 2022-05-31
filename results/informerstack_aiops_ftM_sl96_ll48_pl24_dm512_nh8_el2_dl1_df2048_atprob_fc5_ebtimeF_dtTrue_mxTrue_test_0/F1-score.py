import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##设置全部数据，不输出省略号
import sys
from utils.metrics import metric
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve
dt = pd.DataFrame(columns=['value0'])
df3 = pd.read_pickle("machine-1-1_test_label.pkl")
np.set_printoptions(threshold=sys.maxsize)
dh = pd.DataFrame(columns=['mse','mae'])
mse0=[]
mae0=[]
x = np.linspace(1, 28344, num=28344)
st=[0]*28344
st1=[0]*28344
st2=[0]*28344
st3=[0]*28344
def reverse(list1):
    return list(map(lambda x: -x, list1))
def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)
def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).


    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t
def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN
def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score < threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict
ty=df3[119:-16]


for index in range(38):
    boxes = np.load('true.npy')

    #print(boxes.shape)
    boxes = boxes[:, :, index]
    #print(boxes.shape)
    boxes = np.ravel(boxes)

    boxes2 = np.load('pred.npy')
    # boxes2=np.ravel(boxes2)
    boxes2 = boxes2[:, :, index]
    boxes2 = np.ravel(boxes2)
    dt["value" + str(index)] = boxes2
    #print(len(boxes2))
    np.savetxt('boxes2.txt', boxes2)


    # plt.figure(figsize=(10,5))
    mae, mse, rmse, mape, mspe = metric(boxes, boxes2)
    #print('mse:{}, mae:{}'.format(mse, mae))
    mse0.append(mse)
    mae0.append(mae)
    plt.cla()
    plt.xticks(rotation=45)

    plt.plot(x, boxes, '-', label="true")
    outlier = []  # 将异常值保存
    outlier_x = []

    ypre = []

    IQR = boxes[np.argmax(boxes)] - boxes[np.argmin(boxes)]

    threshold1 = IQR/5
    for i in range(0, len(boxes2)):

        st[i]=st[i]+pow(abs(boxes2[i]-boxes[i]),2)
        if(index==2):
            st1[i] = st1[i] + abs(boxes2[i] - boxes[i])
        if (index == 5):
            st2[i] = st2[i] + abs(boxes2[i] - boxes[i])
        if (index == 13):
            st3[i] = st3[i] + abs(boxes2[i] - boxes[i])

        if (st[i] > threshold1 ):
            outlier.append(boxes[i])
            outlier_x.append(x[i])
            ypre.append(1)
        else:
            ypre.append(0)
            continue

dh['mse']=mse0
dh['mae']=mae0

plt.cla()

outlier1 = []  # 将异常值保存
outlier_x1 = []
ypre2 = [0]*28344

t, th = bf_search(reverse(st), df3[96:-39],
                                      start=-12,
                                      end=12,
                                      step_num=int(abs(1600) /
                                                   1),
                                      display_freq=50)
print("best f1:",t[0])
print("threshold:",th)

