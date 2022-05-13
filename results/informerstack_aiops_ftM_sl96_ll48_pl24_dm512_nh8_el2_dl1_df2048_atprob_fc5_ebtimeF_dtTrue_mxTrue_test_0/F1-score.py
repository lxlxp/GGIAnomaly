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

ty=df3[119:-16]
#for i in range(0, 28344):
    #if(ty[i]==1):
        #print(",")
        #print(i)

for index in range(38):
    boxes = np.load('true.npy')

    # boxes=np.ravel(boxes)

    # print(X[1:3,1:3])

    #print(boxes.shape)
    boxes = boxes[:, :, index]
    #print(boxes.shape)
    boxes = np.ravel(boxes)

    #print(len(boxes))

    # np.savetxt('boxes.txt', boxes)

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
    #statistics = (pd.Series(boxes2.tolist())).describe()
    #IQR = statistics.loc('max') - statistics.loc('min')
    IQR = boxes[np.argmax(boxes)] - boxes[np.argmin(boxes)]
    #print("...")
    #print(IQR)
    threshold1 = IQR/5
    for i in range(0, len(boxes2)):


        st[i]=st[i]+pow(abs(boxes2[i]-boxes[i]),2)
        if(index==2):
            st1[i] = st1[i] + abs(boxes2[i] - boxes[i])
        if (index == 5):
            st2[i] = st2[i] + abs(boxes2[i] - boxes[i])
        if (index == 13):
            st3[i] = st3[i] + abs(boxes2[i] - boxes[i])
        #if (boxes[i] < threshold1) | (boxes[i] > threshold2):
        if (st[i] > threshold1 ):
            #print(q1)
            outlier.append(boxes[i])
            # outlier_x.append(data_x[i])
            outlier_x.append(x[i])
            ypre.append(1)
        else:
            ypre.append(0)
            continue


dh['mse']=mse0
dh['mae']=mae0
#dh.to_csv("metric.csv")
plt.cla()
#plt.xticks(rotation=45)
#plt.plot(x, st, '-', label="true")

#plt.show()
outlier1 = []  # 将异常值保存
outlier_x1 = []
ypre2 = [0]*28344
for i in range(0, len(boxes2)):

    if (st[i] > 0.5 ):





        #ypre2.append(1)
        if(i>40 and i<28430):
            outlier1.append(st[i])
            # outlier_x.append(data_x[i])
            outlier_x1.append(x[i])
            for j in range(0, 40):
                ypre2[i + j] = 1
                ypre2[i - j] = 1






    else:
        #ypre2.append(0)
        continue
num=f1_score(ypre2, df3[96:-39],average='macro')
num1=recall_score(ypre2, df3[96:-39],average='macro')
num2=precision_score(ypre2, df3[96:-39],average='macro')
precisions, recalls, thresholds = precision_recall_curve(df3[119:-16],st )
print(num)
print(num1)
print(num2)
#(thresholds)
f1_scores = (2 * precisions * recalls) / (precisions + recalls)
th = thresholds[np.argmax(f1_scores[np.isfinite(f1_scores)])]
#print('Best threshold: ', th)
#print(f'best F1-score: {np.max(f1_scores[np.isfinite(f1_scores)])}')


#plt.grid(None)
#plt.plot(outlier_x1, outlier1, 'ro')
plt.plot(x,st1,'-')

#plt.tick_params(bottom=False,top=False,left=False,right=False)
plt.legend()
plt.grid(False)
plt.xticks([])
plt.yticks([])
#plt.plot(x,st3,'-',color='g',label="anomaly score3")
plt.savefig('ano.jpg',
                dpi=400, bbox_inches='tight')
plt.clf()
plt.plot(x,st2,'-')
#plt.show()

#plt.tick_params(bottom=False,top=False,left=False,right=False)
plt.legend()
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.savefig('ano2.jpg',
                dpi=400, bbox_inches='tight')
plt.clf()
plt.plot(x,st3,'-')
#plt.show()

#plt.tick_params(bottom=False,top=False,left=False,right=False)
plt.legend()
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.savefig('ano38.jpg',
                dpi=400, bbox_inches='tight')

plt.clf()
plt.plot(x,st,'-')
#plt.show()
#plt.grid(None)
plt.plot(outlier_x1, outlier1,'ro',linewidth=0.2,markersize='3')
#plt.tick_params(bottom=False,top=False,left=False,right=False)
plt.legend()
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.savefig('anototal.jpg',
                dpi=400, bbox_inches='tight')
#print(dt)
dt.to_csv("spe.csv")
