## 協調フィルタリングを用いた映画のレコメンデーションシステム
## 参考
##　Robert M. Bell, Yehuda Koren and Chris Volinsky, “Modeling Relationships at Multiple Scales to Improve Accuracy of Large Recommender System”


import numpy as np
import numpy.matlib
import pandas as pd
from pandas import DataFrame, Series
import csv
import sys

from IPython.core.debugger import Tracer; debug_here = Tracer()

## 読み込んだデータにコラム名を付けて csv 形式で保存
f = open('./MovieLens_100K/ml-100k/u.data.csv','w');

writer = csv.writer(f)
writer.writerow(('u_id','i_id','rate','time'))

for i in range(len(data)):

               tmp = data[i].split()
               tmp = list(map(int,tmp))

               writer.writerow((tmp[0],tmp[1],tmp[2],tmp[3]))
f.close()

## 作成したcsv形式データを読み込み
df = pd.read_csv('./MovieLens_100K/ml-100k/u.data.csv')

## use-item 行列を作成（各要素はユーザーのアイテムに対する評価）
r_num = max(df.u_id)
c_num = max(df.i_id)

ui_mat = np.zeros([r_num,c_num])

for i in range(r_num):
    u_t = i+1
    u_data = df[['i_id','rate']][df.u_id == u_t]

    ui_mat[i][u_data['i_id']-1] = u_data['rate']


## user-item 行列の非ゼロ要素の標準化
nonzero_ui_mat = np.nonzero(ui_mat) # 非ゼロ要素の位置を取得

nonzero_rownum = np.zeros([1,r_num])
nonzero_colnum = np.zeros([1,c_num])

for i in range(r_num):
    nonzero_rownum[0,i] = np.count_nonzero(nonzero_ui_mat[0] == i) #　各行の非ゼロ要素数

for i in range(c_num):
    nonzero_colnum[0,i] = np.count_nonzero(nonzero_ui_mat[1] == i) # 各列の非ゼロ要素数


## user毎の評価の標準化 （非ゼロ要素の間で平均，分散を計算し標準化）

ui_matT = np.transpose(ui_mat)## np.transpose は view を返すことに注意 → viewを代入した先での変更も反映される
u_mean = sum(ui_matT)/nonzero_rownum

u_var = np.zeros([1,r_num])
for i in range(r_num):

    ## 分散を計算
    temp = ui_matT
    dev =  temp[nonzero_ui_mat[1][nonzero_ui_mat[0]==i], i] - u_mean[0,i]
    u_var[0,i] = sum(dev*dev) / len(nonzero_ui_mat[1][nonzero_ui_mat[0]==i])

    ## 標準化
    temp[nonzero_ui_mat[1][nonzero_ui_mat[0]==i], i] = dev / u_var[0,i]

ui_mat = np.transpose(temp)


## item-item の類似度を過去のユーザーの評価履歴から計算 （共通のユーザーが評価しているアイテム間の評価の差の二乗平均の逆数に基づく）
## 一つ一つ関数を用いると遅いので行列演算で計算量削減
## Shrinkage のために 類似度計算に用いる要素数を取得する必要がある　→ 内積計算方法を変更　→

invsim_mat = np.zeros([c_num,c_num])
nonzero_list = [0 for i in range(c_num)]# リスト配列　あるアイテム i と他を評価しているユーザーを特定するために　非ゼロな要素位置を保存

for i in range(c_num):

    i_vec = ui_mat[:,i][:,np.newaxis]

    temp = np.matlib.repmat(i_vec,1,c_num) # 列ベクトルを複製→行列化
    temp = temp * ui_mat

    ## temp 行列内の非ゼロ要素数を列ごとにカウント　→ アイテム i と j を同時に評価しているユーザー特定

    ii_nonzero = np.nonzero(temp) # 非ゼロ要素の位置を取得
    nonzero_list[i] = ii_nonzero

    alpha = 50
    for j in range(c_num):

        if j >= i : break

        nonzero_index = ii_nonzero[0][ii_nonzero[1]==j]##アイテム i と j 両方評価しているユーザーを取得

        sim = len(nonzero_index) /( sum( ( ui_mat[nonzero_index,i] - ui_mat[nonzero_index,j] )**2) + alpha)
        invsim_mat[i,j] = sim

    print(i)

temp = np.transpose(invsim_mat)
invsim_mat = invsim_mat + temp

for i in range(c_num):
    invsim_mat[i,i] = -1


##  u_rate_est （ユーザー u の アイテム i　への評価の推定値）を求めるために i に対する類似度上位 k 個のアイテムを求める


u_rate_est = np.zeros([r_num,c_num])

for u in range(r_num):

    print(u)

    #　ユーザー u における あるアイテム i に対する 類似度での近傍アイテム u_kNN を　求める　論文でのN(i;u)
    ind = nonzero_ui_mat[0] == u
    u_item = nonzero_ui_mat[1][ind] #ユーザー u が評価しているアイテム

    temp = np.argsort(invsim_mat[:,u_item]) # 各行の要素を降順にソートした際の元の列番号を返す

    N_size = 20 # 近傍には類似度上位２０のアイテムを含める
    top_N = temp[:,np.arange(-1,N_size*(-1)-1,-1)] # 各アイテムについての 類似度上位 N_size 個
    u_kNN = u_item[top_N]

    # アイテムi と共通の評価している人がいるアイテムの内で u_kNN に含まれるものを求める

    for i in range(c_num):
        ind1 = nonzero_list[i][1] != i

        i_list = np.unique(nonzero_list[i][1][ind1]) # i番目のアイテムを除く,アイテムi と共通の評価者を有するアイテムのリスト

        kNN_in_i_list_bool = np.in1d(i_list, u_kNN[i,:], assume_unique=True )

        i_list_kNN = i_list[kNN_in_i_list_bool] # アイテムi と共通の評価している人がいるアイテム集合 i_list 内で u_kNN に含まれるものを求める

        ind3 = [0 for m in range(N_size)]

        # 求めた近傍 から 評価値の推定値を算出

        A = np.zeros([N_size, N_size])
        B = np.zeros([N_size, N_size])

        for j in range(N_size):

            if j >=  len(i_list_kNN): ##データが十分に存在しなく， アイテムi と u_kNN のアイテムを同時に評価した人がそもそもいない場合がある　→ Shrink
                break

            ind2 = nonzero_list[i][1] == i_list_kNN[j] ## i_list_kNNに含まれるアイテムを評価しているユーザーを抽出　nonzero_list[i] はアイテム i との内積が 0でなかった位置を格納
            ind3[j] = nonzero_list[i][0][ind2] ## 各アイテム j を評価しているユーザーを格納

        for j in range(N_size):

            for k in range(N_size):

                if k>j:
                    break

                if j >= len(i_list_kNN) or k >= len(i_list_kNN):
                    break

                com_ind = ind3[j][np.in1d(ind3[j], ind3[k], assume_unique = True)] ## アイテム　j , k　間で共通するユーザーを抽出

                A[j][k] = sum((ui_mat[com_ind ,i_list_kNN[j]] - ui_mat[com_ind, i] ) *  (ui_mat[com_ind,i_list_kNN[k]] - ui_mat[com_ind, i] ))

                B[j][k] = len(com_ind)



        A = A + np.transpose(A) - A*np.identity(N_size)
        B = B + np.transpose(B) - B*np.identity(N_size)

        A_prime = A/B
        A_prime[np.isnan(A_prime)] = 0 # supportがない(Bjk=0)場合は /0 で nan になった値を０で置き換える

        ave_diag       = np.mean(np.diag(A_prime))
        ave_notdiag =  np.sum(A_prime*(np.ones([N_size,N_size]) - np.identity(N_size))) / (N_size**2 - N_size)

        gamma = 20

        A_hat_diag       = (B*A + gamma*ave_diag) / (B+gamma)
        A_hat_notdiag = (B*A + gamma*ave_notdiag) / (B+gamma)

        A_hat = A_hat_notdiag - (A_hat_notdiag*np.identity(N_size)) + A_hat_diag*np.identity(N_size)

        const = 0.001 # アイテム間での評価に与える影響度の総和を 1 に制約する条件の強さを定める値,　二次計画問題を解くにあたって行列の半正定値性に影響を与えてしまうので小さく設定

        C = A_hat - const
        b = np.ones([N_size,1])*const

        w = revNNQO(C,b) ## 二次計画問題を解き，アイテム間での評価に与える影響度を算出

　　　　 # 求めたアイテム間の影響度 w を用いて， ユーザー u の既知アイテム(N(i;u))の評価値 から 未知のアイテムの評価値を算出
        u_rate_est[u,i] = np.dot(np.transpose(w),ui_mat[u,u_kNN[i,:]]) / sum(w)






## 二次計画問題を解くためのアルゴリズム　論文参照

def revNNQO(H, b):

    x = np.random.rand(len(b))[:,np.newaxis] #求めたいベクトルを乱数で初期化
    eps = 0.001
    counter = 0
    fail_counter = 0

    while True:

        counter = counter + 1;
        r = np.dot(H, x) - b

        for i in range(len(b)):
            if x[i] == 0 and r[i] > 0:
                r[i] = 0;

        alpha = np.dot(np.transpose(r), r) / np.dot(np.dot(np.transpose(r),H),r) # x の要素に負の値が出ないように更新幅を設定

        for i in range(len(b)):
            if r[i] > 0:
                alpha = min(alpha, x[i]/r[i])

        x = x - alpha*r

        nancheck = np.isnan(x)
        negcheck = x<0

        if fail_counter >20: # 20回以上算出に失敗（収束しない，負の値,nan を出す）した場合は 0ベクトルを返す　→ 対応する評価の推定値は nan になり判別可能
            x = np.zeros([len(b),1])
            return x

        if nancheck.any() or negcheck.any():

            fail_counter = fail_counter +1

            x = np.random.rand(len(b))[:,np.newaxis]
            counter = 0
            continue

        if np.linalg.norm(r) < eps: # 勾配の値が十分小さくなったらこれ以上 x は更新されないので x を返す
            break

        if counter >50: # 50回以上イテレーションしても勾配が収束しない場合には x の再度乱数で初期化し計算を試みる

            fail_counter = fail_counter+1
            x = np.random.rand(len(b))[:,np.newaxis]
            counter = 0

    return x
