# LoRA Merger

このリポジトリはComfyUI上でLoRAのマージを実装したものです。
![image](https://github.com/laksjdjf/cgem156-ComfyUI/assets/22386664/b183a2ab-077f-4e8a-9592-85acb76cc0b5)


## ノード
全てのノードはlora_mergeに入っています。

1. `Load LoRA Weight Only`: LoRAを単体で読み込みます。
2. `LoRA LoRA from Weight`: LoRAをモデルに適用します。
3. `Merge LoRA`: 二つのLoRAをマージします。
4. `Save LoRA`: LoRAを`ComfyUI/models/loras`にセーブします。**※設定したstrengthを適用したLoRAが保存されます。**

## マージについて
4つ設定があります。`dtype`はLoRAの型になります。ファイルサイズを軽くしたいときは`float16`や`bfloat16`にしてください。
`rank`, `device`は`mode`を`svd`にしたときのみ参照されます。

`mode`の説明:
1. `add`:加算によってマージします。二つのLoRAのrankがイコールではない場合、エラーが起きます。また二つのLoRAが全く違うものである場合精度は低くなります。
2. `concat`:結合によってマージします。rankは二つのLoRAの合計になってしまいますが、正確なマージになります。
3. `svd`:特異値分解を利用してマージします。rankを設定することで、自由にマージ先のrankを変えることができます。ただし時間がかかるほか、精度も少し落ちます。`device`でGPUで計算するかCPUで計算するか選べます。

## ブロック別重みについて
`LoRA LoRA from Weight`のlbw欄に、[sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight)に沿った文字列を記入するといい感じになります。プリセットはpreset.txtで追加できます。RとかUには未対応です。

## 注意点
+ メタデータのことはよく分からないので何も考慮してません。
+ LyCORISのLoHAやLokrには未対応です。というかこれらはマージできるのか？
+ 異なる学習コードで作成されたLoRA同士のマージはうまくいかない可能性があります。

## 数学的な話
二つのLoRAのup層、down層、alpha、rank、strengthをそれぞれ、 $A_i, B_i, \alpha_i, r_i, w_i (i = 1,2)$ とします。

すると、欲しいLoRAは、以下の通りです。

$\displaystyle{\frac{\alpha}{r}}AB =  w_1\displaystyle{\frac{\alpha_1}{r_1}}A_1B_1 + w_2\displaystyle{\frac{\alpha_2}{r_2}}A_2B_2 $

1.どちらか一方のLoRAにしかないモジュールについて

片方しかない場合、ある方 $i$ のLoRAを使って、

$A = \sqrt{w_i}A_i,\ \ B =  \sqrt{w_i}B_i,\ \ \alpha = \alpha_i$

とします。行列積をとることを考えて、 $w_i$ の二乗根をとります。どちらか一方にだけそのままかけるという方法でもいいと思いますが、計算精度の問題でどちらがいいのかわからないのでとりあえずこっちにしてます。

2. 両方ある場合

モードによって異なります。

`add`:
down層、up層をそれぞれ重み付き加算することによってマージします。

 $A = w_1A_1 + \displaystyle{\sqrt{\frac{\alpha_2}{\alpha _1}}}w_2A_2,\ \ B = w_1B_1 + \displaystyle{\sqrt{\frac{\alpha_2}{\alpha _1}}}w_2B_2,\ \ \alpha = \alpha_1,\ \ r=r_1=r_2$

 とします。

$\displaystyle{\frac{\alpha}{r}}AB = {w_1}^2\displaystyle{\frac{\alpha_1}{r}}A_1B_1  + w_1w_2\displaystyle{\frac{\sqrt{\alpha_1\alpha_2}}{r}}A_1B_2 + w_1w_2\displaystyle{\frac{\sqrt{\alpha_1\alpha_2}}{r}}A_2B_1 + {w_2}^2\displaystyle{\frac{\alpha_2}{r}}A_2B_2$ 

 となります。正直意味わからないんですが、同じLoRAを $w_1+w_2=1$ でマージしたときに同じLoRAになるよう調整されています。

 `concat`:
down層、up層をそれぞれrank方向に結合することによってマージします。

$A = (\sqrt{w_1}\displaystyle{\sqrt{\frac{r_1+r_2}{r_1}}}A_1\ \ \displaystyle{\sqrt{w_2\frac{\alpha_2}{\alpha_1}}}\displaystyle{\sqrt{\frac{r_1+r_2}{r_2}}}A_2),\ \ B = (\sqrt{w_1}\displaystyle{\sqrt{\frac{r_1+r_2}{r_1}}}B_1\ \ \displaystyle{\sqrt{w_2\frac{\alpha_2}{\alpha_1}}}\displaystyle{\sqrt{\frac{r_1+r_2}{r_2}}}B_2)^T,\ \ \alpha=\alpha_1,\ \ r=r_1+r_2$

とします。

$\displaystyle{\frac{\alpha}{r}}AB =  w_1\displaystyle{\frac{\alpha_1}{r_1}}A_1B_1 + w_2\displaystyle{\frac{\alpha_2}{r_2}}A_2B_2 $
となり完全なマージになります。

$w < 0$ のときはそのままでは二乗根を計算できないので、絶対値をとって、 $A$ 側に後からマイナスをかけることにしています。

`svd`

 $W = \displaystyle{\frac{r}{r_1}}\sqrt{w_1}A_1\sqrt{w_1}B_1 + \displaystyle{\frac{r}{r_2}}\displaystyle{\sqrt{w_2\frac{\alpha_2}{\alpha_1}}}A_2\displaystyle{\sqrt{w_2\frac{\alpha_2}{\alpha_1}}}B_2$

 として $W$ を特異値分解による低ランク近似で二つの $r=$ `rank`の行列 $A,B$ に分解します。

 

