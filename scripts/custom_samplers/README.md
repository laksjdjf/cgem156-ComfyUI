`SampleCustom`ノードで使えるオリジナルのサンプラーたちです。

## Gradual Latent Sampler
高解像度生成のためのサンプラーである[Gradual Latent](https://gist.github.com/kohya-ss/84e7404265910a4c08989ae47e0ab213)を実装するノード。

使い方はよくわからない。

## LCM Sampler RCFG
[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)で紹介された、RCFGを実装するノードです。

original_latentはoptionalで、入力するとself-negative, 省略するとone-step-negativeになります。


## TCD Sampler

[TCD](https://github.com/jabir-zheng/TCD)用のサンプラーだが実装があっているか分からない。


