`SampleCustom`ノードで使えるオリジナルのサンプラーたちです。

## Gradual Latent Sampler
高解像度生成のためのサンプラーである[Gradual Latent](https://gist.github.com/kohya-ss/84e7404265910a4c08989ae47e0ab213)を実装するノード。

使い方はよくわからない。

![image](https://github.com/laksjdjf/cgem156-ComfyUI/assets/22386664/097c51e6-5a9e-4528-aacd-dff763cb179d)

## LCM Sampler RCFG
[StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)で紹介された、RCFGを実装するノードです。

original_latentはoptionalで、入力するとself-negative, 省略するとone-step-negativeになります。

![image](https://github.com/laksjdjf/cgem156-ComfyUI/assets/22386664/9db12b91-552d-4b22-96ac-5b5d635a015f)

## TCD Sampler

[TCD](https://github.com/jabir-zheng/TCD)用のサンプラーだが実装があっているか分からない。

![image](https://github.com/laksjdjf/cgem156-ComfyUI/assets/22386664/2bd62139-94c4-4f25-9289-4a6eaac40f0c)

