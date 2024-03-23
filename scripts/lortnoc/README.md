# Lortnoc
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97HuggingFace-Model-blue)](https://huggingface.co/furusu/lortnoc)

LoRTnoC(LoRA with hint block of ControlNet)をComfyUI上で使うためのリポジトリです。

モデルファイルはcontrolnetと同じ場所に入れてください（適当すぎ？）



# 例
画像にワークフローがついています(多分)。

|     |  reference   | generated  | 
| --- | --- | --- | 
|    canny |  ![canny](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/454f8207-1113-4ec5-8e3f-8ff03844f795)|  ![canny_generated](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/0713691e-a72a-40b3-b941-9fb472634761)|
|depth|![depth](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/96d9600d-3fa9-4402-8cd5-462e2b85a80f)|![depth_generated](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/d4417d8d-9158-44b0-9f95-07e9271a52ef)|
|  hed   |  ![hed](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/c2b1a946-edea-43d9-bfcf-ac6a55eb87ed)|   ![hed_generated](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/e01f7b37-e71e-4d0a-b6fb-27eadbf36a16)|
|fake_scribble|![fake_scribble](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/693cafe2-acda-463a-af5b-c3a399848a68)|![fake_scribble_generated](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/126d6fff-2ff3-4458-85ec-5e2263768276)|
|lineart_anime|![lineart_anime](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/78aa5570-16d0-4c28-b7f9-3167c1667836)|![lineart_anime_generated](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/8928cb7f-f269-4335-8e8a-661a26e4658b)|
|pose(cherry pick)|![pose](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/4b9de05c-7b28-4c59-8b83-ad7189fe51f0)|![pose_generated](https://github.com/laksjdjf/LoRTnoC-ComfyUI/assets/22386664/298c2acf-82ec-4b4c-b45e-30fa32659d53)|
