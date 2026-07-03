from ... import ROOT_NAME, SYMBOL, NODE_SURFIX
import torch
import numpy as np
import cv2
from comfy_api.v0_0_2 import io

# ref:https://qiita.com/fdsafdfadsa/items/4e8046998be9627ca85d
def kmeans_quant(img, K, kmeans_pp):

    flags = cv2.KMEANS_RANDOM_CENTERS if not kmeans_pp else cv2.KMEANS_PP_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1e-4)
    _, label, center = cv2.kmeans(img, K, None, criteria, 10, flags)
    res = center[label.flatten()]

    return res

class KMeansManhattan:
    def __init__(self, n_clusters, max_iters=10, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        # データセットのサイズ
        n_samples, n_features = X.shape

        # クラスタ中心をデータポイントの中からランダムに初期化
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        for i in range(self.max_iters):
            # 各データポイントを最も近いクラスタに割り当てる
            self.labels = self._assign_clusters(X)

            # 新しいクラスタ中心を計算 (マンハッタン距離のためには中央値を使用)
            new_centroids = np.array([np.median(X[self.labels == j], axis=0) for j in range(self.n_clusters)])

            # クラスタ中心の変化が許容範囲内であれば終了
            if np.all(np.abs(self.centroids - new_centroids).sum(axis=1) < self.tol):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        # 各データポイントとクラスタ中心とのマンハッタン距離を計算
        distances = np.sum(np.abs(X[:, np.newaxis] - self.centroids), axis=2)
        # 最も近いクラスタにラベルを割り当てる
        return np.argmin(distances, axis=1)

    def predict(self, X):
        # 新しいデータに対してクラスタを予測
        return self._assign_clusters(X)

def kmeans(img, K, kmeans_pp, manhattan, seed):
    orogin_state = np.random.get_state()
    np.random.seed(seed)

    if manhattan:
        kmeans = KMeansManhattan(n_clusters=K)
        kmeans.fit(img)
        retval =  kmeans.centroids[kmeans.predict(img)]
    else:
        retval = kmeans_quant(img, K, kmeans_pp)

    np.random.set_state(orogin_state)
    return retval

class KmeansQuantize(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"KmeansQuantize{NODE_SURFIX}",
            display_name=f"Kmeans Quantize {SYMBOL}",
            category=ROOT_NAME + "for_test",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("colors", default=256, min=1, max=256, step=1),
                io.Boolean.Input("individual"),
                io.Boolean.Input("kmeans_pp"),
                io.Boolean.Input("manhattan"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, colors: int, individual: bool, kmeans_pp:bool, manhattan: bool, seed: int) -> io.NodeOutput:
        batch_size, height, width, channels = image.shape
        image = image.reshape(batch_size, height * width, channels).float().cpu().numpy()

        if individual:
            result = np.zeros_like(image)
            for i in range(batch_size):
                result[i] = kmeans(image[i], colors, kmeans_pp, manhattan, seed)
        else:
            result = kmeans(image.reshape(-1, channels), colors, kmeans_pp, manhattan, seed).reshape(batch_size, height * width, channels)

        result = torch.from_numpy(result).float().reshape(batch_size, height, width, channels)
        return io.NodeOutput(result)
