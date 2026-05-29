"""最終スタッキング: 各モデルのOOF予測 + embedding(SVD圧縮) + 画像メタ → LightGBM。"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import TruncatedSVD


def svd_embed(feat_tr, feat_te, k=16, seed=42):
    k = min(k, feat_tr.shape[1] - 1)
    s = TruncatedSVD(n_components=k, random_state=seed)
    return s.fit_transform(feat_tr), s.transform(feat_te)


def stack_lgb(X_tr, y, X_te, groups, n_folds=5, seed=42):
    """SGKF(group=art_series) で LGB スタック。OOF と test予測(clip[0,3])を返す。"""
    oof = np.zeros(len(y))
    pred = np.zeros(len(X_te))
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    ystr = np.clip(np.round(y).astype(int), 0, 3)
    for tr, va in sgkf.split(X_tr, ystr, groups):
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                              max_depth=3, subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, verbose=-1)
        m.fit(X_tr[tr], y[tr])
        oof[va] = m.predict(X_tr[va])
        pred += m.predict(X_te) / n_folds
    return np.clip(oof, 0, 3), np.clip(pred, 0, 3)
