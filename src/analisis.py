from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import DIR_FIGURAS, DIR_RESULTADOS, RAIZ, ruta_modelo
from src.datos.division import DivisorEstratificado
from src.datos.lfw import CargadorLFW
from src.datos.olivetti import CargadorOlivetti


class AnalisisLaboratorio:

    def __init__(self):
        self.tabla = pd.read_csv(DIR_RESULTADOS / "resultados_modelos.csv")
        self.resumen = {"protocolo": {"semilla": 42, "test_size": 0.25,
                         "d": [10, 20, 30], "fisherfaces_d_pca": "igual a d_lda"},
                        "versiones": {"python": platform.python_version()},
                        "equipo": {"plataforma": platform.platform(),
                                   "procesador": platform.processor()}, "datasets": {}}
        for paquete in ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib",
                        "umap-learn", "streamlit", "nbclient"]:
            self.resumen["versiones"][paquete] = importlib.metadata.version(paquete)
        self.confusiones = []

    def generar(self):
        assert len(self.tabla) == 54, "Se esperan 27 combinaciones por dataset."
        assert not self.tabla.duplicated(["dataset", "embedding", "d", "modelo"]).any()
        for cargador in [CargadorOlivetti(), CargadorLFW()]:
            conjunto = cargador.cargar()
            p = DivisorEstratificado().dividir(conjunto)
            nombre = conjunto.nombre
            assert not np.intersect1d(p.indices_train, p.indices_test).size
            assert np.array_equal(np.unique(p.y_train), np.unique(p.y_test))
            info = {"N": len(conjunto.y), "C": conjunto.n_clases,
                    "H": conjunto.alto, "W": conjunto.ancho,
                    "p": conjunto.X.shape[1], "train": len(p.y_train),
                    "test": len(p.y_test), "intensidad_min": float(conjunto.X.min()),
                    "intensidad_max": float(conjunto.X.max()),
                    "fotos_min": int(np.bincount(conjunto.y).min()),
                    "fotos_max": int(np.bincount(conjunto.y).max())}
            np.savez(DIR_RESULTADOS / f"indices_{nombre}.npz",
                     train=p.indices_train, test=p.indices_test)
            pd.DataFrame({"etiqueta": np.arange(conjunto.n_clases),
                          "id_original": conjunto.ids_originales or list(range(conjunto.n_clases)),
                          "nombre": conjunto.nombres_identidades,
                          "fotos": np.bincount(conjunto.y)}).to_csv(
                              DIR_RESULTADOS / f"identidades_{nombre}.csv", index=False)
            predicciones = {}
            filas = self.tabla[self.tabla.dataset == nombre]
            for fila in filas.itertuples():
                bundle = joblib.load(ruta_modelo(nombre, fila.embedding, fila.modelo, fila.d))
                Z = bundle["extractor"].transform(p.X_test)
                assert Z.shape == (len(p.y_test), fila.d) and np.isfinite(Z).all()
                pred = bundle["clasificador"].predict(Z)
                assert np.isclose(accuracy_score(p.y_test, pred), fila.accuracy, atol=1e-6)
                assert np.isclose(f1_score(p.y_test, pred, average="macro", zero_division=0),
                                  fila.macro_f1, atol=1e-6)
                predicciones[(fila.embedding, fila.modelo, fila.d)] = pred

            detalle = pd.DataFrame({"indice_original": p.indices_test, "real": p.y_test})
            for (emb, mod, d), pred in predicciones.items():
                detalle[f"{emb}_{mod}_d{d}"] = pred
            detalle.to_csv(DIR_RESULTADOS / f"predicciones_{nombre}.csv", index=False)

            pca = PCA(n_components=min(120, len(p.y_train)-1, p.X_train.shape[1]),
                      svd_solver="full", random_state=42).fit(p.X_train)
            info["varianza"] = {str(d): float(np.sum(pca.explained_variance_ratio_[:d]))
                                for d in [10, 20, 30]}
            pd.DataFrame({"d": np.arange(1, pca.n_components_+1),
                          "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_)}).to_csv(
                              DIR_RESULTADOS / f"varianza_pca_{nombre}.csv", index=False)
            info["maximos_descriptivos"] = filas.loc[
                filas.groupby("embedding").macro_f1.idxmax()].to_dict("records")
            info["pca_svm_por_dimension"] = filas[
                (filas.embedding == "pca") & (filas.modelo == "svm")].to_dict("records")
            self._mapas(nombre)
            self._confusiones(conjunto, p, predicciones, info)
            self._errores(conjunto, p, predicciones, info)
            self.resumen["datasets"][nombre] = info
        pd.DataFrame(self.confusiones).to_csv(DIR_RESULTADOS / "confusiones_pares.csv", index=False)
        (DIR_RESULTADOS / "resumen.json").write_text(
            json.dumps(self.resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.resumen

    def _mapas(self, nombre):
        fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), layout="constrained")
        for row, metodo in enumerate(["tsne", "umap"]):
            for col, emb in enumerate(["pca", "lda", "fisherfaces"]):
                ax = axes[row, col]
                datos = np.load(DIR_FIGURAS / f"coordenadas_{metodo}_{nombre}_{emb}_d20.npz")
                xy, y = datos["coordenadas"], datos["identidades"]
                puntos = ax.scatter(xy[:, 0], xy[:, 1], c=y, cmap="hsv", s=6, alpha=.8,
                                    vmin=-.5, vmax=int(y.max())+.5)
                ax.set_title(f"{metodo.upper()} / {emb.upper()}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.colorbar(puntos, ax=axes, shrink=.65, label="Identidad (etiqueta nominal)")
        fig.savefig(DIR_FIGURAS / f"mapas_comparados_{nombre}.png", dpi=180)
        plt.close(fig)

    def _confusiones(self, c, p, preds, info):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
        matrices = [confusion_matrix(p.y_test, preds[("pca", mod, 20)],
                                    labels=np.arange(c.n_clases)) for mod in ["svm", "knn", "arbol"]]
        vmax = max(m.max() for m in matrices)
        pares = {}
        for ax, mod, matriz in zip(axes, ["svm", "knn", "arbol"], matrices):
            im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=vmax)
            ax.set(title=f"PCA(20) + {mod}", xlabel="Predicha", ylabel="Real")
            err = matriz.copy()
            np.fill_diagonal(err, 0)
            pares[mod] = {(int(i), int(j)) for i, j in zip(*np.where(err > 0))}
        fig.colorbar(im, ax=axes, shrink=0.8, label="Fotos")
        fig.savefig(DIR_FIGURAS / f"confusiones_comparadas_{c.nombre}.png", dpi=180)
        plt.close(fig)
        info["pares_comunes_tres_clasificadores_pca20"] = sorted(
            [list(t) for t in set.intersection(*pares.values())])
        for (emb, mod, d), pred in preds.items():
            if d != 20:
                continue
            pares_df = pd.DataFrame({"real": p.y_test, "predicha": pred})
            pares_df = pares_df[pares_df.real != pares_df.predicha]
            for (real, estimada), n in pares_df.groupby(["real", "predicha"]).size().items():
                self.confusiones.append({"dataset": c.nombre, "embedding": emb, "d": d,
                    "modelo": mod, "real": int(real), "predicha": int(estimada), "cantidad": int(n)})

    def _errores(self, c, p, preds, info):
        fig, axes = plt.subplots(3, 2, figsize=(8, 7.8), layout="constrained")
        ejemplos = []
        for fila, mod in enumerate(["svm", "knn", "arbol"]):
            pred = preds[("pca", mod, 20)]
            errores = np.flatnonzero(pred != p.y_test)
            if not len(errores):
                for ax in axes[fila]:
                    ax.axis("off")
                axes[fila, 0].text(.1, .5, f"{mod}: sin errores en TEST")
                continue
            i = int(errores[0])
            candidatos = np.flatnonzero(p.y_train == pred[i])
            j = int(candidatos[np.argmin(np.sum((p.X_train[candidatos]-p.X_test[i])**2, axis=1))])
            for ax, img in zip(axes[fila], [p.images_test[i], p.images_train[j]]):
                ax.imshow(img, cmap="gray", vmin=c.X.min(), vmax=c.X.max())
                ax.axis("off")
            axes[fila, 0].set_title(f"{mod}: TEST #{p.indices_test[i]}\nreal {p.y_test[i]} / predicha {pred[i]}", fontsize=10)
            axes[fila, 1].set_title(f"TRAIN de identidad {pred[i]}\nreferencia más cercana en píxeles", fontsize=10)
            ejemplos.append({"modelo": mod, "test_posicion": i,
                             "test_indice_original": int(p.indices_test[i]),
                             "real": int(p.y_test[i]), "predicha": int(pred[i]),
                             "nombre_real": c.nombres_identidades[p.y_test[i]],
                             "nombre_predicho": c.nombres_identidades[pred[i]]})
        fig.savefig(DIR_FIGURAS / f"errores_{c.nombre}.png", dpi=180)
        plt.close(fig)
        info["ejemplos_errores_pca20"] = ejemplos
