"""Carga de LFW. La selección de identidades la completa el alumno."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_lfw_people


from src.config import DIR_CACHE
from src.datos.base import CargadorRostros, validar_conjunto
from src.modelos import ConjuntoRostros


class CargadorLFW(CargadorRostros):
    """Fotografías de LFW con mayor diversidad de pose, luz y fondo.

    1. ``fetch_lfw_people(min_faces_per_person=20, resize=0.5, data_home=...)``.
    2. Se conservan como máximo 50 identidades, las de mayor frecuencia.
    3. ``y`` se reindexa con ``np.unique(..., return_inverse=True)``
    4. alto y ancho se leen desde ``images.shape``.
    5. Se llama a ``validar_conjunto`` antes de retornar.
    """

    nombre = "lfw"
    max_identidades = 50
    min_fotos = 20

    def cargar(self) -> ConjuntoRostros:
        lfw = fetch_lfw_people(
            min_faces_per_person=self.min_fotos,
            resize=0.5,
            data_home=str(DIR_CACHE),
            download_if_missing=True,
        )
        conteo = np.bincount(lfw.target, minlength=len(lfw.target_names))
        top = np.argsort(-conteo, kind="stable")[: self.max_identidades]
        mascara = np.isin(lfw.target,top)

        images = np.asarray(lfw.images[mascara])
        X= np.asarray(lfw.data[mascara],dtype=float)
        ids, y = np.unique(lfw.target[mascara], return_inverse=True)
        nombres = [str(lfw.target_names[i]) for i in ids]
        alto, ancho = int(images.shape[1]), int(images.shape[2])

        conjunto = ConjuntoRostros(
            nombre= self.nombre,
            images=images,
            X=X,
            y=y.astype(int),
            nombres_identidades=nombres,
            alto=alto,
            ancho=ancho,
            ids_originales=[int(i) for i in ids],
        )
        validar_conjunto(conjunto)
        print(
            f"LFW: N={X.shape[0]} C={conjunto.n_clases} "
            f"H={alto} W={ancho} fotos/identidad "
            f"min={int(np.bincount(conjunto.y).min())} max={int(np.bincount(conjunto.y).max())}"
        )
        return conjunto
