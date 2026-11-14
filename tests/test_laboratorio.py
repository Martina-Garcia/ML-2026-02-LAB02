import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from src.datos.lfw import CargadorLFW
from src.embeddings.fisherfaces import ExtractorFisherfaces
from src.interfaz.servicio import ServicioReconocimiento


class TestLab(unittest.TestCase):
    def test_prediccion_svm(self):
        class Modelo:
            classes_ = np.array([0, 1])
            def predict(self, X): return np.array([0])
            def predict_proba(self, X): return np.array([[.4, .6]])
        servicio = ServicioReconocimiento.__new__(ServicioReconocimiento)
        servicio.conjunto = SimpleNamespace(nombres_identidades=["A", "B"])
        servicio.particion = SimpleNamespace(X_test=np.ones((1, 3)),
            images_test=np.ones((1, 1, 3)), y_test=np.array([0]))
        servicio.fabrica_clasificadores = SimpleNamespace(nota_probabilidad=lambda _: "Platt")
        servicio._obtener_bundle = lambda *args: {
            "extractor": SimpleNamespace(transform=lambda X: X), "clasificador": Modelo()}
        pred = servicio.predecir(0, "pca", "svm", 3)
        self.assertEqual(pred.identidad_predicha, 0)
        self.assertEqual(pred.probabilidad, .4)
        self.assertTrue(pred.acierto)

    def test_transform_fisherfaces(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 15))
        y = np.repeat(np.arange(3), 10)
        extractor = ExtractorFisherfaces(8, 2).fit(X, y)
        media = extractor.pca._modelo.mean_.copy()
        componentes = extractor.pca._modelo.components_.copy()
        Z = extractor.transform(rng.normal(100, 1, size=(7, 15)))
        self.assertEqual(Z.shape, (7, 2))
        np.testing.assert_array_equal(media, extractor.pca._modelo.mean_)
        np.testing.assert_array_equal(componentes, extractor.pca._modelo.components_)
        with self.assertRaises(ValueError):
            ExtractorFisherfaces(8, 3).fit(X, y)

    def test_etiquetas_lfw(self):
        y = np.repeat(np.arange(4), [21, 24, 22, 23])
        images = np.arange(len(y)*6, dtype=float).reshape(-1, 2, 3)
        bruto = SimpleNamespace(images=images, data=images.reshape(-1, 6),
                                target=y, target_names=np.array(["A", "B", "C", "D"]))
        cargador = CargadorLFW()
        cargador.max_identidades = 2
        with patch("src.datos.lfw.fetch_lfw_people", return_value=bruto) as fetch:
            c = cargador.cargar()
        self.assertEqual(c.ids_originales, [1, 3])
        self.assertEqual(c.nombres_identidades, ["B", "D"])
        np.testing.assert_array_equal(np.unique(c.y), [0, 1])
        np.testing.assert_array_equal(c.images.reshape(-1, 6), c.X)
        self.assertEqual(fetch.call_args.kwargs["min_faces_per_person"], 20)


if __name__ == "__main__":
    unittest.main()
