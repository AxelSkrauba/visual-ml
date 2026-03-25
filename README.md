# 🧠 Visual ML

> Explorador interactivo de **clasificación** y **regresión** para modelos clásicos de Machine Learning.  
> Diseñado para la enseñanza de IA en cursos de ingeniería.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelSkrauba/visual-ml/blob/main/notebooks/visual_ml_colab.ipynb)
[![CI](https://github.com/AxelSkrauba/visual-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/visual-ml/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live-Demo-FF4B4B?logo=streamlit&logoColor=white)](https://visual-ml.streamlit.app/)

---

## Probar online

La aplicación está disponible públicamente en Streamlit Cloud:

**Demo en vivo:** https://visual-ml.streamlit.app/

No requiere instalación ni registro.
Para ejecutarla localmente o en Google Colab, revisar la sección [Instalación y ejecución local](#instalación-y-ejecución-local).

## ¿Qué es Visual ML?

Visual ML es una herramienta pedagógica interactiva que permite explorar intuitivamente cómo los modelos clásicos de ML separan clases (clasificación) o ajustan curvas (regresión) en un espacio de dos features. Visualiza en tiempo real cómo cambia la **frontera de decisión** o la **curva de predicción** al modificar hiperparámetros, tipo de dataset y nivel de ruido.

### Conceptos que se pueden explorar
- **Overfitting y underfitting** — observa la brecha train/test en métricas y frontera/curva.
- **Complejidad del modelo** — desde fronteras lineales hasta espirales complejas; desde rectas hasta curvas no lineales.
- **Impacto de hiperparámetros** — C en SVM/SVR, max_depth en árboles, K en KNN, capas en MLP, alpha en Ridge/Lasso, etc.
- **Comparación de algoritmos** — hasta 6 modelos lado a lado sobre el mismo dataset.
- **Desbalance de clases** — efecto en accuracy vs F1 con dataset configurable.
- **Análisis de residuos** — detecta heterocedasticidad y patrones en errores de regresión.
- **Métricas de regresión** — R², RMSE, MAE con señales pedagógicas automáticas.

---

## Modelos incluidos

### Clasificación (11 modelos)

| Grupo | Modelos |
|---|---|
| **Lineales** | Logistic Regression, Perceptron |
| **Red Neuronal** | MLP (Multi-Layer Perceptron) |
| **Kernel / Margen** | SVC (lineal, RBF, poly, sigmoid) |
| **Vecindad** | K-Nearest Neighbors |
| **Árbol** | Decision Tree |
| **Ensemble Bagging** | Random Forest |
| **Ensemble Boosting** | Gradient Boosting, AdaBoost |
| **Probabilístico** | Gaussian Naive Bayes, QDA |

### Regresión (8 modelos)

| Grupo | Modelos |
|---|---|
| **Lineales** | Linear Regression, Ridge, Lasso |
| **Kernel / Margen** | SVR (lineal, RBF, poly) |
| **Vecindad** | KNN Regressor |
| **Árbol** | Decision Tree Regressor |
| **Ensemble** | Random Forest Regressor |
| **Red Neuronal** | MLP Regressor |

## Datasets de juguete

### Clasificación (7 datasets)

| Dataset | Descripción | Clases |
|---|---|---|
| **Moons** | Dos lunas entrelazadas | 2 |
| **Circles** | Círculos concéntricos | 2 |
| **Blobs** | Clusters gaussianos | 2–5 |
| **XOR** | Patrón XOR en cuadrantes | 2 |
| **Linear** | Linealmente separable | 2–5 |
| **Spirals** | Espirales entrelazadas | 2–5 |
| **Desbalanceado** | Clases gaussianas con ratio de minoría configurable (5–50%) | 2 |

### Regresión (5 datasets)

| Dataset | Descripción |
|---|---|
| **Lineal** | Relación lineal simple con ruido gaussiano |
| **Polinomial** | Curva polinómica de grado configurable (1–6) |
| **Sinusoidal** | Onda sinusoidal con frecuencia ajustable |
| **Escalón** | Función escalonada con saltos abruptos |
| **Exponencial** | Crecimiento exponencial con ruido |

---

## Instalación y ejecución local

### Requisitos
- Python 3.10+
- pip

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/AxelSkrauba/visual-ml.git
cd visual-ml

# 2. Crear y activar entorno virtual (recomendado)
python -m venv venv

# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar la aplicación
streamlit run app.py
```

La aplicación se abre automáticamente en `http://localhost:8501`.

> **Recomendación:** Siempre usar un entorno virtual dedicado para aislar las dependencias del proyecto y evitar conflictos con otros paquetes del sistema.

### Sobre el flag `--server.headless` y el pedido de email de Streamlit

Al ejecutar `streamlit run app.py` por primera vez, Streamlit puede pedir un correo electrónico de forma interactiva. Esto solo ocurre una vez y se puede omitir presionando **Enter**.

Para evitar este prompt (útil en entornos CI/CD o ejecución automatizada), el proyecto incluye un archivo `.streamlit/config.toml` con `headless = true` que lo suprime automáticamente. Si se ejecuta la app desde la línea de comandos y el prompt aparece de todas formas, se puede agregar el flag manualmente:

```bash
streamlit run app.py --server.headless true
```

O simplemente presionar Enter para omitir el registro — no afecta el funcionamiento de la aplicación.

---

## Uso en Google Colab

Clic en el badge para abrir directamente en Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelSkrauba/visual-ml/blob/main/notebooks/visual_ml_colab.ipynb)

El notebook usa el port forwarding **nativo de Google Colab** (`serve_kernel_port_as_iframe`) — no requiere cuenta externa, token ni ngrok. La app aparece embebida directamente en el notebook.

---

## Ejecutar los tests

```bash
# Con entorno virtual activado:
python -m pytest tests/ -v --tb=short
```

La suite de tests cubre todos los módulos core con TDD (270+ tests):

**Clasificación:**
- `test_datasets.py` — generación, shapes, reproducibilidad, multi-clase, dataset desbalanceado.
- `test_models.py` — registro, metadatos, defaults, instanciación, MLP string parsing.
- `test_train.py` — métricas, rangos válidos, señales pedagógicas (7 escenarios).
- `test_visualization.py` — figuras correctas para todos los modelos/temas.

**Regresión:**
- `test_datasets_regression.py` — 5 generadores, shapes, reproducibilidad, noise, parámetros.
- `test_models_regression.py` — 8 modelos, instanciación, hyperparams, grupos.
- `test_train_regression.py` — R², RMSE, MAE, señales pedagógicas.
- `test_visualization_regression.py` — prediction curve, residuos, prediction error, comparación.

---

## Integración continua (CI)

Cada push y pull request a `main` ejecuta automáticamente la suite de tests en Python 3.10 y 3.11 via GitHub Actions. El badge de estado arriba muestra el resultado de la última ejecución.

---

## Estructura del proyecto

```
visual-ml/
├── app.py                      # Aplicación principal Streamlit
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml             # Configuración de tema base y servidor
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
├── core/
│   ├── datasets.py             # Generadores de datasets de clasificación (7 tipos)
│   ├── datasets_regression.py  # Generadores de datasets de regresión (5 tipos)
│   ├── models.py               # Registro de modelos de clasificación (11 modelos)
│   ├── models_regression.py    # Registro de modelos de regresión (8 modelos)
│   ├── train.py                # Entrenamiento + evaluación (clasificación)
│   ├── train_regression.py     # Entrenamiento + evaluación (regresión)
│   ├── visualization.py        # Frontera de decisión, CM, comparación
│   └── visualization_regression.py  # Curva predicción, residuos, error
├── ui/
│   ├── sidebar.py              # Controles interactivos (Clasificación/Regresión)
│   ├── single_view.py          # Tab: exploración clasificación
│   ├── single_view_regression.py  # Tab: exploración regresión
│   ├── compare_view.py         # Tab: comparación clasificación
│   └── compare_view_regression.py  # Tab: comparación regresión
├── notebooks/
│   └── visual_ml_colab.ipynb   # Notebook para Google Colab
└── tests/
    ├── conftest.py
    ├── test_datasets.py
    ├── test_datasets_regression.py
    ├── test_models.py
    ├── test_models_regression.py
    ├── test_train.py
    ├── test_train_regression.py
    ├── test_visualization.py
    └── test_visualization_regression.py
```

---

## Interfaz

### Clasificación
- **Explorar:** frontera de decisión, métricas (accuracy, precision, recall, F1), matriz de confusión, classification report.
- **Comparar:** grid de fronteras de decisión, barras comparativas, tabla resumen.
- Señales pedagógicas automáticas: overfitting, underfitting, desbalance, memorización.

### Regresión
- **Explorar:** curva de predicción, métricas (R², RMSE, MAE), gráfico de error de predicción, residuos train/test.
- **Comparar:** grid de curvas de predicción, barras comparativas, tabla resumen.
- Señales pedagógicas: overfitting, underfitting, heterocedasticidad, R² negativo.

### Sidebar
- Selector de paradigma (Clasificación / Regresión) al inicio.
- Datasets y modelos cambian dinámicamente según el paradigma seleccionado.
- Hiperparámetros con tooltips pedagógicos en español.

### Temas
- **Oscuro** (por defecto): ideal para pantallas.
- **Claro**: recomendado para proyección en aula.

---

## Autor

**Axel Skrauba** · [@AxelSkrauba](https://github.com/AxelSkrauba)

---

## Licencia

MIT — libre para uso académico y educativo.
