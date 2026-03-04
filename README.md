# 🧠 Visual ML

> Explorador interactivo de fronteras de decisión para modelos clásicos de Machine Learning.  
> Diseñado para la enseñanza de IA en cursos de ingeniería.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelSkrauba/visual-ml/blob/main/notebooks/visual_ml_colab.ipynb)
[![CI](https://github.com/AxelSkrauba/visual-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/AxelSkrauba/visual-ml/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ¿Qué es Visual ML?

Visual ML es una herramienta pedagógica interactiva que permite explorar intuitivamente cómo los modelos clásicos de ML separan las clases en un espacio de dos features. Visualiza en tiempo real cómo cambia la **frontera de decisión** al modificar hiperparámetros, tipo de dataset y nivel de ruido.

### Conceptos que se pueden explorar
- **Overfitting y underfitting** — observa la brecha train/test en métricas y frontera.
- **Complejidad del modelo** — desde fronteras lineales hasta espirales complejas.
- **Impacto de hiperparámetros** — C en SVM, max_depth en Decision Tree, K en KNN, capas en MLP, etc.
- **Comparación de algoritmos** — hasta 6 modelos lado a lado sobre el mismo dataset.
- **Desbalance de clases** — efecto en accuracy vs F1 con dataset configurable.

---

## Modelos incluidos

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

## Datasets de juguete

| Dataset | Descripción | Clases |
|---|---|---|
| **Moons** | Dos lunas entrelazadas | 2 |
| **Circles** | Círculos concéntricos | 2 |
| **Blobs** | Clusters gaussianos | 2–5 |
| **XOR** | Patrón XOR en cuadrantes | 2 |
| **Linear** | Linealmente separable | 2–5 |
| **Spirals** | Espirales entrelazadas | 2–5 |
| **Desbalanceado** | Clases gaussianas con ratio de minoría configurable (5–50%) | 2 |

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

La suite de tests cubre todos los módulos core con TDD (150+ tests):
- `test_datasets.py` — generación, shapes, reproducibilidad, multi-clase, dataset desbalanceado.
- `test_models.py` — registro, metadatos, defaults, instanciación, MLP string parsing.
- `test_train.py` — métricas, rangos válidos, señales pedagógicas (7 escenarios).
- `test_visualization.py` — figuras correctas para todos los modelos/temas.

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
│   ├── datasets.py             # Generadores de datasets parametrizables (7 tipos)
│   ├── models.py               # Registro declarativo de modelos (11 modelos)
│   ├── train.py                # Entrenamiento + evaluación + señales pedagógicas
│   └── visualization.py        # Frontera de decisión, CM, comparación
├── ui/
│   ├── sidebar.py              # Controles interactivos
│   ├── single_view.py          # Tab: exploración de un modelo
│   └── compare_view.py         # Tab: comparación multi-modelo
├── notebooks/
│   └── visual_ml_colab.ipynb   # Notebook para Google Colab
└── tests/
    ├── conftest.py
    ├── test_datasets.py
    ├── test_models.py
    ├── test_train.py
    └── test_visualization.py
```

---

## Interfaz

### Tab: Explorar Modelo
- Sidebar: selección de dataset (7), modelo (11) e hiperparámetros dinámicos.
- Panel izquierdo: frontera de decisión con puntos train y test.
- Panel derecho: tabla de métricas con semáforo, matriz de confusión (RdBu divergente), classification report parseado.
- Señales pedagógicas automáticas: overfitting, underfitting, desbalance, memorización.

### Tab: Comparar Modelos
- Grid de fronteras de decisión para múltiples modelos simultáneos.
- Gráfico de barras comparativo (Accuracy y F1 — train vs test).
- Tabla resumen con badges de estado.

### Temas
- **Oscuro** (por defecto): ideal para pantallas.
- **Claro**: recomendado para proyección en aula.

---

## Autor

**Axel Skrauba** · [@AxelSkrauba](https://github.com/AxelSkrauba)

---

## Licencia

MIT — libre para uso académico y educativo.
