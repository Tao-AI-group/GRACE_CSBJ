# GRACE Project

This is the GRACE project using Conda for environment management.

## 🧰 Environment Setup

- **Python version**: 3.12
- **Environment name**: `chatbot`

---

## 🔧 Setup Instructions

1. **Clone the repository** :

    ```bash
    cd GRACE_CSBJ
    ```

2. **Create a Conda environment** named `GRACE` with Python 3.12:

    ```bash
    conda create -n GRACE python=3.12
    ```

3. **Activate the environment**:

    ```bash
    conda activate GRACE
    ```

4. **Install required packages** from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Run the Project

After installing dependencies, you can run the project:

```bash
cd GRACE_CSBJ
streamlit ./frontend/GRACE.py