# Detector de Piletas con YOLO y Streamlit

Aplicacion web para detectar piletas en imagenes usando un modelo YOLO entrenado y Streamlit.

## Problema mas probable

Tu entorno virtual actual fue creado con **Python 3.14.5**. Muchas librerias de vision artificial, especialmente `torch`, `torchvision`, `opencv` y `ultralytics`, pueden fallar si se instalan con una version de Python demasiado nueva o si se fijan versiones viejas.

Para este proyecto conviene usar **Python 3.11** o **Python 3.12**.

## Instalacion recomendada en Windows

Si al ejecutar `py -3.11 -m venv venv` aparece este error:

```text
No suitable Python runtime found
```

significa que Python 3.11 no esta instalado en tu maquina. Puedes instalarlo con una de estas opciones.

### Opcion A: instalar Python 3.11 con winget

```powershell
winget install -e --id Python.Python.3.11
```

Cierra y abre de nuevo la terminal, y verifica:

```powershell
py -0p
```

Despues crea el entorno:

```powershell
py -3.11 -m venv venv
```

### Opcion B: instalar Python 3.11 manualmente

1. Descarga Python 3.11 desde:

   <https://www.python.org/downloads/release/python-3119/>

2. Durante la instalacion, marca la opcion **Add python.exe to PATH**.

3. Cierra y abre de nuevo la terminal.

4. Verifica que Windows lo detecte:

   ```powershell
   py -0p
   ```

## Crear e instalar el entorno

Desde esta carpeta del proyecto, borra el entorno viejo si no lo necesitas:

```powershell
Remove-Item -Recurse -Force .\venv
```

Crea un entorno nuevo con Python 3.11:

```powershell
py -3.11 -m venv venv
```

Activa el entorno:

```powershell
.\venv\Scripts\Activate.ps1
```

Actualiza pip:

```powershell
python -m pip install --upgrade pip
```

Instala las dependencias:

```powershell
pip install -r requirements.txt
```

Ejecuta la app:

```powershell
streamlit run app.py
```

## Alternativa si quieres seguir con Python 3.14

Como el `requirements.txt` ya no fija versiones viejas de `torch` y `numpy`, tambien puedes probar con el entorno actual de Python 3.14:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Si esta alternativa falla, usa Python 3.11.

La aplicacion se abrira normalmente en:

```text
http://localhost:8501
```

## Estructura esperada

```text
deploy_piletas/
+-- app.py
+-- requirements.txt
+-- README.md
\-- modelo/
    \-- best.pt
```

## Notas

- El archivo `modelo/best.pt` debe ser el modelo entrenado con YOLO.
- Si el modelo fue entrenado con Ultralytics YOLOv8/YOLOv11, la carga con `YOLO("modelo/best.pt")` deberia funcionar.
- Si usaste otra familia de YOLO distinta a Ultralytics, puede hacer falta adaptar el codigo de inferencia.
