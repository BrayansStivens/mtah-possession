"""Carga de datos externos etiquetados, para entrenar o calibrar el modelo.

**Este paquete no esta en la ruta de reproduccion.** RULES.md exige que ejecutar el repo
contra los datos del partido reproduzca el CSV entregado, no aproximadamente sino el mismo
fichero, y por eso `requirements.txt` solo tiene numpy y pandas pinneados. Las dependencias
de entrenamiento (kloppy) viven en `requirements-train.txt` y se importan unicamente desde
aqui. `python -m mtah build` no toca nada de este paquete.
"""
