"""Detector de balon sobre el video tactico (plan-detector-balon.md).

Las etiquetas salen del propio ATD: donde el tracker de Metrica ve el balon, la homografia
lo proyecta al pixel. El detector se entrena con esos frames y se ejecuta donde el tracker
lo pierde. Ninguna etiqueta humana en ninguna parte.

Regla de la competicion que este paquete toca de lleno: ni frames, ni recortes, ni
coordenadas del partido salen a un LLM en la nube. Todo se mide con agregados.

Dependencias de vision (torch, opencv) en requirements-vision.txt, fuera de la ruta de
`python -m mtah build` mientras el detector no entre en la entrega.
"""
