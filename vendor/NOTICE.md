# Codigo de terceros

`evaluation/` y `games.py` son copia literal del repositorio oficial de la competicion:

    https://github.com/metrica-sports/more-than-a-hack-2027

Licencia MIT, incluida en `LICENSE.metrica`. Copyright de Metrica Sports y FC Barcelona
Innovation Hub.

Se vendoriza sin modificar para que este repositorio pueda puntuarse en local con **el mismo
scorer** que decide la competicion, sin depender de tener el repo oficial clonado al lado.
No lo edites: si hay que cambiar algo, el bug esta en nuestro codigo.

`evaluation/gt/` contiene el ground truth **publico** (5 minutos), que se distribuye con ese
repositorio. Los assets del partido (video, tracking, homografia) NO estan aqui ni pueden estarlo.
