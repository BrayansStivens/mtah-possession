"""Red totalmente convolucional: pila de 5 canales -> mapa de calor del balon a media resolucion.

Pequena a proposito (~180 k parametros, entrenada desde cero, sin pesos preentrenados ni sus
licencias): el balon mide ~7 px y lo que lo delata es local -- una mancha clara, redonda,
que se mueve respecto al cesped una vez compensada la camara. Se entrena con recortes y se
ejecuta sobre el frame entero.

La primera convolucion ya tiene paso 2 y la salida sale a media resolucion (OUTPUT_STRIDE):
a resolucion completa la red iba a 18 fps en la RTX 3050 (7 h el partido); asi, ~4x menos
operaciones. Un pixel de salida son 2 px de imagen, y la posesion necesita ~30 px.
"""

import torch
from torch import nn
from torch.nn import functional as F

OUTPUT_STRIDE = 2


def _conv(cin, cout, dilation=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    def __init__(self, channels=5, width=24):
        super().__init__()
        w1, w2, w3 = width, 2 * width, 4 * width
        self.enc1 = nn.Sequential(_conv(channels, w1, stride=2), _conv(w1, w1))   # 1/2
        self.enc2 = nn.Sequential(_conv(w1, w2), _conv(w2, w2))                   # 1/4
        self.enc3 = nn.Sequential(_conv(w2, w3), _conv(w3, w3), _conv(w3, w3, 2),
                                  _conv(w3, w3, 4))                               # 1/8
        self.dec2 = _conv(w3 + w2, w2)
        self.dec1 = _conv(w2 + w1, w1)
        self.head = nn.Conv2d(w1, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)      # arranca diciendo "no hay balon"

    def forward(self, x):
        x = x / 255.0 - 0.5
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.head(d1)[:, 0]                   # logits (N, H/2, W/2)


def to_output(pixel):
    """Coordenada de imagen -> coordenada en la rejilla de salida."""
    return (pixel - 0.5 * (OUTPUT_STRIDE - 1)) / OUTPUT_STRIDE


def to_image(cell):
    """Celda de la rejilla de salida -> centro en pixel de imagen."""
    return cell * OUTPUT_STRIDE + 0.5 * (OUTPUT_STRIDE - 1)


def focal_heatmap_loss(logits, target, alpha=2.0, beta=4.0):
    """Perdida focal de CenterNet sobre mapas gaussianos. target==1 marca el pico.

    Normalizada por el numero de picos; un recorte sin balon solo aporta negativos.
    """
    probability = torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)
    peak = target >= 0.999
    positive = -((1 - probability) ** alpha) * torch.log(probability) * peak
    negative = (-((1 - target) ** beta) * (probability ** alpha)
                * torch.log(1 - probability) * ~peak)
    return (positive.sum() + negative.sum()) / peak.sum().clamp(min=1)
